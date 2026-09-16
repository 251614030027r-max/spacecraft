"""SAC-MPC coupling: the learned layer chooses where, the optimiser flies there.

The measured gap this exists to close is the *entry window*. On twelve seeds a
fixed-setpoint MPC failed four times at horizon 20 and eleven times at horizon
50, and every one of those fifteen failures was the same event: an illegal
crossing of the 6 m entry plane that never re-latched, with zero
truth-geometry violations of any kind. Afterwards the chaser flew to the
desired pose anyway and parked on it -- median distance to the goal when the
clock ran out was 0.000 m. A receding-horizon regulator has no term for "that
crossing was the last one that counted", so it cannot decide to back out and
come round with the tumble. That decision is what the learned layer supplies.
See ``docs/PRECAPTURE_ENTRY_WINDOW_GAP.md``.

The interface is the one already declared, with the frame corrected: a single
3D waypoint in the **target body frame**, held for the decision period. Under
the older inertially oriented reading the same channel could not even carry
the task's own desired pose -- 0/3 completions against 3/3 direct -- because a
body-fixed goal has to be re-issued as a rotating inertial point, which the
horizon map turns into a circular reference. In the target frame the channel
is bitwise transparent. See
``docs/PRECAPTURE_REFERENCE_TRACKING_DIAGNOSIS.md``.

Three properties are deliberate, because each one is a place this could have
been made to look better than it is:

* **One architecture for the whole mission.** The MPC is the only thing that
  ever touches the actuators, from 17 m to contact. There is no phase switch
  and no hand-off, so a result cannot be confounded with one.
* **The physical reward terms are summed over the decision period.** Time,
  force, torque, safety and event terms keep their micro-step integral. The
  potential term is evaluated once at the 2 s decision boundary with the SAC
  discount, so it telescopes at the level the policy actually optimises.
  Nothing here rewards entering legally, waiting, or approaching a window.
* **The action is an absolute waypoint, not a displacement.** A displacement
  parametrisation would bias the policy toward moving, and "hold position and
  let the target turn" is exactly one of the two answers the task poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from controllers.mpc import MPCController
from controllers.mpc.config import MPCConfig, precapture_mpc_config
from controllers.mpc.prediction import (
    LocalRelativePredictionModel,
    RelativePredictionModel,
    relative_to_vector,
)
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.phase2_env import precapture_planning_environment_config
from env.reward import PrecaptureReward
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.task import compute_precapture_metrics

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PrecaptureHybridConfig:
    """Everything the coupling adds on top of the frozen precapture task."""

    #: Control steps the learned waypoint is held for. 20 steps is 2.0 s, and
    #: it must equal the controller's ``external_reference_hold_steps`` so the
    #: optimiser adopts a new waypoint exactly on a decision boundary.
    decision_period_steps: int = 20
    #: Discount used by the upper policy and therefore by the potential term
    #: evaluated at one complete decision boundary.
    decision_discount_factor: float = 0.99
    #: ``"absolute"`` maps the action straight onto a target-frame point,
    #: ``a * waypoint_scale_m``. It is the parametrisation the first training
    #: attempt used and it is kept as the default so that run stays
    #: reproducible, but it is **measured to be unlearnable**: sampled
    #: uniformly, 70% of that box commands a point beyond 15 m -- outward,
    #: away from the target -- only 1.94% commands anything inside the 6 m
    #: entry sphere, and 0.24% lands within 3 m of the desired pose. A policy
    #: exploring it spends most of its decisions pushing the chaser out of the
    #: episode, and 400 episodes of SAC moved neither reward (-27.00 to
    #: -27.74) nor episode length (48.0 to 49.9).
    #:
    #: ``"radial_local"`` names the same kind of point -- an absolute
    #: target-frame waypoint, still not a displacement -- in coordinates built
    #: from where the chaser currently is: one radial component and a
    #: three-component lateral nudge, so the action is 4D while the *interface*
    #: to the MPC is the unchanged 3D waypoint. Every action in the box is then
    #: a sane command. The zero action commands exactly the chaser's present
    #: target-frame point, which is the co-rotating hold; the inertially frozen
    #: hold -- the one that actually opens an entry window -- is the small
    #: lateral offset that undoes ``omega * 2 s = 4.7 deg`` per decision.
    #: **This biases exploration toward holding.** That is stated rather than
    #: hidden: the alternative bias, which the absolute box has, is 70% toward
    #: flying out, and it was measured to be fatal.
    #: ``"arrival_condition"`` is the T6 interface. The action is 2D and names
    #: an *arrival condition* rather than a free point: how far along the
    #: commit is (a blend between an inertially frozen hold and the desired
    #: pose) and at what radius the hold sits. Both channels are the two
    #: variables T2 actually measured -- commit timing was the only variable
    #: that rescued a persistently failing seed, and the hold radius was the
    #: only one that moved fuel (-20.6% on a matched cell). Lateral adjustment
    #: rescued nothing in T4 and is therefore not given a channel.
    #:
    #: ``a = (+1, *)`` commands the desired pose on every decision, which the
    #: external channel carries as a constant broadcast -- the fixed-setpoint
    #: Pure MPC row, reproduced bitwise. ``a = (-1, r)`` holds the inertially
    #: frozen point, which is what "wait for the target to turn" has to mean
    #: (a fixed *target-frame* point co-rotates and no window ever opens).
    #: ``a = 0`` is the midpoint: a staging point at intermediate radius, a
    #: sane exploratory default rather than a degenerate one.
    waypoint_parametrization: str = "absolute"
    #: ``a = +-1`` scales the commanded radius by ``exp(+-0.7)``, about
    #: double or half per decision. Reaching 3 m from a 17 m start therefore
    #: takes three sustained decisions rather than one jump -- which is also
    #: what the optimiser can actually fly in 2 s.
    radial_action_gain: float = 0.7
    #: A full-scale lateral nudge tilts the commanded direction by about 41 deg.
    lateral_action_gain: float = 0.5
    #: ``arrival_condition`` only. ``a[1] = +-1`` scales the hold radius by
    #: ``exp(+-0.35)`` about the radius the hold point was frozen at, so the
    #: channel spans roughly the 12-18 m band T2 scanned from a 15-20 m start.
    hold_radius_action_gain: float = 0.35
    #: ``arrival_condition`` only. The commit blend may only advance:
    #: ``b_k = max(b_{k-1}, b_k_commanded)``.
    #:
    #: The blend spans a fixed 13 m of commanded radius -- the hold freezes at
    #: 12-19 m and the desired pose sits at 3 m -- so ``d r_ref / d a_commit``
    #: is about -6.5 m per unit of action wherever the policy sits. A chaser
    #: can move about 0.094 m in one 2 s decision from rest, so ordinary
    #: policy variation of 0.09 re-aims the setpoint six times further than it
    #: can be followed. Measured: every episode that completed held the
    #: waypoint to a median 0.000-0.015 m per decision at 0.20-0.28 actuator
    #: usage, and every episode that timed out moved it 0.27-0.62 m at
    #: 0.84-0.95, closing at a third of the rate. Warping the blend cannot fix
    #: that -- the 13 m of travel is fixed, warping only moves where the gain
    #: sits -- and a rate limit sized to the authority would need 138
    #: decisions to reach the commit, which is 276 s of a 300 s episode.
    #:
    #: What the traces show is oscillation, not travel: the dead seeds walk
    #: ``a_commit`` up and down and spend the authority reversing. The ratchet
    #: removes exactly that, with no tunable constant, while leaving a commit
    #: reachable in a single decision. ``a = (+1, *)`` still commands the
    #: desired pose on every decision, so the bitwise Pure MPC floor is
    #: unaffected, and holding stays available for as long as the policy keeps
    #: ``a_commit`` low.
    monotone_commit: bool = True
    #: ``arrival_condition`` only. Interpret the policy action as a *residual*
    #: on the nominal arrival action rather than an absolute arrival condition.
    #: The nominal is ``(a_commit, a_radius) = (+1, 0)`` -- the fixed-setpoint
    #: Pure MPC command -- and the executed candidate is
    #: ``clip(nominal + residual)``. A zero residual therefore recovers the
    #: nominal desired-pose command bitwise; the policy only ever learns *how
    #: far to pull back from full commit*, which is the direct answer to T12's
    #: negative transfer (a learned layer that rewrote the whole reference
    #: destroyed states the nominal already solved).
    baseline_anchored_residual: bool = False
    #: The action is ``a * waypoint_scale_m`` in the target body frame, so the
    #: box reaches past the 17-20 m start without reaching the 30 m distance
    #: failure. It is an absolute point, not a displacement.
    waypoint_scale_m: float = 18.0
    #: Radial clip applied after scaling. The lower bound keeps the commanded
    #: point outside the 2 m keepout by a margin the optimiser can hold; the
    #: upper bound keeps it inside the episode's distance limit.
    minimum_waypoint_radius_m: float = 2.5
    maximum_waypoint_radius_m: float = 25.0
    horizon_steps: int = 20
    terminal_weight: float = 1000.0
    input_weight: float = 0.01
    #: Post-solve rollout, truth-margin and predicted-cost reporting.  It never
    #: participates in the QP or selected command; training disables it while
    #: evaluation/profiling chooses explicitly whether those diagnostics matter.
    runtime_diagnostics: bool = True
    #: V4 single factor: expose the target's absolute attitude phase and the
    #: remaining episode budget to the scheduling policy.  The first two
    #: inertial columns of the target rotation are a continuous, non-redundant
    #: 6D rotation representation; the seventh value is remaining time in
    #: [0, 1].  It changes only the upper policy observation.
    include_target_phase_and_time_observation: bool = False

    def __post_init__(self) -> None:
        if self.decision_period_steps < 1:
            raise ValueError("decision period must be at least one control step")
        if not 0.0 < self.decision_discount_factor <= 1.0:
            raise ValueError("decision discount factor must lie in (0, 1]")
        if self.waypoint_scale_m <= 0.0:
            raise ValueError("waypoint scale must be positive")
        if not (
            0.0
            < self.minimum_waypoint_radius_m
            < self.maximum_waypoint_radius_m
        ):
            raise ValueError("waypoint radius bounds must be ordered and positive")
        if self.horizon_steps < 1:
            raise ValueError("horizon must be at least one step")
        if self.waypoint_parametrization not in {
            "absolute",
            "radial_local",
            "arrival_condition",
        }:
            raise ValueError("unknown waypoint parametrisation")
        if min(self.radial_action_gain, self.lateral_action_gain) <= 0.0:
            raise ValueError("action gains must be positive")
        if (
            self.baseline_anchored_residual
            and self.waypoint_parametrization != "arrival_condition"
        ):
            raise ValueError(
                "baseline_anchored_residual requires the arrival_condition "
                "parametrisation"
            )

    #: The T6 reverse channel: three bounded summaries of how the lower layer
    #: coped with the previous decision -- fallback fraction, peak solved-step
    #: slack and peak actuator usage. All three are free: the first two fall
    #: out of the QP solve itself and the third is the commanded wrench. The
    #: predicted-margin diagnostic is deliberately *not* here; it needs a full
    #: horizon rollout that the deployment path does not pay for.
    include_execution_feedback_observation: bool = False

    @property
    def action_dimension(self) -> int:
        if self.waypoint_parametrization == "absolute":
            return 3
        if self.waypoint_parametrization == "arrival_condition":
            return 2
        return 4


#: The nominal ``arrival_condition`` action: full commit to the desired pose,
#: which reaches the optimiser as the constant broadcast that is the
#: fixed-setpoint Pure MPC row. Residual anchoring is defined relative to it.
_NOMINAL_ARRIVAL_ACTION = np.array([1.0, 0.0], dtype=np.float64)


def _slerp(start: FloatArray, goal: FloatArray, weight: float) -> FloatArray:
    """Constant-rate interpolation between two unit directions."""

    cosine = float(np.clip(start @ goal, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-8:
        return goal if weight >= 1.0 else start
    sine = float(np.sin(angle))
    return (
        float(np.sin((1.0 - weight) * angle)) * start
        + float(np.sin(weight * angle)) * goal
    ) / sine


def hybrid_mpc_config(
    hybrid: PrecaptureHybridConfig,
    environment: SE3RendezvousConfig,
) -> MPCConfig:
    """The lower layer, pinned to the corrected channel and the measured mode."""

    return replace(
        precapture_mpc_config(),
        horizon_steps=hybrid.horizon_steps,
        outer_iterations=1,
        input_weight=hybrid.input_weight,
        terminal_weight=hybrid.terminal_weight,
        reference_source="external_local",
        # The frame the channel is lossless in. Under "inertial" the same
        # channel cannot carry a body-fixed point at all.
        external_reference_frame="target",
        # Measured best of three formulations; see the diagnosis document.
        precapture_attitude_reference="frozen",
        external_reference_hold_steps=hybrid.decision_period_steps,
        runtime_diagnostics=hybrid.runtime_diagnostics,
    )


class PrecaptureHybridEnv(gym.Env[np.ndarray, np.ndarray]):
    """One learned waypoint per decision period, flown by the constrained MPC.

    The observation is the underlying environment's own, unchanged, so the
    learned layer sees exactly what Pure SAC sees. Time, force, torque, safety
    and event rewards are summed over the decision period. Micro-step potential
    shaping is removed and replaced by
    ``w * (gamma_decision * Phi(s_20) - Phi(s_0))``. The episode is therefore
    about 150 consistently discounted decisions instead of about 950 control
    steps, while the physical path costs retain their original integrals.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment_config: SE3RendezvousConfig | None = None,
        hybrid_config: PrecaptureHybridConfig | None = None,
    ) -> None:
        self.hybrid_config = hybrid_config or PrecaptureHybridConfig()
        self.environment_config = (
            environment_config or precapture_planning_environment_config()
        )
        if not self.environment_config.precapture_planning_enabled:
            raise ValueError("the hybrid requires the precapture planning task")
        self.env = SE3RendezvousEnv(self.environment_config)
        self._potential_evaluator = PrecaptureReward(
            task=self.environment_config.precapture_task,
            settings=self.environment_config.precapture_reward,
            time_step_s=self.environment_config.dt_s,
        )
        self.mpc_config = hybrid_mpc_config(
            self.hybrid_config, self.environment_config
        )
        reference_model = RelativePredictionModel(
            target_parameters=target_parameters(),
            chaser_parameters=chaser_parameters(),
            dt_s=self.environment_config.dt_s,
            gravity_options=GravityOptions(
                include_j2=self.environment_config.include_j2
            ),
            solver_settings=RK45Settings(
                rtol=self.environment_config.solver_rtol,
                atol=self.environment_config.solver_atol,
                max_step=self.environment_config.dt_s,
            ),
        )
        self.controller = MPCController(
            self.mpc_config,
            LocalRelativePredictionModel(
                reference_model.chaser_parameters, self.environment_config.dt_s
            ),
            reference_model,
        )
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=(self.hybrid_config.action_dimension,),
            dtype=np.float32,
        )
        base = self.env.observation_space
        assert isinstance(base, spaces.Box)
        low, high = base.low, base.high
        if self.hybrid_config.include_target_phase_and_time_observation:
            low = np.concatenate((low, -np.ones(6), np.zeros(1)))
            high = np.concatenate((high, np.ones(7)))
        if self._ratchet_observation_active():
            low = np.concatenate((low, np.zeros(1)))
            high = np.concatenate((high, np.ones(1)))
        if self.hybrid_config.include_execution_feedback_observation:
            low = np.concatenate((low, np.zeros(3)))
            high = np.concatenate((high, np.ones(3)))
        if (
            self.hybrid_config.include_target_phase_and_time_observation
            or self._ratchet_observation_active()
            or self.hybrid_config.include_execution_feedback_observation
        ):
            self.observation_space = spaces.Box(
                low=low.astype(np.float32),
                high=high.astype(np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = self.env.observation_space
        self._last_info: dict[str, Any] = {}
        self._hold_inertial: FloatArray | None = None
        self._hold_radius_m = 0.0
        self._commit_blend = 0.0
        self._feedback = np.zeros(3, dtype=np.float64)

    def _ratchet_observation_active(self) -> bool:
        return (
            self.hybrid_config.waypoint_parametrization == "arrival_condition"
            and self.hybrid_config.monotone_commit
        )

    def _policy_observation(self, observation: np.ndarray) -> np.ndarray:
        """Append the scheduling state and the lower layer's execution feedback.

        The canonical 24D core is preserved untouched in every configuration,
        so a run with both flags off is bitwise the pre-T6 observation.
        """

        parts = [np.asarray(observation, dtype=np.float64)]
        if self.hybrid_config.include_target_phase_and_time_observation:
            assert self.env.target_state is not None
            parts.append(self.env.target_state.rotation[:, :2].reshape(-1))
            parts.append(
                np.array(
                    [
                        np.clip(
                            (
                                self.environment_config.max_time_s
                                - self.env.time_seconds
                            )
                            / self.environment_config.max_time_s,
                            0.0,
                            1.0,
                        )
                    ]
                )
            )
        if self._ratchet_observation_active():
            # The ratchet carries state between decisions: the same action
            # commands a different waypoint depending on how far the commit has
            # already advanced. Without it in the observation the decision
            # problem is no longer Markov, and the policy would be guessing at
            # its own past. It is a scalar in [0, 1], not a tuning knob.
            parts.append(np.array([self._commit_blend], dtype=np.float64))
        if self.hybrid_config.include_execution_feedback_observation:
            parts.append(self._feedback)
        if len(parts) == 1:
            return observation
        augmented = np.concatenate(parts).astype(np.float32)
        if not self.observation_space.contains(augmented):
            raise RuntimeError("invalid augmented policy observation")
        return augmented

    def _current_reward_potential(self) -> float:
        assert self.env.target_state is not None
        assert self.env.chaser_state is not None
        assert self.env.relative is not None
        metrics = compute_precapture_metrics(
            self.env.target_state,
            self.env.chaser_state,
            self.env.relative,
            self.environment_config.precapture_task,
            terminal_region_active=bool(
                self._last_info.get("terminal_region_active", False)
            ),
        )
        return self._potential_evaluator.potential(metrics)

    # -- waypoint ---------------------------------------------------------

    def _current_position(self) -> FloatArray:
        from dynamics.lie import se3_exp

        assert self.env.relative is not None
        return se3_exp(relative_to_vector(self.env.relative)[:6])[:3, 3]

    def waypoint_from_action(self, action: np.ndarray) -> FloatArray:
        """Map a bounded action to a target-body-frame point.

        Absolute under both parametrisations: the policy names a place to be,
        and "stay out here while the target turns" is as expressible as "close
        on the port". The radial clip is a safety envelope on the *command*,
        not on the vehicle -- the MPC's own constraints remain the only thing
        that keeps the trajectory legal, and the truth geometry remains the
        only thing that adjudicates it.
        """

        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw.size != self.hybrid_config.action_dimension:
            raise ValueError("action has the wrong dimension")
        if not np.all(np.isfinite(raw)):
            raise ValueError("action must be finite")
        raw = np.clip(raw, -1.0, 1.0)
        if self.hybrid_config.waypoint_parametrization == "arrival_condition":
            if self.hybrid_config.baseline_anchored_residual:
                # ``raw`` is the residual on the nominal arrival action; a zero
                # residual recovers the nominal desired-pose command bitwise.
                raw = np.clip(_NOMINAL_ARRIVAL_ACTION + raw, -1.0, 1.0)
            return self._arrival_condition_waypoint(raw)
        if self.hybrid_config.waypoint_parametrization == "radial_local":
            return self._radial_local_waypoint(raw)
        return self._clip_radius(raw * self.hybrid_config.waypoint_scale_m)

    def action_for_waypoint(self, waypoint: np.ndarray) -> FloatArray:
        """Return the closest bounded action that names an absolute waypoint."""

        target = np.asarray(waypoint, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(target)):
            raise ValueError("waypoint must be finite")
        if self.hybrid_config.waypoint_parametrization == "arrival_condition":
            desired = np.asarray(
                self.environment_config.precapture_task.desired_position,
                dtype=np.float64,
            )
            base = (
                np.array([1.0, 0.0])
                if bool(np.allclose(target, desired))
                else np.array([-1.0, 0.0])
            )
            if self.hybrid_config.baseline_anchored_residual:
                # Residual that reproduces ``base``: exact for the nominal
                # desired-pose command (-> zero residual), best-effort clip for
                # the hold. The desired-pose control only ever asks for the
                # former, so its fixed-setpoint floor stays bitwise.
                return np.clip(base - _NOMINAL_ARRIVAL_ACTION, -1.0, 1.0)
            return base
        if self.hybrid_config.waypoint_parametrization == "absolute":
            return np.clip(
                target / self.hybrid_config.waypoint_scale_m, -1.0, 1.0
            )
        current = self._current_position()
        radius = float(np.linalg.norm(current))
        goal_radius = float(np.linalg.norm(target))
        if radius < 1.0e-9 or goal_radius < 1.0e-9:
            return np.zeros(4, dtype=np.float64)
        direction = current / radius
        radial = float(
            np.clip(
                np.log(goal_radius / radius)
                / self.hybrid_config.radial_action_gain,
                -1.0,
                1.0,
            )
        )
        goal_direction = target / goal_radius
        cosine = float(goal_direction @ direction)
        perpendicular = goal_direction - cosine * direction
        if cosine > 1.0e-3:
            nudge = perpendicular / (
                self.hybrid_config.lateral_action_gain * cosine
            )
        else:
            norm = float(np.linalg.norm(perpendicular))
            nudge = perpendicular / norm if norm > 1.0e-9 else np.zeros(3)
        return np.clip(np.concatenate(([radial], nudge)), -1.0, 1.0)

    def _arrival_condition_waypoint(self, action: FloatArray) -> FloatArray:
        """Map ``(a_commit, a_radius)`` onto a target-frame arrival condition.

        ``a_commit`` blends between two references that are geometrically
        different objects, not two ends of one continuum:

        * the **inertially frozen hold**, re-expressed in the target frame on
          every decision. The approach axis is body-fixed, so a chaser holding
          a fixed *target-frame* point co-rotates with it and the entry
          geometry never changes -- there is no window to wait for, only fuel
          to spend. Waiting has to be inertial, and a scripted policy that got
          this wrong is on the record.
        * the **desired pose**, a constant in the target frame. Commanded on
          every decision it reaches the optimiser as a constant broadcast,
          which is the fixed-setpoint Pure MPC row.

        ``a_radius`` sets how far out the hold sits, about the radius the hold
        direction was frozen at. It has no effect once the commit is complete,
        which is the honest statement of what T2 measured: the radius bought
        fuel on cells that already completed, not new rescues.
        """

        assert self.env.target_state is not None
        position = self._current_position()
        rotation = self.env.target_state.rotation
        if self._hold_inertial is None:
            inertial = rotation @ position
            norm = float(np.linalg.norm(inertial))
            if norm < 1.0e-9:
                inertial, norm = np.array([1.0, 0.0, 0.0]), 1.0
            self._hold_inertial = inertial / norm
            self._hold_radius_m = float(np.linalg.norm(position))
        blend = 0.5 * (float(action[0]) + 1.0)
        if self.hybrid_config.monotone_commit:
            blend = max(self._commit_blend, blend)
            self._commit_blend = blend
        hold_radius = float(
            np.clip(
                self._hold_radius_m
                * float(
                    np.exp(
                        self.hybrid_config.hold_radius_action_gain * float(action[1])
                    )
                ),
                self.hybrid_config.minimum_waypoint_radius_m,
                self.hybrid_config.maximum_waypoint_radius_m,
            )
        )
        commit_point = np.asarray(
            self.environment_config.precapture_task.desired_position,
            dtype=np.float64,
        )
        if blend >= 1.0:
            # Exactly the desired pose, so the channel carries a constant and
            # the reference is the fixed-setpoint broadcast bitwise.
            return commit_point
        # Interpolate direction and radius separately rather than the two
        # points. A straight chord between a 17 m off-axis hold and the 3 m
        # on-axis pose is dominated by the hold end: at the blend that gives a
        # 7.5 m radius the direction is still mostly the hold direction. The
        # offline feasibility plan that solves the hardest seed stages *on the
        # approach axis* at 7.5 m, so the intermediate references have to swing
        # towards the axis as they close, which is what a direction slerp does
        # and a chord does not.
        commit_radius = float(np.linalg.norm(commit_point))
        hold_direction = rotation.T @ self._hold_inertial
        commit_direction = commit_point / commit_radius
        direction = _slerp(hold_direction, commit_direction, blend)
        radius = (1.0 - blend) * hold_radius + blend * commit_radius
        return self._clip_radius(radius * direction)

    def _radial_local_waypoint(self, action: FloatArray) -> FloatArray:
        position = self._current_position()
        radius = float(np.linalg.norm(position))
        if radius < 1.0e-9:
            return np.array(
                [self.hybrid_config.minimum_waypoint_radius_m, 0.0, 0.0]
            )
        direction = position / radius
        commanded_radius = float(
            np.clip(
                radius
                * float(np.exp(self.hybrid_config.radial_action_gain * action[0])),
                self.hybrid_config.minimum_waypoint_radius_m,
                self.hybrid_config.maximum_waypoint_radius_m,
            )
        )
        nudge = np.asarray(action[1:4], dtype=np.float64)
        # Project the nudge into the plane perpendicular to the current
        # direction. Doing it this way rather than through a chosen basis keeps
        # the map continuous everywhere -- a basis picked from "the least
        # aligned axis" would flip meaning across a switching surface.
        lateral = nudge - float(nudge @ direction) * direction
        tilted = direction + self.hybrid_config.lateral_action_gain * lateral
        norm = float(np.linalg.norm(tilted))
        if norm < 1.0e-9:
            tilted, norm = direction, 1.0
        return commanded_radius * (tilted / norm)

    def _clip_radius(self, point: FloatArray) -> FloatArray:
        radius = float(np.linalg.norm(point))
        if radius < 1.0e-9:
            # A degenerate command would ask for the target's own centre; hold
            # the current relative position instead of inventing a direction.
            point = self._current_position()
            radius = float(np.linalg.norm(point))
            if radius < 1.0e-9:
                return np.array(
                    [self.hybrid_config.minimum_waypoint_radius_m, 0.0, 0.0]
                )
        clipped = float(
            np.clip(
                radius,
                self.hybrid_config.minimum_waypoint_radius_m,
                self.hybrid_config.maximum_waypoint_radius_m,
            )
        )
        return point * (clipped / radius)

    # -- gymnasium --------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self.controller.reset()
        self._last_info = dict(info)
        self._hold_inertial = None
        self._hold_radius_m = 0.0
        self._commit_blend = 0.0
        self._feedback = np.zeros(3, dtype=np.float64)
        return self._policy_observation(observation), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        waypoint = self.waypoint_from_action(action)
        initial_potential = self._current_reward_potential()
        integrated_reward_without_shaping = 0.0
        removed_micro_shaping = 0.0
        terminated = truncated = False
        observation = None
        info: dict[str, Any] = self._last_info
        control_steps = 0
        zero_fallbacks = 0
        valid_slacks: list[float] = []
        force_usage: list[float] = []
        torque_usage: list[float] = []
        solved_steps = 0
        solved_peak_slack = 0.0
        actuator_usage_sum = 0.0
        for _ in range(self.hybrid_config.decision_period_steps):
            assert self.env.relative is not None and self.env.target_state is not None
            state = relative_to_vector(self.env.relative)
            wrench, diagnostics = self.controller.command(
                state,
                target_state=self.env.target_state,
                time_seconds=self.env.time_seconds,
                terminal_latched=bool(info["terminal_region_active"]),
                external_reference=waypoint,
            )
            zero_fallbacks += int(diagnostics.used_zero_fallback)
            # Solver failures can leave a prior slack value behind. Never
            # interpret it as a current solution, including a fallback zero.
            if not diagnostics.used_zero_fallback:
                valid_slacks.append(float(diagnostics.maximum_slack))
            force_usage.append(float(np.max(np.abs(wrench[3:])) / self.environment_config.max_force_per_axis_n))
            torque_usage.append(float(np.max(np.abs(wrench[:3])) / self.environment_config.max_torque_per_axis_nm))
            # Read the slack only on steps that actually solved. On a fallback
            # step ``self._slack.value`` can still hold the previous solve's
            # numbers, so an unguarded read reports a stale margin as if it
            # described this step.
            if not diagnostics.used_zero_fallback:
                solved_steps += 1
                solved_peak_slack = max(
                    solved_peak_slack, float(diagnostics.maximum_slack)
                )
            # Mean, not peak. Peak saturates on essentially every decision --
            # sustained co-rotation already needs the full three-axis
            # authority at range -- so a peak channel would be a constant and
            # carry nothing the upper layer could act on.
            actuator_usage_sum += max(
                float(
                    np.max(
                        np.abs(wrench[:3])
                        / self.environment_config.max_torque_per_axis_nm
                    )
                ),
                float(
                    np.max(
                        np.abs(wrench[3:])
                        / self.environment_config.max_force_per_axis_n
                    )
                ),
            )
            observation, step_reward, terminated, truncated, info = self.env.step(
                wrench_to_normalized(
                    GeneralizedForce.from_vector(wrench),
                    max_torque_per_axis_nm=(
                        self.environment_config.max_torque_per_axis_nm
                    ),
                    max_force_per_axis_n=(
                        self.environment_config.max_force_per_axis_n
                    ),
                )
            )
            micro_shaping = float(info["reward_shaping"])
            removed_micro_shaping += micro_shaping
            integrated_reward_without_shaping += float(step_reward) - micro_shaping
            control_steps += 1
            if terminated or truncated:
                break
        assert observation is not None
        final_potential = float(info["reward_potential"])
        macro_shaping = self.environment_config.precapture_reward.potential_weight * (
            self.hybrid_config.decision_discount_factor * final_potential
            - initial_potential
        )
        reward = integrated_reward_without_shaping + macro_shaping
        info = dict(info)
        info["hybrid_waypoint_target_frame"] = [float(v) for v in waypoint]
        info["hybrid_waypoint_radius_m"] = float(np.linalg.norm(waypoint))
        info["hybrid_control_steps"] = control_steps
        info["hybrid_qp_zero_fallbacks"] = zero_fallbacks
        info["hybrid_feedback_fallback_fraction"] = zero_fallbacks / control_steps
        info["hybrid_feedback_valid_solve_steps"] = len(valid_slacks)
        info["hybrid_feedback_slack_max"] = max(valid_slacks) if valid_slacks else None
        info["hybrid_feedback_force_mean"] = float(np.mean(force_usage))
        info["hybrid_feedback_force_peak"] = max(force_usage)
        info["hybrid_feedback_torque_mean"] = float(np.mean(torque_usage))
        info["hybrid_feedback_torque_peak"] = max(torque_usage)
        fallback_fraction = (
            zero_fallbacks / control_steps if control_steps else 0.0
        )
        normalised_slack = (
            solved_peak_slack / self.mpc_config.constraint_slack_limit
            if solved_steps
            else 0.0
        )
        self._feedback = np.clip(
            np.array(
                [
                    fallback_fraction,
                    normalised_slack,
                    actuator_usage_sum / control_steps if control_steps else 0.0,
                ],
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )
        info["hybrid_feedback_fallback_fraction"] = float(self._feedback[0])
        info["hybrid_feedback_solved_peak_slack"] = float(self._feedback[1])
        info["hybrid_feedback_mean_actuator_usage"] = float(self._feedback[2])
        info["hybrid_feedback_solved_steps"] = solved_steps
        info["hybrid_reward_shaping"] = macro_shaping
        info["hybrid_removed_micro_shaping"] = removed_micro_shaping
        info["hybrid_integrated_reward_without_shaping"] = (
            integrated_reward_without_shaping
        )
        self._last_info = info
        return (
            self._policy_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self.env.close()


def make_precapture_hybrid_env(
    hybrid_config: PrecaptureHybridConfig | None = None,
) -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(hybrid_config=hybrid_config)


__all__ = [
    "PrecaptureHybridConfig",
    "PrecaptureHybridEnv",
    "hybrid_mpc_config",
    "make_precapture_hybrid_env",
]
