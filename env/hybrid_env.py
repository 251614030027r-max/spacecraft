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
* **The reward is the environment's own, summed over the decision period.**
  Nothing here rewards entering legally, waiting, or approaching a window.
  If the policy learns to time the entry it is because completing pays and
  being locked out does not.
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
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PrecaptureHybridConfig:
    """Everything the coupling adds on top of the frozen precapture task."""

    #: Control steps the learned waypoint is held for. 20 steps is 2.0 s, and
    #: it must equal the controller's ``external_reference_hold_steps`` so the
    #: optimiser adopts a new waypoint exactly on a decision boundary.
    decision_period_steps: int = 20
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
    waypoint_parametrization: str = "absolute"
    #: ``a = +-1`` scales the commanded radius by ``exp(+-0.7)``, about
    #: double or half per decision. Reaching 3 m from a 17 m start therefore
    #: takes three sustained decisions rather than one jump -- which is also
    #: what the optimiser can actually fly in 2 s.
    radial_action_gain: float = 0.7
    #: A full-scale lateral nudge tilts the commanded direction by about 41 deg.
    lateral_action_gain: float = 0.5
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

    def __post_init__(self) -> None:
        if self.decision_period_steps < 1:
            raise ValueError("decision period must be at least one control step")
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
        if self.waypoint_parametrization not in {"absolute", "radial_local"}:
            raise ValueError("unknown waypoint parametrisation")
        if min(self.radial_action_gain, self.lateral_action_gain) <= 0.0:
            raise ValueError("action gains must be positive")

    @property
    def action_dimension(self) -> int:
        return 3 if self.waypoint_parametrization == "absolute" else 4


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
    )


class PrecaptureHybridEnv(gym.Env[np.ndarray, np.ndarray]):
    """One learned waypoint per decision period, flown by the constrained MPC.

    The observation is the underlying environment's own, unchanged, so the
    learned layer sees exactly what Pure SAC sees. The reward is the sum of
    the environment's rewards over the decision period, so a return here and a
    return there differ only by the discounting the coarser step implies --
    which is the point: the episode is about 150 decisions instead of about
    950 control steps, so the completion bonus is worth ``gamma**150`` at
    reset rather than ``gamma**950``. The credit-assignment path to the entry
    decision is six times shorter, and that is a structural property of the
    coupling, not a tuned one.
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
        self.observation_space = self.env.observation_space
        self._last_info: dict[str, Any] = {}

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
        if self.hybrid_config.waypoint_parametrization == "radial_local":
            return self._radial_local_waypoint(raw)
        return self._clip_radius(raw * self.hybrid_config.waypoint_scale_m)

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
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        waypoint = self.waypoint_from_action(action)
        reward = 0.0
        terminated = truncated = False
        observation = None
        info: dict[str, Any] = self._last_info
        control_steps = 0
        zero_fallbacks = 0
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
            reward += float(step_reward)
            control_steps += 1
            if terminated or truncated:
                break
        assert observation is not None
        info = dict(info)
        info["hybrid_waypoint_target_frame"] = [float(v) for v in waypoint]
        info["hybrid_waypoint_radius_m"] = float(np.linalg.norm(waypoint))
        info["hybrid_control_steps"] = control_steps
        info["hybrid_qp_zero_fallbacks"] = zero_fallbacks
        self._last_info = info
        return observation, reward, terminated, truncated, info

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
