"""Evaluate a trained coupled policy on the same block, through the same metrics.

This is the row the paper's table takes from. It emits the shared
``eval.metrics.main_table_metrics`` block, so the coupled row is assembled by
the same code as the Pure MPC and scripted rows and cannot drift into a private
convention.

Compute accounting, which is the column most easily got wrong here: the MPC
still runs once per 0.1 s control step, and the learned layer runs once per
decision. So the per-control-step controller cost is the MPC solve, plus the
policy's forward pass charged to the first control step of its decision. That
is what a flight computer would actually pay.

**Run this serially, single process.** A parallel run's timings are not a
real-time claim and must not be reported as one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from controllers.mpc.constraints import normalized_precapture_truth_margins
from controllers.mpc.prediction import relative_to_vector
from dynamics.lie import so3_exp
from dynamics.relative import reconstruct_target_state
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import (
    precapture_adaptive_capture_environment_config,
    precapture_opportunity_environment_config,
    precapture_perception_environment_config,
    precapture_planning_environment_config,
    precapture_timing_probe_environment_config,
)
from eval.metrics import PRECAPTURE_MARGIN_KEYS, main_table_metrics
from train.hybrid_configs import SAC_MPC_HYBRID

VIOLATION_STEP_KEYS = (
    "keepout_violation_steps",
    "fov_violation_steps",
    "outer_speed_violation_steps",
    "outer_radial_violation_steps",
    "outer_approach_violation_steps",
    "corridor_violation_steps",
    "total_speed_violation_steps",
    "closing_speed_violation_steps",
)


#: Arrival-condition actions for the scripted timing arm. ``a=(+1,0)`` is full
#: commit to the desired pose -- the fixed-setpoint Pure MPC command, bitwise --
#: and ``a=(-1,0)`` holds the inertially frozen staging point (a cheap,
#: non-corotating wait). ``monotone_commit`` ratchets commit, so the arm is a
#: clean bang-bang: hold until the phase is right, then commit and stay.
_TIMED_COMMIT_ACTION = np.array([1.0, 0.0], dtype=np.float64)
_TIMED_HOLD_ACTION = np.array([-1.0, 0.0], dtype=np.float64)


class ScriptedEntryTiming:
    """B arm: a scripted oracle that times the commit to the entry phase.

    It waits at the inertial staging point until the tumbling capture port is
    predicted to face the staging direction at the moment the chaser would
    arrive, then commits the fixed-setpoint MPC. The prediction leads by the
    transit time ``range / lead_speed`` and reads the target's **actual future
    attitude** from the episode's cached truth target trajectory
    (``cache_target_trajectory``), which propagates the real 6-DoF free
    rigid-body tumble (non-spherical inertia, gravity gradient, J2). It is a
    **truth-future-assisted scripted timing comparator** -- a cheap approximate
    stand-in for "stage then enter", not a learned policy, not a strict upper
    bound or an optimal timing oracle. If it does well, staging has a potential
    resource advantage; if it does not, that does not mean the learned layer has
    no room. Constant-body-rate extrapolation is only a fallback if the cache is
    off. It never weakens the MPC: the executed command once committed is exactly
    the immediate arm's.

    The A arm is ``--control desired_pose`` (commit on the first decision); the
    only difference between the arms is *when* commit happens.
    """

    def __init__(
        self,
        env: PrecaptureHybridEnv,
        *,
        commit_favourability_cos: float,
        lead_speed_m_s: float,
        commit_deadline_margin_s: float = 20.0,
    ) -> None:
        self._env = env
        self._commit_cos = float(commit_favourability_cos)
        self._lead_speed_m_s = float(lead_speed_m_s)
        self._commit_deadline_margin_s = float(commit_deadline_margin_s)
        self._task = env.environment_config.precapture_task
        self._max_time_s = float(env.environment_config.max_time_s)
        self.committed = False
        self.commit_time_s: float | None = None
        self.favourability_at_commit: float | None = None
        self.predicted_favourability_at_commit: float | None = None

    def _future_target_rotation(self, lead_s: float) -> np.ndarray:
        """Target attitude ``lead_s`` ahead, from the cached truth trajectory if
        available (the real 6-DoF tumble the episode actually follows), else a
        constant-body-rate fallback."""

        target = self._env.env.target_state
        assert target is not None
        trajectory = self._env.env._target_trajectory
        if trajectory is not None:
            dt_s = self._env.environment_config.dt_s
            future_step = self._env.env.step_count + int(round(lead_s / dt_s))
            future_step = min(max(future_step, 0), len(trajectory) - 1)
            return np.asarray(trajectory[future_step].rotation, dtype=np.float64)
        rotation = np.asarray(target.rotation, dtype=np.float64)
        omega = np.asarray(target.omega, dtype=np.float64)
        return rotation @ so3_exp(omega * lead_s)

    def _predicted_favourability(self, lead_s: float) -> float:
        staging = self._env.env._staging_direction_inertial
        assert staging is not None
        normal = self._future_target_rotation(lead_s) @ self._task.approach_axis
        return float(np.clip(normal @ np.asarray(staging, dtype=np.float64), -1.0, 1.0))

    def action(self, info: dict[str, Any]) -> np.ndarray:
        if self.committed:
            return _TIMED_COMMIT_ACTION
        time_now = float(self._env.env.time_seconds)
        remaining_s = self._max_time_s - time_now
        range_m = float(info["target_center_distance_m"])
        lead_s = min(max(range_m / self._lead_speed_m_s, 0.0), max(remaining_s, 0.0))
        predicted = self._predicted_favourability(lead_s)
        # Wait only while there is still room to transit after waiting; otherwise
        # commit now so the arm never times out having never committed.
        must_commit = remaining_s <= lead_s + self._commit_deadline_margin_s
        if predicted >= self._commit_cos or must_commit:
            self.committed = True
            self.commit_time_s = time_now
            self.favourability_at_commit = float(
                self._env.env.entry_phase_favourability
            )
            self.predicted_favourability_at_commit = predicted
            return _TIMED_COMMIT_ACTION
        return _TIMED_HOLD_ACTION


def critic_min_q(policy: Any, observation: np.ndarray, action: np.ndarray) -> float:
    """Minimum over the SAC twin critics of Q(s, a) -- the value the deployment
    gate arbitrates on. The critic was trained on actions in the [-1, 1] box, so
    ``action`` is passed through unscaled (predict returns that box, and the zero
    residual is the nominal)."""

    import torch

    obs_tensor, _ = policy.policy.obs_to_tensor(observation)
    with torch.no_grad():
        action_tensor = torch.as_tensor(
            np.asarray(action, dtype=np.float32), device=obs_tensor.device
        ).reshape(1, -1)
        q_values = torch.cat(list(policy.critic(obs_tensor, action_tensor)), dim=1)
        return float(q_values.min(dim=1).values.item())


def truth_violation_step_counts(info: dict[str, Any]) -> dict[str, int | None]:
    """Preserve cumulative truth-geometry counters without inventing missing data."""

    return {
        key: int(info[key]) if key in info else None
        for key in VIOLATION_STEP_KEYS
    }


def arrival_blend_before_ratchet(
    action: np.ndarray, *, baseline_anchored_residual: bool
) -> float:
    """Return the arrival blend named by ``action`` before monotone latching.

    This is diagnostic instrumentation only.  Keeping the calculation here
    makes the distinction between an accepted raw residual and an effective
    task-reference change explicit without changing the environment mapping.
    """

    raw_commit = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
    if baseline_anchored_residual:
        raw_commit = float(np.clip(1.0 + 2.0 * raw_commit, -1.0, 1.0))
    return 0.5 * (raw_commit + 1.0)


def summarize_entry_channel(
    records: list[dict[str, Any]], entry_limits: dict[str, float]
) -> dict[str, Any]:
    """Aggregate the pre-registered entry/retry questions from episode records."""

    crossings = [event for record in records for event in record["entry_crossings"]]
    illegal = [event for event in crossings if not event["legal"]]
    retries = [event for event in crossings if event["retry_after_illegal"]]

    def fraction_with_positive_excess(key: str) -> float | None:
        if not illegal:
            return None
        return sum(event["violation_excess"][key] > 0.0 for event in illegal) / len(
            illegal
        )

    violation_fractions = {
        "radial_distance": fraction_with_positive_excess("radial_distance_m"),
        "target_frame_speed": fraction_with_positive_excess(
            "target_frame_speed_m_s"
        ),
        "closing_speed": fraction_with_positive_excess("closing_speed_m_s"),
    }
    available = {
        name: value for name, value in violation_fractions.items() if value is not None
    }
    primary = "no_illegal_crossing"
    if available:
        candidate = max(available, key=available.__getitem__)
        primary = candidate if available[candidate] > 0.60 else "mixed_no_cause_above_60pct"

    episodes_with_illegal = [
        record for record in records if record["illegal_entry_crossing_count"] > 0
    ]
    episodes_with_exit = [
        record for record in records if record["illegal_exit_count"] > 0
    ]
    episodes_with_retry = [
        record for record in records if record["retry_crossing_count"] > 0
    ]
    return {
        "entry_limits": entry_limits,
        "crossings_total": len(crossings),
        "legal_crossings": sum(event["legal"] for event in crossings),
        "illegal_crossings": len(illegal),
        "illegal_violation_fractions": violation_fractions,
        "primary_illegal_cause_by_preregistered_60pct_rule": primary,
        "episodes_with_illegal_crossing": len(episodes_with_illegal),
        "episodes_with_exit_after_illegal": len(episodes_with_exit),
        "episodes_with_exit_and_retry": len(episodes_with_retry),
        "exit_and_retry_fraction_all_episodes": len(episodes_with_retry)
        / max(1, len(records)),
        "exit_and_retry_fraction_illegal_episodes": len(episodes_with_retry)
        / max(1, len(episodes_with_illegal)),
        "retry_crossings": len(retries),
        "legal_retry_crossings": sum(event["legal"] for event in retries),
        "legal_retry_fraction": (
            sum(event["legal"] for event in retries) / len(retries) if retries else None
        ),
        "retry_remaining_time_s": [
            event["remaining_time_s"] for event in retries
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--tumble-scale",
        type=float,
        default=None,
        help=(
            "Override the target tumble scale (0.20 = 0.0412 rad/s nominal). Used "
            "for the decision-margin calibration and cross-regime generalization; "
            "the same reward/policy face a different tumble rate."
        ),
    )
    parser.add_argument(
        "--parametrization",
        choices=["absolute", "radial_local", "arrival_condition"],
        default="radial_local",
        help="Action parametrisation used by the trained policy or control.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Trained SAC checkpoint. Omit to run one of the controls.",
    )
    parser.add_argument(
        "--stochastic-policy",
        action="store_true",
        help=(
            "Sample the loaded policy instead of using its deterministic mean. "
            "Diagnostic only: approximates the final policy's training-time "
            "exploration distribution. Default remains deterministic."
        ),
    )
    parser.add_argument(
        "--phase-time-observation",
        action="store_true",
        help="Use the V4 31D observation expected by phase/time-trained policies.",
    )
    parser.add_argument(
        "--execution-feedback",
        action="store_true",
        help=(
            "Append the lower layer's 3D execution summary, as the coupled "
            "policies trained with the reverse channel expect. A policy is "
            "loaded against the observation it was trained on, so this has to "
            "match the run's manifest rather than be guessed."
        ),
    )
    parser.add_argument(
        "--control",
        choices=["desired_pose", "random", "timed_entry"],
        help=(
            "Run without a model. 'desired_pose' is the fixed-setpoint lower "
            "layer delivered through the wrapper (the immediate-entry A arm and "
            "the row the coupled policy has to beat). 'timed_entry' is the "
            "scripted timing oracle B arm (hold at the inertial staging point, "
            "commit when the entry phase is favourable). 'random' is the floor."
        ),
    )
    parser.add_argument(
        "--baseline-anchored-residual",
        action="store_true",
        help=(
            "Evaluate the policy in the baseline-anchored residual interface: "
            "the action is a residual on the nominal (fixed-setpoint) arrival "
            "action, so zero residual recovers Pure MPC bitwise. Required to "
            "evaluate a model trained with --baseline-anchored-residual, and the "
            "nominal that the deployment gate falls back to."
        ),
    )
    parser.add_argument(
        "--deployment-gate",
        action="store_true",
        help=(
            "Critic-advantage deployment gate (proposed method). Deviate from "
            "the nominal only when the SAC critic prefers the policy's residual "
            "action over the zero residual by more than --gate-advantage-margin; "
            "otherwise command the nominal. Requires --model and "
            "--baseline-anchored-residual."
        ),
    )
    parser.add_argument(
        "--gate-advantage-margin",
        type=float,
        default=0.0,
        help=(
            "deployment gate only: minimum critic advantage "
            "Q(s,a_policy) - Q(s,a_nom) required to deviate from the nominal."
        ),
    )
    parser.add_argument(
        "--opportunity-task",
        action="store_true",
        help=(
            "Evaluate on the opportunity task (outer frozen approach corridor + "
            "inner rotating capture). Adds the staging-direction observation. "
            "Must match how the model was trained."
        ),
    )
    parser.add_argument(
        "--adaptive-task",
        action="store_true",
        help=(
            "Evaluate on the adaptive sync-entry mainline task (opened initial "
            "distribution, no hard far-range corridor). Adds the staging-direction "
            "observation. Must match how the model was trained."
        ),
    )
    parser.add_argument(
        "--timing-probe",
        action="store_true",
        help=(
            "Use the opened-distribution timing-value config "
            "(precapture_timing_probe_environment_config): wider initial range "
            "and pointing, same frozen constraints/authority/tumble. The A/B "
            "timing comparison runs on this config."
        ),
    )
    parser.add_argument(
        "--entry-phase-gate-deg",
        type=float,
        default=180.0,
        help=(
            "Entry-phase gate half-angle for the timing-probe config. Default "
            "180 (gate OFF) -- the primary A/B measures timing value from "
            "physics alone. A finite value (e.g. 90) is a diagnostic guardrail "
            "only, never the paper mechanism; a gate-manufactured A/B difference "
            "is not admissible evidence."
        ),
    )
    parser.add_argument(
        "--commit-favourability-cos",
        type=float,
        default=float(np.cos(np.deg2rad(45.0))),
        help=(
            "timed_entry only: commit once the predicted arrival favourability "
            "(cosine) clears this threshold. Default cos(45 deg) ~ 0.707."
        ),
    )
    parser.add_argument(
        "--commit-lead-speed-m-s",
        type=float,
        default=0.20,
        help=(
            "timed_entry only: nominal closing speed used to estimate the "
            "transit lead (range / speed) the phase prediction leads by."
        ),
    )
    parser.add_argument(
        "--perception",
        action="store_true",
        help=(
            "Build the non-cooperative environment (A1 camera + relative EKF): "
            "the 29D estimated observation and a live EKF estimate. Required for "
            "--control-source estimated."
        ),
    )
    parser.add_argument(
        "--control-source",
        choices=["truth", "estimated"],
        default="truth",
        help=(
            "State the MPC flies on. 'truth' is the full-information controller; "
            "'estimated' feeds the EKF estimate (and the target pose reconstructed "
            "from the known chaser), so the controller coasts on a stale estimate "
            "while the target is out of view -- the non-cooperative operational "
            "baseline. Requires --perception."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    partial_output = args.output.with_suffix(args.output.suffix + ".partial")
    if (args.model is None) == (args.control is None):
        raise ValueError("give exactly one of --model or --control")
    if args.stochastic_policy and args.model is None:
        raise ValueError("--stochastic-policy requires --model")
    control_source_estimated = args.control_source == "estimated"
    if control_source_estimated and not args.perception:
        raise ValueError("--control-source estimated requires --perception")
    if args.control == "timed_entry" and args.parametrization != "arrival_condition":
        raise ValueError(
            "--control timed_entry requires --parametrization arrival_condition "
            "(the hold/commit actions live on that interface)"
        )
    if args.deployment_gate and (
        args.model is None or not args.baseline_anchored_residual
    ):
        raise ValueError(
            "--deployment-gate requires --model and --baseline-anchored-residual"
        )
    if args.baseline_anchored_residual and args.parametrization != "arrival_condition":
        raise ValueError(
            "--baseline-anchored-residual requires --parametrization arrival_condition"
        )

    policy = None
    if args.model is not None:
        from stable_baselines3 import SAC

        policy = SAC.load(args.model, device="cpu")

    records: list[dict[str, Any]] = []
    controller_times_s: list[float] = []
    environment_times_s: list[float] = []
    generator = np.random.default_rng(args.seed)
    if args.adaptive_task:
        if args.perception or args.timing_probe or args.opportunity_task:
            raise ValueError(
                "--adaptive-task is exclusive with "
                "--perception / --timing-probe / --opportunity-task"
            )
        base_environment_config = precapture_adaptive_capture_environment_config()
    elif args.opportunity_task:
        if args.perception or args.timing_probe:
            raise ValueError(
                "--opportunity-task is exclusive with --perception / --timing-probe"
            )
        base_environment_config = precapture_opportunity_environment_config()
    elif args.timing_probe:
        if args.perception:
            raise ValueError("--timing-probe and --perception are exclusive")
        base_environment_config = precapture_timing_probe_environment_config(
            entry_phase_gate_deg=args.entry_phase_gate_deg
        )
    elif args.perception:
        base_environment_config = precapture_perception_environment_config()
    else:
        base_environment_config = precapture_planning_environment_config()
    # The timing oracle (B arm) reads the target's future attitude from the
    # cached truth trajectory, so it needs the cache on. The cache is the same
    # deterministic tumble the live step follows, so both arms see identical
    # target dynamics either way -- only the lookahead differs.
    if args.tumble_scale is not None:
        base_environment_config = replace(
            base_environment_config, phase2_target_tumble_scale=args.tumble_scale
        )
    environment_config = replace(
        base_environment_config,
        cache_target_trajectory=(args.control == "timed_entry"),
    )
    task = environment_config.precapture_task
    entry_position = (
        task.port_position
        + task.entry_port_axial_distance_m * task.approach_axis
    )
    entry_axial_remaining = float(
        task.approach_axis @ (entry_position - task.desired_position)
    )
    entry_limits = {
        "radial_distance_m": float(task.entry_disc_radius_m),
        "target_frame_speed_m_s": float(task.terminal_total_speed_limit_m_s),
        "closing_speed_m_s": float(task.closing_speed_limit(entry_axial_remaining)),
    }

    for episode in range(args.episodes):
        seed = args.seed + episode
        env = PrecaptureHybridEnv(
            environment_config=environment_config,
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=args.horizon,
                waypoint_parametrization=args.parametrization,
                runtime_diagnostics=False,
                include_target_phase_and_time_observation=(
                    args.phase_time_observation
                ),
                include_execution_feedback_observation=args.execution_feedback,
                baseline_anchored_residual=args.baseline_anchored_residual,
                include_staging_direction_observation=(
                    args.opportunity_task or args.adaptive_task
                ),
            )
        )
        if policy is not None and episode == 0:
            expected = int(np.prod(policy.observation_space.shape))
            actual = int(np.prod(env.observation_space.shape))
            if expected != actual:
                raise ValueError(
                    f"the checkpoint expects a {expected}D observation and this "
                    f"configuration builds a {actual}D one. Set "
                    "--phase-time-observation / --execution-feedback to match "
                    "the run's manifest instead of loading a policy against an "
                    "observation it never saw."
                )
            expected_action = int(np.prod(policy.action_space.shape))
            actual_action = int(np.prod(env.action_space.shape))
            if expected_action != actual_action:
                raise ValueError(
                    f"the checkpoint expects a {expected_action}D action and "
                    f"this parametrisation gives {actual_action}D"
                )
        observation, info = env.reset(seed=seed)
        desired = env.environment_config.precapture_task.desired_position
        timing = (
            ScriptedEntryTiming(
                env,
                commit_favourability_cos=args.commit_favourability_cos,
                lead_speed_m_s=args.commit_lead_speed_m_s,
            )
            if args.control == "timed_entry"
            else None
        )
        minimum_margins = {
            key: float(info[key]) for key in PRECAPTURE_MARGIN_KEYS if key in info
        }
        force_impulse = torque_impulse = 0.0
        peak_force_n = peak_torque_nm = 0.0
        minimum_truth_normalized_margin = float("inf")
        waypoints: list[list[float]] = []
        decision_times_s: list[float] = []
        policy_actions_raw: list[list[float]] | None = (
            [] if policy is not None else None
        )
        applied_actions: list[list[float]] = []
        blends_before_ratchet: list[float | None] = []
        blends_after_ratchet: list[float | None] = []
        references_changed: list[bool] = []
        latch_step: int | None = None
        latch_cause: str | None = None
        qp_infeasible_steps_total = 0
        zero_fallback_steps_total = 0
        maximum_successful_slack: float | None = None
        first_infeasible_time_s: float | None = None
        consecutive_zero_wrench_steps = 0
        max_consecutive_zero_wrench_steps = 0
        entry_crossings: list[dict[str, Any]] = []
        gate_decisions = 0
        gate_deviations = 0
        gate_advantages: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated):
            gate_fallback = False
            if policy is not None:
                started = perf_counter()
                action, _ = policy.predict(
                    observation, deterministic=not args.stochastic_policy
                )
                raw_policy_action = np.asarray(action, dtype=float).copy()
                if args.deployment_gate:
                    nominal_action = np.zeros_like(action)
                    advantage = critic_min_q(
                        policy, observation, action
                    ) - critic_min_q(policy, observation, nominal_action)
                    gate_decisions += 1
                    gate_advantages.append(advantage)
                    if advantage > args.gate_advantage_margin:
                        gate_deviations += 1
                    else:
                        # Fall back to the nominal (fixed-setpoint Pure MPC).
                        action = nominal_action
                        gate_fallback = True
                inference_s = perf_counter() - started
            elif args.control == "desired_pose":
                started = perf_counter()
                action = env.action_for_waypoint(desired)
                inference_s = perf_counter() - started
            elif args.control == "timed_entry":
                assert timing is not None
                started = perf_counter()
                action = timing.action(info)
                inference_s = perf_counter() - started
            else:
                started = perf_counter()
                action = generator.uniform(-1.0, 1.0, size=env.action_space.shape)
                inference_s = perf_counter() - started

            decision_times_s.append(float(env.env.time_seconds))
            if policy_actions_raw is not None:
                policy_actions_raw.append(
                    [float(v) for v in raw_policy_action.reshape(-1)]
                )
            applied_actions.append(
                [float(v) for v in np.asarray(action).reshape(-1)]
            )
            blend_before = (
                arrival_blend_before_ratchet(
                    action,
                    baseline_anchored_residual=args.baseline_anchored_residual,
                )
                if args.parametrization == "arrival_condition"
                else None
            )
            waypoint = env.waypoint_from_action(action)
            waypoints.append([float(v) for v in waypoint])
            blend_after = (
                float(env._commit_blend)
                if args.parametrization == "arrival_condition"
                and env.hybrid_config.monotone_commit
                else blend_before
            )
            reference_changed = not bool(
                np.allclose(waypoint, desired, rtol=0.0, atol=1.0e-9)
            )
            blends_before_ratchet.append(blend_before)
            blends_after_ratchet.append(blend_after)
            references_changed.append(reference_changed)
            if latch_step is None and blend_after is not None and blend_after >= 1.0:
                latch_step = len(waypoints) - 1
                if gate_fallback:
                    latch_cause = "gate_fallback"
                elif policy is not None:
                    latch_cause = "policy"
                else:
                    latch_cause = "scripted"
            # Inline the decision so each control step's cost is separable.
            for index in range(env.hybrid_config.decision_period_steps):
                assert env.env.relative is not None
                assert env.env.target_state is not None
                if control_source_estimated:
                    assert env.env.chaser_state is not None
                    control_relative = env.env.observed_relative
                    control_target = reconstruct_target_state(
                        env.env.chaser_state, control_relative
                    )
                else:
                    control_relative = env.env.relative
                    control_target = env.env.target_state
                started = perf_counter()
                wrench, diagnostics = env.controller.command(
                    relative_to_vector(control_relative),
                    target_state=control_target,
                    time_seconds=env.env.time_seconds,
                    terminal_latched=bool(info["terminal_region_active"]),
                    external_reference=waypoint,
                )
                controller_s = perf_counter() - started
                if str(diagnostics.status).startswith("infeasible"):
                    qp_infeasible_steps_total += 1
                    if first_infeasible_time_s is None:
                        first_infeasible_time_s = float(env.env.time_seconds)
                if diagnostics.used_zero_fallback:
                    zero_fallback_steps_total += 1
                else:
                    successful_slack = float(diagnostics.maximum_slack)
                    maximum_successful_slack = (
                        successful_slack
                        if maximum_successful_slack is None
                        else max(maximum_successful_slack, successful_slack)
                    )
                if np.count_nonzero(wrench) == 0:
                    consecutive_zero_wrench_steps += 1
                    max_consecutive_zero_wrench_steps = max(
                        max_consecutive_zero_wrench_steps,
                        consecutive_zero_wrench_steps,
                    )
                else:
                    consecutive_zero_wrench_steps = 0
                # The policy is charged to the first control step of its
                # decision, which is when a flight computer would pay it.
                controller_times_s.append(
                    controller_s + (inference_s if index == 0 else 0.0)
                )
                force_norm = float(np.linalg.norm(wrench[3:]))
                torque_norm = float(np.linalg.norm(wrench[:3]))
                force_impulse += force_norm * env.environment_config.dt_s
                torque_impulse += torque_norm * env.environment_config.dt_s
                peak_force_n = max(peak_force_n, force_norm)
                peak_torque_nm = max(peak_torque_nm, torque_norm)
                started = perf_counter()
                observation, _, terminated, truncated, info = env.env.step(
                    wrench_to_normalized(
                        GeneralizedForce.from_vector(wrench),
                        max_torque_per_axis_nm=(
                            env.environment_config.max_torque_per_axis_nm
                        ),
                        max_force_per_axis_n=(
                            env.environment_config.max_force_per_axis_n
                        ),
                    )
                )
                observation = env._policy_observation(observation)
                environment_times_s.append(perf_counter() - started)
                # An illegal crossing does not latch.  Once the chaser returns
                # outside the entry plane, the next outside-to-inside crossing
                # is a directly observed retry rather than an inferred intent.
                for prior in entry_crossings:
                    if (
                        not prior["legal"]
                        and prior["exit_time_s"] is None
                        and float(info["port_axial_distance_m"])
                        > task.entry_port_axial_distance_m
                    ):
                        prior["exit_time_s"] = float(env.env.time_seconds)
                if bool(info.get("terminal_entry_crossed", False)):
                    radial = float(info["entry_crossing_radial_distance_m"])
                    target_speed = float(
                        info["entry_crossing_target_frame_speed_m_s"]
                    )
                    closing_speed = float(info["entry_crossing_closing_speed_m_s"])
                    retry_of = next(
                        (
                            prior
                            for prior in reversed(entry_crossings)
                            if not prior["legal"]
                            and prior["exit_time_s"] is not None
                            and prior["next_crossing_attempt"] is None
                        ),
                        None,
                    )
                    event = {
                        "attempt": len(entry_crossings) + 1,
                        "time_s": float(env.env.time_seconds),
                        "remaining_time_s": float(
                            environment_config.max_time_s - env.env.time_seconds
                        ),
                        "legal": bool(info["terminal_entry_legal"]),
                        # Entry-phase favourability at the crossing: the physical
                        # quantity the timing arm targets, recorded for both arms
                        # so the A/B can compare where each one actually enters.
                        "favourability": float(env.env.entry_phase_favourability),
                        "radial_distance_m": radial,
                        "target_frame_speed_m_s": target_speed,
                        "closing_speed_m_s": closing_speed,
                        "violation_excess": {
                            "radial_distance_m": radial
                            - entry_limits["radial_distance_m"],
                            "target_frame_speed_m_s": target_speed
                            - entry_limits["target_frame_speed_m_s"],
                            "closing_speed_m_s": closing_speed
                            - entry_limits["closing_speed_m_s"],
                        },
                        "retry_after_illegal": retry_of is not None,
                        "retry_of_attempt": (
                            retry_of["attempt"] if retry_of is not None else None
                        ),
                        "exit_time_s": None,
                        "next_crossing_attempt": None,
                        "next_crossing_legal": None,
                    }
                    if retry_of is not None:
                        retry_of["next_crossing_attempt"] = event["attempt"]
                        retry_of["next_crossing_legal"] = event["legal"]
                    entry_crossings.append(event)
                for key in PRECAPTURE_MARGIN_KEYS:
                    if key in info:
                        minimum_margins[key] = min(
                            minimum_margins.get(key, float(info[key])),
                            float(info[key]),
                        )
                # The per-key minima above are the raw task fields, and every
                # one of them exists at every step whether or not its
                # constraint is live. A chaser 17 m out and off the approach
                # axis has a corridor margin of -19 m, which is not a
                # violation -- the corridor is not a constraint out there --
                # yet a table built from those minima reports a completing,
                # zero-violation episode as a massive violator, and mixes
                # metres, radians and m/s into one column besides. The truth
                # margin is the gated, normalised quantity: inactive rows are
                # large positive and every active row is divided by its own
                # limit, so one number is comparable across rows and against
                # the Pure MPC row, which already reports it.
                assert env.env.relative is not None
                assert env.env.target_state is not None
                truth_margins = normalized_precapture_truth_margins(
                    relative_to_vector(env.env.relative),
                    task,
                    target_angular_velocity_rad_s=env.env.target_state.omega,
                    terminal_latched=bool(info["terminal_region_active"]),
                )
                minimum_truth_normalized_margin = min(
                    minimum_truth_normalized_margin,
                    float(np.min(truth_margins)),
                )
                if terminated or truncated:
                    break

        zero_violation = all(
            int(info[key]) == 0 for key in VIOLATION_STEP_KEYS if key in info
        )
        records.append(
            {
                "episode": episode,
                "seed": seed,
                "completed": bool(info["completed"]),
                "truth_geometry_zero_violation_completed": bool(
                    info["completed"] and zero_violation
                ),
                "constraint_violated": not zero_violation,
                **truth_violation_step_counts(info),
                "survival_s": float(env.env.time_seconds),
                "steps": int(env.env.step_count),
                "decisions": len(waypoints),
                "force_impulse_n_s": force_impulse,
                "torque_impulse_nm_s": torque_impulse,
                "peak_force_n": peak_force_n,
                "peak_torque_nm": peak_torque_nm,
                "equivalent_delta_v_m_s": (
                    force_impulse / env.env.chaser_parameters.mass
                ),
                "minimum_margins": minimum_margins,
                "minimum_truth_normalized_margin": (
                    minimum_truth_normalized_margin
                    if np.isfinite(minimum_truth_normalized_margin)
                    else None
                ),
                "illegal_terminal_entry_count": int(
                    info.get("illegal_terminal_entry_count", 0)
                ),
                "terminal_region_active": float(info["terminal_region_active"]),
                "time_failure": bool(info.get("time_failure", False)),
                "distance_failure": bool(info.get("distance_failure", False)),
                "qp_infeasible_steps_total": qp_infeasible_steps_total,
                "zero_fallback_steps_total": zero_fallback_steps_total,
                "maximum_successful_slack": maximum_successful_slack,
                "first_infeasible_time_s": first_infeasible_time_s,
                "max_consecutive_zero_wrench_steps": (
                    max_consecutive_zero_wrench_steps
                ),
                "entry_crossings": entry_crossings,
                "entry_crossing_count": len(entry_crossings),
                "legal_entry_crossing_count": sum(
                    event["legal"] for event in entry_crossings
                ),
                "illegal_entry_crossing_count": sum(
                    not event["legal"] for event in entry_crossings
                ),
                "illegal_exit_count": sum(
                    not event["legal"] and event["exit_time_s"] is not None
                    for event in entry_crossings
                ),
                "retry_crossing_count": sum(
                    event["retry_after_illegal"] for event in entry_crossings
                ),
                "retry_legal_count": sum(
                    event["retry_after_illegal"] and event["legal"]
                    for event in entry_crossings
                ),
                "final_remaining_time_s": float(
                    environment_config.max_time_s - env.env.time_seconds
                ),
                "final_target_center_distance_m": float(
                    info["target_center_distance_m"]
                ),
                "final_position_error_m": float(info["position_error_m"]),
                "commit_time_s": (timing.commit_time_s if timing is not None else None),
                "favourability_at_commit": (
                    timing.favourability_at_commit if timing is not None else None
                ),
                "predicted_favourability_at_commit": (
                    timing.predicted_favourability_at_commit
                    if timing is not None
                    else None
                ),
                "gate_decisions": gate_decisions,
                "gate_deviations": gate_deviations,
                "gate_deviation_fraction": (
                    gate_deviations / gate_decisions if gate_decisions else None
                ),
                "gate_mean_advantage": (
                    float(np.mean(gate_advantages)) if gate_advantages else None
                ),
                "decision_times_s": decision_times_s,
                "policy_actions_raw": policy_actions_raw,
                "applied_actions": applied_actions,
                "blend_before_ratchet": blends_before_ratchet,
                "blend_after_ratchet": blends_after_ratchet,
                "reference_changed": references_changed,
                "latch_step": latch_step,
                "latch_cause": latch_cause,
                "waypoints_target_frame": waypoints,
            }
        )
        partial_output.parent.mkdir(parents=True, exist_ok=True)
        partial_output.write_text(
            json.dumps(
                {
                    "source": str(args.model) if args.model is not None else args.control,
                    "episodes_finished": len(records),
                    "entry_limits": entry_limits,
                    "entry_channel": summarize_entry_channel(records, entry_limits),
                    "records": records,
                },
                indent=1,
            )
        )
        env.close()
        last = records[-1]
        print(
            f"episode {episode + 1}/{args.episodes} seed={seed} "
            f"completed={last['completed']} t={last['survival_s']:.1f}s "
            f"dv={last['equivalent_delta_v_m_s']:.3f} "
            f"illegal={last['illegal_terminal_entry_count']} "
            f"crossings={last['entry_crossing_count']} "
            f"retries={last['retry_crossing_count']}"
        )

    payload = {
        "source": str(args.model) if args.model is not None else args.control,
        "episodes": args.episodes,
        "seed_block": args.seed,
        "horizon": args.horizon,
        "waypoint_parametrization": args.parametrization,
        "phase_time_observation": args.phase_time_observation,
        "execution_feedback": args.execution_feedback,
        "tumble_scale": args.tumble_scale,
        "adaptive_task": args.adaptive_task,
        "opportunity_task": args.opportunity_task,
        "baseline_anchored_residual": args.baseline_anchored_residual,
        "deployment_gate": args.deployment_gate,
        "stochastic_policy": args.stochastic_policy,
        "gate_advantage_margin": (
            args.gate_advantage_margin if args.deployment_gate else None
        ),
        "action_trace_semantics": {
            "decision_times_s": "episode time at the start of each decision",
            "policy_actions_raw": (
                "deterministic normalized policy action before the deployment gate; "
                "null for scripted controls"
            ),
            "applied_actions": (
                "action passed to waypoint_from_action after any deployment-gate "
                "fallback"
            ),
            "alignment": (
                "decision_times_s, policy_actions_raw when present, applied_actions, "
                "and waypoints_target_frame are index-aligned"
            ),
        },
        "timing_probe": args.timing_probe,
        "entry_phase_gate_deg": args.entry_phase_gate_deg if args.timing_probe else None,
        "timed_entry_settings": (
            {
                "commit_favourability_cos": args.commit_favourability_cos,
                "commit_lead_speed_m_s": args.commit_lead_speed_m_s,
            }
            if args.control == "timed_entry"
            else None
        ),
        "hyperparameters": {"gamma": SAC_MPC_HYBRID.gamma},
        "compute_note": (
            "valid only if this ran serially in a single process; the MPC is "
            "charged every control step and the policy once per decision; "
            "post-solve runtime diagnostics are disabled"
        ),
        "records": records,
        "entry_channel": summarize_entry_channel(records, entry_limits),
        "main_table": {
            **main_table_metrics(
                records,
                controller_times_s=controller_times_s,
                environment_step_times_s=environment_times_s,
                control_period_s=0.1,
            ),
            # The comparable safety column. It spans every episode, not only
            # the completed ones -- grazing a boundary on an episode a method
            # loses does not earn it a clean margin.
            "worst_truth_normalized_margin": min(
                (
                    record["minimum_truth_normalized_margin"]
                    for record in records
                    if record.get("minimum_truth_normalized_margin") is not None
                ),
                default=None,
            ),
        },
    }
    args.output.write_text(json.dumps(payload, indent=1))
    partial_output.unlink(missing_ok=True)
    table = payload["main_table"]
    print(
        f"\ncompleted {table['completed_episodes']}/{table['episodes']}  "
        f"controller p95/budget "
        f"{table['per_step_compute_s']['controller_p95_over_budget']:.2f}x"
    )


if __name__ == "__main__":
    main()
