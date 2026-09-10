"""Scripted upper layers through the coupling interface: the feasibility gate.

Before any training is authorised the project rules ask for two things: that
the lower layer genuinely fails, and that an ideal controller can still solve
the task through the *same* interface the learned layer will use. This
supplies the second. Nothing here is a reportable baseline row -- these are
hand-written waypoint policies whose only job is to bound what the 3D action
space can express.

Policies
--------
``desired_pose``
    One constant waypoint at the task's own desired pose. This is the
    fixed-setpoint MPC, delivered through the coupling wrapper, and it should
    reproduce the no-upper-layer result. It is the control that says the
    wrapper adds nothing and loses nothing.

``hold_then_enter``
    The window decision made by hand, and the only policy here that uses more
    than a constant. Waiting has to mean *inertially* still: the approach axis
    is body-fixed on the target, so a chaser that holds a fixed target-frame
    point co-rotates with it and the entry geometry never changes -- there is
    no window to wait for, only fuel to spend. So the hold command is the
    chaser's frozen inertial position, re-expressed in the target frame on
    every decision, which the 2 s channel can carry because the policy
    re-issues it. The rule commits to the desired pose once the body-fixed
    approach axis has swung to within ``--entry-alignment-deg`` of that held
    inertial direction. It is fixed before the run and not tuned against the
    outcome.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dynamics.lie import se3_exp
from controllers.mpc.prediction import relative_to_vector
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
from experiments.evaluate_mpc import _active_precapture_margins


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-12 else vector


class ScriptedWaypointPolicy:
    def __init__(
        self,
        env: PrecaptureHybridEnv,
        kind: str,
        alignment_deg: float,
        commit_time_s: float = 0.0,
        hold_radius_m: float | None = None,
        radius_step_m: float = 0.4,
    ):
        self.radius_step_m = radius_step_m
        self.ramp_radius = None
        self.hold_radius_m = hold_radius_m
        self.commit_time_s = float(commit_time_s)
        self.env = env
        self.kind = kind
        self.cosine_threshold = float(np.cos(np.deg2rad(alignment_deg)))
        self.task = env.environment_config.precapture_task
        self.committed = False
        self.held_inertial: np.ndarray | None = None

    def _action_for(self, point: np.ndarray) -> np.ndarray:
        """Invert the parametrisation: the action that names ``point``.

        Under ``radial_local`` a single decision cannot always name an
        arbitrary point -- the radius moves by at most a factor of e**0.7 and
        the direction tilts by at most about 41 deg -- so this returns the
        closest reachable command and the policy converges over a few
        decisions. That is a property of the parametrisation, not a
        shortcoming of the script: the optimiser could not fly a bigger jump
        in 2 s either.
        """

        return self.env.action_for_waypoint(point)

    def act(self) -> tuple[np.ndarray, bool]:
        desired = self.task.desired_position
        if self.kind == "desired_pose":
            return self._action_for(desired), True
        assert self.env.env.relative is not None
        assert self.env.env.target_state is not None
        rotation = self.env.env.target_state.rotation
        state = relative_to_vector(self.env.env.relative)
        position = se3_exp(state[:6])[:3, 3]
        if self.held_inertial is None:
            self.held_inertial = rotation @ position
            if self.kind == "hold_radius" and self.hold_radius_m is not None:
                self.held_inertial = self.hold_radius_m * _unit(self.held_inertial)
        if self.kind == "ramp_in":
            if float(self.env.env.time_seconds) < self.commit_time_s:
                return self._action_for(rotation.T @ self.held_inertial), False
            if self.ramp_radius is None:
                self.ramp_radius = float(np.linalg.norm(position))
            goal_radius = float(np.linalg.norm(desired))
            self.ramp_radius = max(goal_radius, self.ramp_radius - self.radius_step_m)
            return self._action_for(_unit(desired) * self.ramp_radius), True
        # The window is an inertial-frame alignment: the approach axis is
        # body-fixed and sweeps a cone as the target turns, and it opens when
        # that axis swings towards where the chaser is holding.
        if self.kind in {"commit_at", "hold_radius"}:
            ready = float(self.env.env.time_seconds) >= self.commit_time_s
        else:
            axis_inertial = _unit(rotation @ self.task.approach_axis)
            ready = (
                float(axis_inertial @ _unit(self.held_inertial))
                >= self.cosine_threshold
            )
        if self.committed or ready:
            self.committed = True
            return self._action_for(desired), True
        # Wait: name the frozen inertial point, expressed in the target frame
        # for this decision. It is a different target-frame point every time,
        # which is exactly what "stay put while the target turns" requires.
        return self._action_for(rotation.T @ self.held_inertial), False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--policy",
        choices=["desired_pose", "hold_then_enter", "commit_at", "hold_radius", "ramp_in"],
        required=True,
    )
    parser.add_argument("--entry-alignment-deg", type=float, default=40.0)
    parser.add_argument(
        "--commit-time-s",
        type=float,
        default=0.0,
        help=(
            "``commit_at`` only: hold the frozen inertial point until this "
            "time, then command the desired pose. One decision variable -- the "
            "entry time -- so a sweep over it says whether the action space "
            "contains a solution at all, independently of any rule for "
            "choosing it."
        ),
    )
    parser.add_argument("--hold-radius-m", type=float, default=None)
    parser.add_argument("--radius-step-m", type=float, default=0.4)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--no-target-cache", action="store_true")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--parametrization", choices=["absolute", "radial_local"], default="absolute"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = PrecaptureHybridEnv(
            environment_config=replace(precapture_planning_environment_config(), cache_target_trajectory=not args.no_target_cache),
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=args.horizon,
                runtime_diagnostics=not args.no_diagnostics,
                waypoint_parametrization=args.parametrization,
            )
        )
        _, info = env.reset(seed=seed)
        policy = ScriptedWaypointPolicy(
            env, args.policy, args.entry_alignment_deg, args.commit_time_s, args.hold_radius_m, args.radius_step_m
        )
        recorder = {"force_impulse_n_s": 0.0,
                    "minimum_truth_normalized_margin": _active_precapture_margins(info, env.environment_config)[1],
                    "fov_peak_deg": float(np.rad2deg(info["fov_angle_rad"])),
                    "control_steps": 0}
        original_step = env.env.step

        def recorded_step(action):
            result = original_step(action)
            micro_info = result[-1]
            cfg = env.environment_config
            force = np.clip(np.asarray(action)[3:], -1.0, 1.0) * cfg.max_force_per_axis_n
            recorder["force_impulse_n_s"] += float(np.linalg.norm(force)) * cfg.dt_s
            recorder["minimum_truth_normalized_margin"] = min(
                recorder["minimum_truth_normalized_margin"],
                _active_precapture_margins(micro_info, cfg)[1])
            recorder["fov_peak_deg"] = max(recorder["fov_peak_deg"], float(np.rad2deg(micro_info["fov_angle_rad"])))
            recorder["control_steps"] += 1
            return result

        env.env.step = recorded_step
        total_fallbacks = 0
        total_reward = 0.0
        decisions = 0
        commit_time_s = None
        terminated = truncated = False
        while not (terminated or truncated):
            action, committed = policy.act()
            if committed and commit_time_s is None:
                commit_time_s = float(env.env.time_seconds)
            _, reward, terminated, truncated, info = env.step(action)
            total_fallbacks += int(info["hybrid_qp_zero_fallbacks"])
            total_reward += reward
            decisions += 1
        records.append(
            {
                **recorder,
                "seed": seed,
                "hold_radius_m": args.hold_radius_m,
                "radius_step_m": args.radius_step_m,
                "runtime_diagnostics": not args.no_diagnostics,
                "cache_target_trajectory": not args.no_target_cache,
                "waypoint_parametrization": args.parametrization,
                "qp_fallbacks_total": total_fallbacks,
                "termination_reason": "completed" if info["completed"] else ",".join(k for k,v in info.items() if k.endswith("_failure") and bool(v)),
                "policy": args.policy,
                "entry_alignment_deg": args.entry_alignment_deg,
                "commit_time_s_setting": args.commit_time_s,
                "horizon": args.horizon,
                "completed": bool(info["completed"]),
                "final_time_s": float(env.env.time_seconds),
                "commit_time_s": commit_time_s,
                "decisions": decisions,
                "undiscounted_reward": total_reward,
                "terminal_region_active": float(info["terminal_region_active"]),
                "illegal_terminal_entry_count": float(
                    info["illegal_terminal_entry_count"]
                ),
                "violation_steps": {
                    key: float(value)
                    for key, value in info.items()
                    if key.endswith("violation_steps") and float(value) > 0.0
                },
                "qp_zero_fallbacks": int(info["hybrid_qp_zero_fallbacks"]),
            }
        )
        env.close()
        last = records[-1]
        print(
            f"seed={last['seed']} policy={args.policy} completed={last['completed']} "
            f"t={last['final_time_s']:.1f}s commit={last['commit_time_s']} "
            f"latch={last['terminal_region_active']:.0f} "
            f"illegal={last['illegal_terminal_entry_count']:.0f}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
