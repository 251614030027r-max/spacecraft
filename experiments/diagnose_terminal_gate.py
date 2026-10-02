"""Which states does the MPC enforce the approach corridor on?

The lower layer's terminal rows -- corridor axial, the eight corridor lateral
facets, terminal total speed and closing speed -- are switched on by
``controllers.mpc.constraints._predicted_terminal_active``.  That predicate is
a **half-space**::

    terminal_latched or approach_axis @ (p - port) < entry_port_axial_distance_m

With the task's own numbers (``port = (-1.5, 0, 0)``, ``axis = (-1, 0, 0)``,
``entry_port_axial_distance_m = 4.5``) the second clause is ``p_x > -6``, and it
carries **no bound on range**.  A chaser 16.75 m from the target with a 15 m
lateral offset therefore has the approach corridor imposed on it, and the
linearised corridor facet it violates asks for about 4.0 normalised units of
slack against a ``constraint_slack_limit`` of 2.0.  The QP reports
``infeasible``, ``infeasible_fallback="zero"`` commands zero wrench, and at
``Lambda > 1`` a chaser with no wrench is handed to ``omega * r``.

The truth side does not agree with this.  ``normalized_precapture_truth_margins``
gates the same rows on ``terminal_latched`` alone, and the environment scores a
corridor violation only once ``_terminal_region_entered`` has latched on a legal
entry-disc crossing.  So the optimiser is refusing to solve because of a region
the task never judges.

The repository now ships that bound; ``--terminal-gate half_space`` restores
the pre-fix predicate so the comparison stays runnable. The bound is::

    ... and ||p|| <= port_radius + entry_port_axial_distance_m   (= 6.0 m)

Both terms are frozen task parameters -- that distance is where the entry plane
sits relative to the target centre -- so the treatment introduces no tunable
constant.  The reference, horizon, solver, slack cap and fallback are untouched,
which makes this a single interpretable factor against the repository default.

The upper action is the fixed setpoint throughout (``a = (+1, *)`` under
``arrival_condition``), so this measures the lower layer alone.

Reproduce::

    python -B -m experiments.diagnose_terminal_gate --horizon 20 \\
        --seeds 262000 262001 262002 262003 262004 262005 \\
                262006 262007 262008 262009 262010 262011 \\
        --output logs/terminal_gate/gate_h20.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import controllers.mpc.constraints as constraints
from controllers.mpc.prediction import relative_to_vector
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_planning_environment_config
from env.task import PrecaptureTaskConfig

_REPOSITORY_GATE = constraints._predicted_terminal_active
entry_plane_radius_m = constraints.entry_plane_radius_m


def install_half_space_gate() -> None:
    """Restore the pre-fix gate: the axial half-space with no bound on range.

    The repository now ships the range-bounded gate, so this is what the
    ``half_space`` arm measures -- the historical behaviour, kept runnable so
    the comparison in ``docs/TERMINAL_GATE_DEFECT.md`` stays reproducible.
    """

    def gated(position: Any, task: PrecaptureTaskConfig, *, terminal_latched: bool) -> bool:
        port_displacement = np.asarray(position, dtype=np.float64) - task.port_position
        port_axial_distance = float(task.approach_axis @ port_displacement)
        return bool(
            terminal_latched
            or port_axial_distance < task.entry_port_axial_distance_m
        )

    constraints._predicted_terminal_active = gated


def restore_gate() -> None:
    constraints._predicted_terminal_active = _REPOSITORY_GATE


def run_episode(seed: int, horizon: int) -> dict[str, Any]:
    env = PrecaptureHybridEnv(
        environment_config=precapture_planning_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=horizon,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
        ),
    )
    task = env.environment_config.precapture_task
    _, info = env.reset(seed=seed)
    desired = np.asarray(task.desired_position, dtype=np.float64)
    infeasible_steps = 0
    first_infeasible_time_s: float | None = None
    consecutive_zero = 0
    max_consecutive_zero = 0
    force_impulse = 0.0
    torque_impulse = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        waypoint = env.waypoint_from_action(env.action_for_waypoint(desired))
        for _ in range(env.hybrid_config.decision_period_steps):
            assert env.env.relative is not None
            assert env.env.target_state is not None
            wrench, diagnostics = env.controller.command(
                relative_to_vector(env.env.relative),
                target_state=env.env.target_state,
                time_seconds=env.env.time_seconds,
                terminal_latched=bool(info["terminal_region_active"]),
                external_reference=waypoint,
            )
            if str(diagnostics.status).startswith("infeasible"):
                infeasible_steps += 1
                if first_infeasible_time_s is None:
                    first_infeasible_time_s = round(float(env.env.time_seconds), 1)
            if np.count_nonzero(wrench) == 0:
                consecutive_zero += 1
                max_consecutive_zero = max(max_consecutive_zero, consecutive_zero)
            else:
                consecutive_zero = 0
            force_impulse += (
                float(np.linalg.norm(wrench[3:])) * env.environment_config.dt_s
            )
            torque_impulse += (
                float(np.linalg.norm(wrench[:3])) * env.environment_config.dt_s
            )
            _, _, terminated, truncated, info = env.env.step(
                wrench_to_normalized(
                    GeneralizedForce.from_vector(wrench),
                    max_torque_per_axis_nm=(
                        env.environment_config.max_torque_per_axis_nm
                    ),
                    max_force_per_axis_n=env.environment_config.max_force_per_axis_n,
                )
            )
            steps += 1
            if terminated or truncated:
                break
    return {
        "seed": seed,
        "horizon": horizon,
        "completed": bool(info["completed"]),
        "end_time_s": round(float(env.env.time_seconds), 2),
        "steps": steps,
        "qp_infeasible_steps": infeasible_steps,
        "first_infeasible_time_s": first_infeasible_time_s,
        "max_consecutive_zero_wrench_steps": max_consecutive_zero,
        "force_impulse_n_s": round(force_impulse, 3),
        "torque_impulse_nm_s": round(torque_impulse, 4),
        "illegal_terminal_entry_count": int(info["illegal_terminal_entry_count"]),
        "terminal_region_active": bool(info["terminal_region_active"]),
        "final_range_m": round(float(info["target_center_distance_m"]), 3),
        "final_fov_margin_rad": round(float(info["fov_margin_rad"]), 4),
        "time_failure": bool(info["time_failure"]),
        "distance_failure": bool(info["distance_failure"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--terminal-gate",
        choices=["half_space", "proximity"],
        default="proximity",
        help="proximity is the repository behaviour; half_space restores the "
        "pre-fix gate for comparison",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    radius_m = entry_plane_radius_m(PrecaptureTaskConfig())
    if args.terminal_gate == "half_space":
        install_half_space_gate()
    try:
        records = []
        for seed in args.seeds:
            record = run_episode(seed, args.horizon)
            record["terminal_gate"] = args.terminal_gate
            record["proximity_radius_m"] = radius_m
            records.append(record)
            print(json.dumps(record), flush=True)
            args.output.write_text(json.dumps(records, indent=2))
    finally:
        restore_gate()
    completed = sum(record["completed"] for record in records)
    print(
        f"{args.terminal_gate} h{args.horizon}: "
        f"{completed}/{len(records)} completed",
        flush=True,
    )


if __name__ == "__main__":
    main()
