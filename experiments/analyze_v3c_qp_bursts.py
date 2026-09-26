"""Summarise the v3c 262422 QP zero-fallback diagnosis into one artifact.

Inputs:
- per-episode JSON written by ``experiments.probe_v3_qp_policy`` (stochastic
  rollouts of the stopped run's 15k / 20k checkpoints);
- the three v3c ``train.monitor.csv`` files from the diagnostic handoff;
- optional ``*.gatecount.json`` files from a Pure MPC evaluation run with the
  back-hemisphere gate in place, counting predicate calls where the pre-fix
  gate would have answered differently.

It also evaluates the linearised corridor rows at the recorded burst geometry,
to show that behind the port plane they exceed the QP slack cap.

    python -B -m experiments.analyze_v3c_qp_bursts --rollouts DIR --monitors DIR [--gatecounts DIR] --output eval/adp/v3c_262422_qp_burst_diagnosis.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from controllers.mpc.config import precapture_mpc_config
from controllers.mpc.constraints import normalized_precapture_constraint_margins
from env.task import PrecaptureTaskConfig

BURST_STEPS = 20


def corridor_rows(range_m: float, port_axial_m: float, task: PrecaptureTaskConfig, facets: int) -> dict:
    """Worst corridor rows at a point with the given range and port-axial distance, forced active."""

    axis = np.asarray(task.approach_axis, dtype=np.float64)
    port = np.asarray(task.port_position, dtype=np.float64)
    # A point in the plane spanned by the axis and an orthogonal direction.
    lateral_direction = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    lateral_direction /= np.linalg.norm(lateral_direction)
    along = float(axis @ port) + port_axial_m
    lateral = float(np.sqrt(max(range_m**2 - along**2, 0.0)))
    position = along * axis + lateral * lateral_direction
    state = np.zeros(12)
    state[3:6] = position  # identity attitude, so rho == position
    margins = normalized_precapture_constraint_margins(
        state,
        task,
        target_angular_velocity_rad_s=np.array([0.0, 0.0, 0.041231]),
        terminal_latched=True,
        corridor_facets=facets,
    )
    return {
        "range_m": range_m,
        "port_axial_m": port_axial_m,
        "axial_row": float(margins[4]),
        "worst_facet_row": float(np.min(margins[5 : 5 + facets])),
    }


def monitors(directory: Path) -> dict:
    out = {}
    for path in sorted(directory.glob("v3c_*_train.monitor.csv")):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        fallbacks = [float(r["hybrid_v3_episode_qp_zero_fallbacks"]) for r in rows]
        steps = [float(r["hybrid_v3_episode_control_steps"]) for r in rows]
        bursts = [(i, int(f)) for i, f in enumerate(fallbacks) if f >= BURST_STEPS]
        out[path.name.split("_train")[0]] = {
            "episodes": len(rows),
            "decisions": int(sum(float(r["l"]) for r in rows)),
            "rate_all": sum(fallbacks) / sum(steps),
            "rate_last100": sum(fallbacks[-100:]) / sum(steps[-100:]),
            "burst_episodes": [{"episode_index": i, "fallback_steps": f} for i, f in bursts],
            "share_of_fallbacks_in_bursts": sum(f for _, f in bursts) / max(sum(fallbacks), 1.0),
            "completed_last100": sum(r["completed"] == "True" for r in rows[-100:]),
        }
    return out


def rollouts(directory: Path) -> dict:
    episodes = []
    for path in sorted(directory.glob("k*_*.json")):
        data = json.loads(path.read_text())
        events = data["fallback_events"]
        row = {
            "file": path.name,
            "checkpoint": path.name.split("_")[0],
            "seed": data["seed"],
            "completed": data["completed"],
            "failure": data["failure"],
            "decisions": data["decisions"],
            "fallback_steps": data["fallback_steps"],
        }
        if events:
            axial = [e["port_axial_m"] for e in events]
            ranges = [e["range_m"] for e in events]
            row.update(
                {
                    "burst_t_s": [events[0]["t_s"], events[-1]["t_s"]],
                    "range_m_first_last": [ranges[0], ranges[-1]],
                    "range_m_min_max": [min(ranges), max(ranges)],
                    "steps_behind_port_plane": sum(a < 0.0 for a in axial),
                    "steps_gate_active_at_current_position": sum(e["predicted_terminal_active"] for e in events),
                    "steps_latched": sum(e["terminal_latched"] for e in events),
                    "commitment_values": sorted({round(e["commitment"], 3) for e in events}),
                    "reference_radius_m_min": min(e["reference_radius_m"] for e in events),
                    "reference_off_axis_deg_min_max": [
                        min(e["reference_off_axis_deg"] for e in events),
                        max(e["reference_off_axis_deg"] for e in events),
                    ],
                    "min_truth_normalized_margin": min(min(e["truth_margins"]) for e in events),
                }
            )
        episodes.append(row)
    total_fallbacks = sum(e["fallback_steps"] for e in episodes)
    total_steps = sum(20 * e["decisions"] for e in episodes)
    return {
        "episodes": len(episodes),
        "fallback_steps": total_fallbacks,
        "control_steps": total_steps,
        "rate": total_fallbacks / total_steps if total_steps else None,
        "episodes_with_fallback": [e for e in episodes if e["fallback_steps"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--monitors", type=Path, required=True)
    parser.add_argument("--gatecounts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = PrecaptureTaskConfig()
    facets = precapture_mpc_config().corridor_facets
    slack_limit = precapture_mpc_config().constraint_slack_limit
    result = {
        "monitors": monitors(args.monitors),
        "rollouts_262422": rollouts(args.rollouts),
        "corridor_rows_forced_active": {
            "slack_limit": slack_limit,
            "points": [
                corridor_rows(6.652, -4.375, task, facets),  # k15 264038 burst start
                corridor_rows(4.437, -3.039, task, facets),  # k15 264038 burst end
                corridor_rows(5.37, 0.5, task, facets),  # k20 264005 (front side)
            ],
        },
    }
    if args.gatecounts is not None:
        counts = {p.name: json.loads(p.read_text()) for p in sorted(args.gatecounts.glob("*.gatecount.json"))}
        result["pure_mpc_gate_differences"] = {
            "per_block": counts,
            "calls": sum(c["calls"] for c in counts.values()),
            "differ": sum(c["differ"] for c in counts.values()),
        }
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rollouts_262422"}, indent=1))


if __name__ == "__main__":
    main()
