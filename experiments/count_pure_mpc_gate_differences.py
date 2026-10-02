"""Count predicted-terminal gate decisions that differ from earlier versions.

Runs ``experiments.evaluate_hybrid_policy`` with the repository's gate and, at
every call, also evaluates the two earlier gates:

    pre_7555185   entry radius + axial half-space         (commit 86b5517)
    at_7555185    ... + back hemisphere excluded          (commit 7555185)

Zero differing calls over an evaluation block means every MPC solve in it saw
the same constraint rows as under that earlier gate, so the change cannot have
altered those episodes on any platform. Also records the closest unlatched
approach behind the port plane and the largest unlatched lateral offset seen
inside the old gate region (the ring the cylinder condition removes).
Arguments pass through to the evaluator; counts go to
``<output>.gatecount.json``.

    python -B experiments/count_pure_mpc_gate_differences.py --episodes 12 --seed 262000 --horizon 35 --parametrization arrival_condition --adaptive-task --control desired_pose --output OUT.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from controllers.mpc import constraints as C

_current = C._predicted_terminal_active
counts = {
    "calls": 0,
    "differ_vs_pre_7555185": 0,
    "differ_vs_at_7555185": 0,
    "min_unlatched_back_range_m": None,
    "max_unlatched_lateral_in_old_region_m": None,
}


def _pre_7555185(position, task, *, terminal_latched):
    if terminal_latched:
        return True
    axial = float(task.approach_axis @ (np.asarray(position) - task.port_position))
    if axial >= task.entry_port_axial_distance_m:
        return False
    return bool(float(np.linalg.norm(position)) <= C.entry_plane_radius_m(task))


def _at_7555185(position, task, *, terminal_latched):
    if terminal_latched:
        return True
    axial = float(task.approach_axis @ (np.asarray(position) - task.port_position))
    if axial < 0.0:
        return False
    return _pre_7555185(position, task, terminal_latched=False)


def _update_extreme(key, value, pick):
    counts[key] = value if counts[key] is None else pick(counts[key], value)


def counted(position, task, *, terminal_latched):
    current = _current(position, task, terminal_latched=terminal_latched)
    counts["calls"] += 1
    counts["differ_vs_pre_7555185"] += int(
        current != _pre_7555185(position, task, terminal_latched=terminal_latched)
    )
    counts["differ_vs_at_7555185"] += int(
        current != _at_7555185(position, task, terminal_latched=terminal_latched)
    )
    if not terminal_latched:
        displacement = np.asarray(position, dtype=np.float64) - task.port_position
        axial = float(task.approach_axis @ displacement)
        if axial < 0.0:
            _update_extreme("min_unlatched_back_range_m", float(np.linalg.norm(position)), min)
        if _pre_7555185(position, task, terminal_latched=False):
            lateral = float(np.linalg.norm(displacement - axial * task.approach_axis))
            _update_extreme("max_unlatched_lateral_in_old_region_m", lateral, max)
    return current


C._predicted_terminal_active = counted
out = sys.argv[sys.argv.index("--output") + 1]
from experiments import evaluate_hybrid_policy as E  # noqa: E402

sys.argv = ["evaluate_hybrid_policy"] + sys.argv[1:]
try:
    E.main()
finally:
    Path(out + ".gatecount.json").write_text(json.dumps(counts))
    print("GATECOUNT", counts)
