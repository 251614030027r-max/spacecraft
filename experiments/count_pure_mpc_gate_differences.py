"""Run the evaluator with the back-hemisphere gate, counting predicate calls
where the pre-fix gate (commit 86b5517) would have answered differently.

Zero differing calls over an evaluation block means every MPC solve in it saw
the same constraint rows as before the fix, so the fix cannot have changed
those episodes on any platform. Arguments are passed through to
``experiments.evaluate_hybrid_policy``; the counts are written next to
``--output`` as ``<output>.gatecount.json``.

    python -B experiments/count_pure_mpc_gate_differences.py --episodes 12 --seed 262000 --horizon 35 --parametrization arrival_condition --adaptive-task --control desired_pose --output OUT.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from controllers.mpc import constraints as C

_fixed = C._predicted_terminal_active
counts = {"calls": 0, "differ": 0, "min_back_range_m": None}

def _old(position, task, *, terminal_latched):
    if terminal_latched:
        return True
    axial = float(task.approach_axis @ (position - task.port_position))
    if axial >= task.entry_port_axial_distance_m:
        return False
    return bool(float(np.linalg.norm(np.asarray(position, dtype=np.float64))) <= C.entry_plane_radius_m(task))

def counted(position, task, *, terminal_latched):
    new = _fixed(position, task, terminal_latched=terminal_latched)
    counts["calls"] += 1
    if new != _old(position, task, terminal_latched=terminal_latched):
        counts["differ"] += 1
    axial = float(task.approach_axis @ (np.asarray(position) - task.port_position))
    if axial < 0.0 and not terminal_latched:
        r = float(np.linalg.norm(position))
        if counts["min_back_range_m"] is None or r < counts["min_back_range_m"]:
            counts["min_back_range_m"] = r
    return new

C._predicted_terminal_active = counted
out = sys.argv[sys.argv.index("--output") + 1]
from experiments import evaluate_hybrid_policy as E
sys.argv = ["evaluate_hybrid_policy"] + sys.argv[1:]
try:
    E.main()
finally:
    json.dump(counts, open(out + ".gatecount.json", "w"))
    print("GATECOUNT", counts)
