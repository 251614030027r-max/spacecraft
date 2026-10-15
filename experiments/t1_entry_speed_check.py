"""Admission B of the entry-speed authority (plan 2026-10-15, sec. 5): does a[2] reach the loop?

For each opening and entry speed in {0.02, 0.06, 0.10} m/s (a[2] = -1, 0, +1), the first decision
of the earliest-feasible rule (kappa 0.7) is taken and compared with the 0.06 m/s one:

* the accepted plan: status, arrival time, entry speed, terminal velocity (target frame and its
  inward component), planned delta-v, largest position difference along the plan;
* the shared MPC: the first commanded wrench and the 20 commanded wrenches of the 2 s decision;
* the 2 s truth step: end position difference.

Closed-loop outcomes (capture, time, delta-v, safety, reference jumps) come from
``experiments.t1_dev_eval --policy rule:earliest_k0.7_v<speed>`` on the same openings and are
read in with ``--closed-loop DIR`` (one DIR per speed, ``speed=dir``).

    python -B -m experiments.t1_entry_speed_check --openings 295003,295004 --output entry_speed_check.json \
        --closed-loop 0.02=eval/v02,0.06=eval/v06,0.10=eval/v10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.t1_gstar import _plan_position
from env.t1_gym import T1GStarEnv
from experiments.t1_dev_eval import rule_action

SPEEDS = (0.02, 0.06, 0.10)


def first_decision(opening: int, speed: float) -> dict:
    env = T1GStarEnv(openings=[opening])
    env.reset()
    wrenches: list[np.ndarray] = []
    command = env.env.controller.command

    def recording(*args, **kwargs):
        wrench, diagnostics = command(*args, **kwargs)
        wrenches.append(np.asarray(wrench, dtype=np.float64).copy())
        return wrench, diagnostics

    env.env.controller.command = recording  # type: ignore[method-assign]
    action = rule_action(env, 0.7, speed)
    _, reward, _, _, info = env.step(action)
    g = env.generator
    plan = g.plan
    out = {"opening": opening, "speed_m_s": speed, "action": [float(v) for v in action], "reward": float(reward),
           "status": info["execution"]["plan_status"], "tau_effective_s": g.tau_effective_s,
           "entry_speed_plan_m_s": info["execution"]["entry_speed_plan_m_s"],
           "wrenches": np.asarray(wrenches), "end_position": env.env.chaser_state.position.copy()}
    if plan is not None:
        n = len(plan["f"])
        out["plan_delta_v_m_s"] = plan["delta_v_m_s"]
        out["plan_terminal_velocity"] = np.asarray(plan["v"][-1])
        out["plan_positions"] = np.array([_plan_position(plan, k * plan["dt"], g.planner.mass) for k in range(n + 1)])
    return out


def compare(base: dict, other: dict) -> dict:
    row = {
        "speed_m_s": other["speed_m_s"], "status": other["status"], "tau_effective_s": other["tau_effective_s"],
        "entry_speed_plan_m_s": other["entry_speed_plan_m_s"], "plan_delta_v_m_s": other.get("plan_delta_v_m_s"),
        "first_wrench": other["wrenches"][0].round(6).tolist(),
        "first_wrench_diff_norm": float(np.linalg.norm(other["wrenches"][0] - base["wrenches"][0])),
        "decision_wrench_diff_max": float(np.max(np.linalg.norm(other["wrenches"] - base["wrenches"], axis=1))),
        "end_position_diff_m": float(np.linalg.norm(other["end_position"] - base["end_position"])),
        "reward": other["reward"],
    }
    if "plan_positions" in other and "plan_positions" in base:
        row["terminal_velocity_diff_m_s"] = float(np.linalg.norm(other["plan_terminal_velocity"] - base["plan_terminal_velocity"]))
        if len(other["plan_positions"]) == len(base["plan_positions"]) and other["tau_effective_s"] == base["tau_effective_s"]:
            row["plan_position_diff_max_m"] = float(np.max(np.linalg.norm(other["plan_positions"] - base["plan_positions"], axis=1)))
    return row


def closed_loop(directory: Path, opening: int) -> dict | None:
    path = directory / f"{opening}.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    keys = ("event", "time_s", "delta_v_m_s", "discounted_return", "violations", "end_phase", "min_margins",
            "reference_jump_max_m", "force_saturated_fraction", "qp_zero_fallback_steps", "new_plans", "kept",
            "replan_deviation", "entry_speed_applied")
    return {k: r.get(k) for k in keys}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--closed-loop", default="", help="speed=dir,... (t1_dev_eval output directories)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    loops = {float(k): Path(v) for k, v in (x.split("=", 1) for x in args.closed_loop.split(",") if x)}
    result: dict = {"openings": {}}
    for opening in (int(x) for x in args.openings.split(",")):
        runs = {speed: first_decision(opening, speed) for speed in SPEEDS}
        base = runs[0.06]
        result["openings"][opening] = {
            "first_decision": {str(s): compare(base, runs[s]) for s in SPEEDS},
            "closed_loop": {str(s): closed_loop(loops[s], opening) for s in SPEEDS if s in loops},
        }
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))
    return result


if __name__ == "__main__":
    main()
