"""Limited closed-loop admission of the entry-speed authority (supplementary instruction 2026-10-15).

Earliest-feasible rule (earliest arrival, then keep), kappa 0.7, entry speed v_in in
{0.02, 0.06, 0.10} m/s, on n development openings (deterministic, serial per episode). Per
episode and decision: time, G* status, new-plan trigger, entry speed requested / applied /
of the live plan, descent flag, relative ECI position, the first commanded wrench and all
20 commanded wrenches of the decision.

Four checks, against the 0.06 m/s run of the same opening:

1. compatibility (hard): 0.06 reproduces the published rule rows (``--reference``, the
   frozen-rule evidence) value for value;
2. effectiveness (hard): differences of trajectory and control *before the descent switch*
   (first decision at which either run is in G*'s descent mode; G*'s switch rule is unchanged),
   of the switch time itself, and of the outcome (time, delta-v, return); compared with the
   numerical floor (the 0.06 replay against the reference, which is exactly 0);
3. basic safety (hard): truth violations, holds, recoveries, deadline events, zero-input
   fallbacks, numerical failures and new plans per episode, against 0.06;
4. state dependence (descriptive): per-opening ranking of the three speeds by discounted
   return, the best uniform speed, and the per-opening best (an upper bound of what a
   state-dependent choice of these three could gain on these openings, not a target).

    python -B -m experiments.t1_entry_speed_check --openings experiments/t1_dev_openings_295000.json --first 8 \
        --reference evidence/T1_dev_rule_295000/episodes.jsonl.gz --workers 4 --output entry_speed_admission.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

SPEEDS = (0.02, 0.06, 0.10)
NOMINAL = 0.06


def run(opening: int, speed: float) -> dict:
    import torch

    from env.t1_gym import T1GStarEnv
    from experiments.t1_dev_eval import rule_action

    torch.set_num_threads(1)
    env = T1GStarEnv(openings=[opening])
    env.reset()
    wrenches: list[np.ndarray] = []
    command = env.env.controller.command

    def recording(*args, **kwargs):
        wrench, diagnostics = command(*args, **kwargs)
        wrenches.append(np.asarray(wrench, dtype=np.float64).copy())
        return wrench, diagnostics

    env.env.controller.command = recording  # type: ignore[method-assign]
    gamma = env.env.config.reward.gamma
    log, ret, k = [], 0.0, 0
    while True:
        descending = bool(env.generator._descending(env.env, env.env.metrics()))
        q = env.env.chaser_state.position - env.env.target_state.position
        start = len(wrenches)
        _, reward, terminated, truncated, info = env.step(rule_action(env, 0.7, speed))
        x = info["execution"]
        w = np.asarray(wrenches[start:])
        log.append({"t": 2.0 * k, "descent_at_start": descending, "q": q.tolist(), "status": x["plan_status"],
                    "trigger": x["new_plan_trigger"], "deadline": x["deadline"],
                    "v_req": x["entry_speed_request_m_s"], "v_applied": x["entry_speed_applied"],
                    "v_plan": x["entry_speed_plan_m_s"], "tau_eff": x["tau_effective_s"],
                    "zero_fallbacks": x["zero_fallbacks"], "u": w.tolist(), "r": float(reward)})
        ret += gamma ** k * reward
        k += 1
        if terminated or truncated:
            break
    s = info["episode_summary"]
    statuses = [d["status"] for d in log]
    return {
        "opening": opening, "speed": speed, "event": s["event"], "time_s": s["time_s"],
        "delta_v_m_s": s["delta_v_m_s"], "discounted_return": ret, "violations": s.get("violations", {}),
        "numerical_error": info.get("numerical_error"),
        "new_plans": sum(st.startswith(("optimal", "recovery")) for st in statuses),
        "recoveries": sum(st.startswith("recovery") for st in statuses),
        "holds": sum(st.startswith("hold") for st in statuses),
        "deviation_replans": sum(d["trigger"] == "deviation" for d in log),
        "deadline_events": sum(d["deadline"] != "none" for d in log),
        "zero_fallbacks": sum(d["zero_fallbacks"] for d in log),
        "entry_speeds_applied": [d["v_plan"] for d in log if d["v_applied"]],
        "first_descent_decision": next((i for i, d in enumerate(log) if d["descent_at_start"]), None),
        "reference_jump_max_m": s.get("reference_jump_max_m"), "force_saturated_fraction": s.get("force_saturated_fraction"),
        "log": log,
    }


def pair(base: dict, other: dict) -> dict:
    lb, lo = base["log"], other["log"]
    switches = [x for x in (base["first_descent_decision"], other["first_descent_decision"]) if x is not None]
    k_switch = min(switches) if switches else min(len(lb), len(lo))
    n = min(k_switch, len(lb), len(lo))
    pos = [float(np.linalg.norm(np.subtract(lo[i]["q"], lb[i]["q"]))) for i in range(n)]
    ctrl = []
    for i in range(n):
        ub, uo = np.asarray(lb[i]["u"]), np.asarray(lo[i]["u"])
        m = min(len(ub), len(uo))
        if m:
            ctrl.append(float(np.max(np.linalg.norm(uo[:m] - ub[:m], axis=1))))
    force = lambda log, j: float(sum(np.linalg.norm(np.asarray(d["u"])[:, 3:], axis=1).sum() for d in log[:j]))  # noqa: E731
    return {
        "speed": other["speed"],
        "first_descent_decision": other["first_descent_decision"], "first_descent_decision_0p06": base["first_descent_decision"],
        "decisions_compared_before_switch": n,
        "position_diff_before_switch_max_m": max(pos) if pos else 0.0,
        "position_diff_at_switch_m": pos[-1] if pos else 0.0,
        "control_diff_before_switch_max": max(ctrl) if ctrl else 0.0,
        "force_impulse_before_switch_ns": 0.1 * force(lo, n), "force_impulse_before_switch_0p06_ns": 0.1 * force(lb, n),
        "d_time_s": other["time_s"] - base["time_s"], "d_delta_v_m_s": other["delta_v_m_s"] - base["delta_v_m_s"],
        "d_return": other["discounted_return"] - base["discounted_return"],
        "same_event": other["event"] == base["event"],
    }


def _reference(path: Path) -> dict[int, dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {int(r.get("opening", r.get("seed"))): r for r in rows}


def analyse(runs: dict[tuple[int, float], dict], reference: dict[int, dict] | None) -> dict:
    openings = sorted({o for o, _ in runs})
    out: dict = {"openings": openings, "speeds": SPEEDS, "per_opening": {}, "checks": {}}
    compat, effective, safety = [], [], []
    for o in openings:
        base = runs[(o, NOMINAL)]
        row = {"outcomes": {str(v): {k: runs[(o, v)][k] for k in (
            "event", "time_s", "delta_v_m_s", "discounted_return", "violations", "numerical_error", "new_plans",
            "recoveries", "holds", "deviation_replans", "deadline_events", "zero_fallbacks", "entry_speeds_applied",
            "first_descent_decision", "reference_jump_max_m", "force_saturated_fraction")} for v in SPEEDS},
            "vs_0p06": {str(v): pair(base, runs[(o, v)]) for v in SPEEDS if v != NOMINAL}}
        if reference is not None and o in reference:
            ref = reference[o]
            same = all(base[k] == ref[k] for k in ("event", "time_s", "delta_v_m_s", "discounted_return"))
            row["compatibility_vs_reference"] = same
            compat.append(same)
        ranking = sorted(SPEEDS, key=lambda v: -runs[(o, v)]["discounted_return"])
        row["ranking_by_return"] = [str(v) for v in ranking]
        out["per_opening"][o] = row
        for p in row["vs_0p06"].values():
            effective.append(p)
        for v in SPEEDS:
            r = runs[(o, v)]
            safety.append({"opening": o, "speed": v, "unsafe": r["event"] == "unsafe", "numerical": r["event"] == "numerical_failure",
                           "holds": r["holds"], "recoveries": r["recoveries"], "zero_fallbacks": r["zero_fallbacks"],
                           "extra_new_plans_vs_0p06": r["new_plans"] - base["new_plans"],
                           "extra_deviation_replans_vs_0p06": r["deviation_replans"] - base["deviation_replans"],
                           "deadline_events": r["deadline_events"]})
    mean = {str(v): float(np.mean([runs[(o, v)]["discounted_return"] for o in openings])) for v in SPEEDS}
    best_uniform = max(SPEEDS, key=lambda v: mean[str(v)])
    oracle = float(np.mean([max(runs[(o, v)]["discounted_return"] for v in SPEEDS) for o in openings]))
    out["checks"]["1_compatibility"] = {"openings_compared": len(compat), "all_identical": bool(compat) and all(compat)}
    out["checks"]["2_effectiveness"] = {
        "pairs": len(effective),
        "pairs_with_control_diff_before_switch_gt_0p01": sum(p["control_diff_before_switch_max"] > 0.01 for p in effective),
        "pairs_with_position_diff_before_switch_gt_0p05m": sum(p["position_diff_before_switch_max_m"] > 0.05 for p in effective),
        "pairs_with_abs_d_time_ge_2s": sum(abs(p["d_time_s"]) >= 2.0 for p in effective),
        "pairs_with_abs_d_delta_v_ge_0p02": sum(abs(p["d_delta_v_m_s"]) >= 0.02 for p in effective),
        "pairs_with_switch_decision_changed": sum(p["first_descent_decision"] != p["first_descent_decision_0p06"] for p in effective),
        "numerical_floor": "0.06 replay vs reference: identical (check 1); runs are deterministic",
    }
    out["checks"]["3_safety"] = {
        "unsafe": {str(v): sum(s["unsafe"] for s in safety if s["speed"] == v) for v in SPEEDS},
        "numerical_failures": sum(s["numerical"] for s in safety),
        "holds": {str(v): sum(s["holds"] for s in safety if s["speed"] == v) for v in SPEEDS},
        "recoveries": {str(v): sum(s["recoveries"] for s in safety if s["speed"] == v) for v in SPEEDS},
        "zero_fallbacks": {str(v): sum(s["zero_fallbacks"] for s in safety if s["speed"] == v) for v in SPEEDS},
        "new_plans_total": {str(v): sum(runs[(o, v)]["new_plans"] for o in openings) for v in SPEEDS},
        "deviation_replans_total": {str(v): sum(runs[(o, v)]["deviation_replans"] for o in openings) for v in SPEEDS},
        "deadline_events": {str(v): sum(s["deadline_events"] for s in safety if s["speed"] == v) for v in SPEEDS},
        "captures": {str(v): sum(runs[(o, v)]["event"] == "success" for o in openings) for v in SPEEDS},
    }
    out["checks"]["4_state_dependence"] = {
        "mean_return": mean, "best_uniform_speed": best_uniform, "per_opening_best_mean_return": oracle,
        "per_opening_best_minus_best_uniform": oracle - mean[str(best_uniform)],
        "openings_where_best_uniform_is_not_best": sum(out["per_opening"][o]["ranking_by_return"][0] != str(best_uniform) for o in openings),
        "note": "descriptive: an upper bound for a state-dependent choice among the three speeds on these openings",
    }
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--first", type=int, default=8)
    parser.add_argument("--reference", type=Path, default=None, help="published frozen-rule rows (kappa 0.7, 0.06 m/s)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=None, help="also write each episode (with its log) here")
    args = parser.parse_args(argv)
    seeds = json.loads(args.openings.read_text())["seeds"][: args.first]
    jobs = [(o, v) for o in seeds for v in SPEEDS]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.starmap(run, jobs)
    else:
        results = [run(o, v) for o, v in jobs]
    runs = {(r["opening"], r["speed"]): r for r in results}
    if args.runs_dir is not None:
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        for (o, v), r in runs.items():
            (args.runs_dir / f"{o}_v{v:.2f}.json").write_text(json.dumps(r, default=float))
    result = analyse(runs, _reference(args.reference) if args.reference else None)
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result["checks"], indent=1, default=float))
    return result


if __name__ == "__main__":
    main()
