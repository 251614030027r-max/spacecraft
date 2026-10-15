"""Readout of the T1-L three-arm experiment (plan 2026-10-15): B2, N0, NDelta, rules, Pure.

Descriptive tables only (the run order decides nothing on them):

* outcomes per arm and mark (capture / timeout / unsafe, mean discounted return, delta-v,
  time with failures at 300 s), on the openings the mark evaluated;
* paired per-opening differences: each mark against the arm's own 0k, the strong fixed
  reference and Pure; and between arms at the same mark (N0 - B2, NDelta - N0, NDelta - B2);
* the task decisions of the deterministic policy: reviews per step, first-review anchor and
  delay, new plans by trigger, plan events, entry speed applied, kappa;
* critic at review states of the deterministic episodes: min_j Q(taken) - realised discounted
  return-to-go, by decision kind; Q of the one-factor neighbours minus Q(taken);
* training logs: reviews and plan events per 1k steps (behaviour policy, Trap 11), the
  NDelta counterfactual by kind (candidate / physically distinct / gate-valid, and the
  prediction differences of the valid ones), relative-term gradient share and valid fraction,
  critic audits, wall time per step.

    python -B -m experiments.t1_l_readout --runs b2=<dir>,n0=<dir>,ndelta=<dir> --rules <rules root> \
        --strong rule:earliest_k1.0_v0.02 --pure <pure dir> --output readout.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.t1_fix_readout import _jsonl, _quantiles, outcome_table

GAMMA = 0.999


def rows_of(directory: Path) -> dict[int, dict]:
    rows = {}
    for f in directory.glob("*.json"):
        if f.stem.isdigit():
            rows[int(f.stem)] = json.loads(f.read_text())
    if not rows:  # t1_gstar_probe output (Pure): "<row>__<seed>.json"
        for f in directory.glob("*__*.json"):
            r = json.loads(f.read_text())
            rows[int(r.get("seed", f.stem.split("__")[-1]))] = r
    return rows


def paired(a: dict[int, dict], b: dict[int, dict]) -> dict | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = np.array([a[o]["discounted_return"] - b[o]["discounted_return"] for o in common])
    cap = sum(a[o]["event"] == "success" for o in common) - sum(b[o]["event"] == "success" for o in common)
    return {"openings": len(common), "mean_return_diff": float(d.mean()), "median_return_diff": float(np.median(d)),
            "openings_better": int((d > 0).sum()), "openings_worse": int((d < 0).sum()), "capture_diff": int(cap)}


def decisions(rows: dict[int, dict]) -> dict:
    triggers, kinds = Counter(), Counter()
    first_delay, first_anchor, first_eff, v_applied, kappa, per_step = [], [], [], [], [], []
    for r in rows.values():
        log = r.get("decision_log", [])
        if not log:
            continue
        per_step.append(r["reviews"] / max(r["steps"], 1))
        if log[0].get("delay_s") is not None:
            first_delay.append(log[0]["delay_s"])
            first_anchor.append(log[0]["tau_min_feasible_s"])
            first_eff.append(log[0]["tau_effective_s"])
        for s in log:
            if s.get("trigger"):
                triggers[s["trigger"]] += 1
            if s["reviewed"]:
                kinds["descent" if s["descent"] else s["request_kind"]] += 1
            if s.get("entry_speed_applied"):
                v_applied.append(s["entry_speed_request_m_s"])
            kappa.append(s["kappa"])
    return {"reviews_per_step": _quantiles(per_step), "first_delay_s": _quantiles(first_delay),
            "first_tau_min_feasible_s": _quantiles(first_anchor), "first_tau_effective_s": _quantiles(first_eff),
            "review_kinds": dict(kinds), "new_plans_by_trigger": dict(triggers),
            "plan_events": int(sum(r.get("plan_events", 0) for r in rows.values())),
            "entry_speed_applied": _quantiles(v_applied), "kappa": _quantiles(kappa)}


def critic(rows: dict[int, dict]) -> dict | None:
    gaps, neighbour = defaultdict(list), defaultdict(list)
    found = False
    for r in rows.values():
        log = r.get("decision_log", [])
        g, to_go = 0.0, []
        for s in reversed(log):
            g = s["r"] + GAMMA * g
            to_go.append(g)
        to_go = to_go[::-1]
        for k, s in enumerate(log):
            q = s.get("q")
            if q is None:
                continue
            found = True
            kind = "first" if k == 0 else ("descent" if s["descent"] else s["request_kind"])
            gaps[kind].append(q["taken"] - to_go[k])
            gaps["all_reviews"].append(q["taken"] - to_go[k])
            for name, value in q.items():
                if name != "taken":
                    neighbour[name].append(value - q["taken"])
    if not found:
        return None
    return {"q_minus_g_by_kind": {k: _quantiles(v) for k, v in gaps.items()},
            "neighbour_minus_taken": {k: _quantiles(v) for k, v in neighbour.items()}}


def training(run: Path) -> dict:
    out: dict = {}
    by = defaultdict(Counter)
    cf_kind = defaultdict(Counter)
    cf_diff = defaultdict(lambda: defaultdict(list))
    for d in _jsonl(run / "decisions.jsonl"):
        w = int(d["t"] // 1000) * 1000
        by[w]["steps"] += 1
        rv = d.get("rv") or {}
        by[w]["reviews"] += int(rv.get("due", 0))
        by[w]["plan_events"] += int(rv.get("event", 0))
        if d.get("trig"):
            by[w]["replan_" + d["trig"]] += 1
        cf = d.get("cf")
        if cf and cf["kind"] != "continue":
            k = cf_kind[cf["kind"]]
            k["candidate"] += int(cf.get("alt_status") is not None)
            k["physically_distinct"] += int(cf.get("alt_status") is not None and not any(
                r.startswith(("same_effective_tau", "same_entry_speed")) for r in cf["reasons"]))
            k["gate_valid"] += int(cf["w"])
            k["reviews"] += 1
            if cf["w"]:
                for key in ("du0", "du_max", "dr", "dz"):
                    if cf.get(key) is not None:
                        cf_diff[cf["kind"]][key].append(abs(cf[key]))
    out["behaviour_per_1k"] = {w: dict(c) for w, c in sorted(by.items())}
    ev = defaultdict(Counter)
    for e in _jsonl(run / "episodes.jsonl"):
        ev[int(e["num_timesteps"] // 1000) * 1000][e.get("event", "?")] += 1
    out["behaviour_outcomes_per_1k"] = {w: dict(c) for w, c in sorted(ev.items())}
    if cf_kind:
        out["cf_by_kind"] = {k: dict(c) for k, c in cf_kind.items()}
        out["cf_valid_differences_by_kind"] = {k: {key: _quantiles(v) for key, v in d.items()} for k, d in cf_diff.items()}
    for name in ("delta_stats", "fix_stats"):
        path = run / f"{name}.json"
        if not path.exists():
            marks = sorted(run.glob(f"checkpoints/*_{name}.json"), key=lambda p: int(p.stem.split("_")[1]))
            path = marks[-1] if marks else path
        if path.exists():
            s = json.loads(path.read_text())
            if name == "delta_stats" and s.get("update"):
                out["delta"] = {"updates_sampled": len(s["update"]), "valid_fraction": _quantiles(s["valid_fraction"]),
                                "g_delta_over_g_td": _quantiles(list(np.asarray(s["g_delta"]) / np.asarray(s["g_td"]))),
                                "abs_target": _quantiles(s["abs_target_mean"]), "abs_dq": _quantiles(s["abs_dq_mean"]),
                                "applied_share_of_td": _quantiles(s.get("applied_share", [])),
                                "capped_fraction": float(np.mean(s["capped"])) if s.get("capped") else None}
            if name == "fix_stats" and s.get("audit_decisions"):
                out["critic_audit"] = {k: _quantiles([v for v in s.get(k, []) if v == v]) for k in (
                    "q_minus_g_plan", "q_minus_g_no_plan", "q_minus_g_descent", "band_q_spread")}
    calibration = run / "delta_calibration.json"
    if not calibration.exists():
        marks = sorted(run.glob("checkpoints/*_delta_calibration.json"), key=lambda p: int(p.stem.split("_")[1]))
        calibration = marks[-1] if marks else calibration
    if calibration.exists():
        c = json.loads(calibration.read_text())
        out["delta_calibration"] = {"probes": len(c.get("ratios", [])), "median_ratio": c.get("median_ratio"),
                                    "lambda": c.get("lambda"), "frozen_at_decisions": c.get("frozen_at_decisions")}
    hb = run / "heartbeat.json"
    if hb.exists():
        h = json.loads(hb.read_text())
        out["wall_s_per_step"] = h["wall_s"] / max(h["decisions"], 1)
        out["heartbeat"] = h
    return out


def readout(runs: dict[str, Path], rules_root: Path | None, strong: str | None, pure: Path | None) -> dict:
    refs: dict[str, dict[int, dict]] = {}
    if rules_root is not None:
        for d in sorted(p for p in rules_root.iterdir() if p.is_dir()):
            refs[d.name] = rows_of(d)
    if pure is not None:
        refs["pure"] = rows_of(pure)
    strong_key = None if strong is None else strong.replace("rule:earliest_", "rule_")
    result: dict = {"runs": {k: str(v) for k, v in runs.items()}, "strong_reference": strong_key,
                    "references": {k: outcome_table(v) for k, v in refs.items()}, "arms": {}, "between_arms": {}}
    marks_rows: dict[str, dict[int, dict[int, dict]]] = {}
    for arm, run in runs.items():
        marks = sorted(int(p.name) for p in (run / "dev_eval").glob("*") if p.name.isdigit())
        rows = {n: rows_of(run / "dev_eval" / str(n)) for n in marks}
        marks_rows[arm] = rows
        per = {}
        for n in marks:
            per[n] = {"outcome": outcome_table(rows[n]), "vs_0k": paired(rows[n], rows.get(0, {})),
                      **{f"vs_{k}": paired(rows[n], v) for k, v in refs.items() if k in (strong_key, "pure")},
                      "decisions": decisions(rows[n]), "critic": critic(rows[n])}
        result["arms"][arm] = {"marks": per, "training": training(run)}
    for a, b in (("n0", "b2"), ("ndelta", "n0"), ("ndelta", "b2")):
        if a in marks_rows and b in marks_rows:
            result["between_arms"][f"{a}-{b}"] = {n: paired(marks_rows[a][n], marks_rows[b][n])
                                                  for n in sorted(set(marks_rows[a]) & set(marks_rows[b]))}
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="arm=dir,...")
    parser.add_argument("--rules", type=Path, default=None, help="directory holding one sub-directory per rule")
    parser.add_argument("--strong", default=None, help="the frozen strong fixed reference, e.g. rule:earliest_k1.0_v0.02")
    parser.add_argument("--pure", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {k: Path(v) for k, v in (x.split("=", 1) for x in args.runs.split(",") if x)}
    result = readout(runs, args.rules, args.strong, args.pure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=float))
    for arm, r in result["arms"].items():
        for n, m in r["marks"].items():
            t = m["outcome"]
            if t.get("episodes"):
                print(arm, n, t["episodes"], "capture", t["capture"], "timeout", t["timeout"], "unsafe", t["unsafe"],
                      "return", round(t["mean_discounted_return"], 2), "dv", round(t["mean_delta_v_m_s"], 3))
    return result


if __name__ == "__main__":
    main()
