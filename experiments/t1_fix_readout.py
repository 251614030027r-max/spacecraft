"""Four audit curves of the first learnability-fix training pair (task book 2026-10-15, sec. "四条审计曲线").

Inputs: the two training run directories (M0, M2; each with ``dev_eval/<decisions>/`` written by
``experiments.t1_dev_eval --watch`` and the training logs), the fixed-rule evaluation
directories (kappa 0.7 and 1.0, same G* version) and the frozen Pure MPC evidence.

1. deterministic development return, capture / timeout / unsafe, delta-v, time per checkpoint,
   against 0k, the fixed rules and Pure MPC (on the openings each mark evaluated; a mark on fewer
   openings is compared with the others on that same subset);
2. effective action: first-plan requested vs effective tau, keep / later / earlier request rates
   with a live plan, new plans by trigger (deviation > 1 m, deadline, request, no plan), kappa
   in and out of descent and its relation to delta-v; the same from the training decisions per
   1k window (behaviour policy, Trap 11);
3. critic: on the deterministic development episodes min_j Q(s, a_taken) - G_to_go by decision
   kind (first / keep / later / earlier / new plan / descent / no plan), Q(later 15 s) - Q(keep)
   and Q(earlier 15 s) - Q(keep) with a live plan; and the in-training audits (Q - G by state
   group on the replay MC labels, keep-band Q spread) per 1k window;
4. M2: gate open rate, closing reasons, per counterfactual kind (tau kinds, ``entry_speed``)
   the candidate / physically distinct / gate-valid counts, CF gradient share, real-vs-alternative
   tau difference, CPU per decision of both runs.

Curve 2 also carries the entry speed (interface v4): requested, applied by new plans, requested
without effect (kept plan).

Descriptive only: no number here is a gate unless the run order says so.

    python -B -m experiments.t1_fix_readout --m0 logs/t1/t1fix_m0_s0 --m2 logs/t1/t1fix_m2_s0 \
        --rules k07=eval/t1fix/rule_k07,k10=eval/t1fix/rule_k10 --pure evidence/T1_dev_pure_295000 \
        --output eval/t1fix/first_train_readout.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

GAMMA = 0.999


def _jsonl(path: Path):
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if not gz.exists():
            return
        path = gz
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _rows_dir(directory: Path) -> dict[int, dict]:
    rows = {}
    for f in directory.glob("*.json"):
        if f.stem.isdigit():
            rows[int(f.stem)] = json.loads(f.read_text())
    if not rows:
        for r in _jsonl(directory / "episodes.jsonl"):
            rows[int(r.get("opening", r.get("seed")))] = r
    return rows


def outcome_table(rows: dict[int, dict], subset: list[int] | None = None) -> dict:
    keys = sorted(rows) if subset is None else [o for o in subset if o in rows]
    rs = [rows[o] for o in keys]
    if not rs:
        return {"episodes": 0}
    events = Counter(r["event"] for r in rs)
    return {
        "episodes": len(rs), "capture": events.get("success", 0), "timeout": events.get("timeout", 0),
        "unsafe": events.get("unsafe", 0), "numerical_failure": events.get("numerical_failure", 0),
        "mean_discounted_return": float(np.mean([r["discounted_return"] for r in rs])),
        "mean_delta_v_m_s": float(np.mean([r["delta_v_m_s"] for r in rs])),
        "mean_time_s_fail_as_300": float(np.mean([r["time_s"] if r["event"] == "success" else 300.0 for r in rs])),
    }


def _kind(step: dict) -> str:
    if step.get("descent"):
        return "descent"
    return step.get("request_kind") or "unknown"


def action_stats(rows: dict[int, dict]) -> dict:
    first_req, first_eff, kinds, triggers = [], [], Counter(), Counter()
    v_requested, v_applied, v_requested_kept = [], [], []
    kappa_in, kappa_out, per_episode = [], [], []
    for r in rows.values():
        log = r.get("decision_log", [])
        new = next((s for s in log if str(s["status"]).startswith(("optimal", "recovery"))), None)
        if new is not None:
            first_req.append(new["tau_request_s"])
            first_eff.append(new["tau_effective_s"])
        for s in log:
            kinds[_kind(s)] += 1
            if s.get("trigger"):
                triggers[s["trigger"]] += 1
            (kappa_in if s.get("descent") else kappa_out).append(s["kappa"])
            if s.get("entry_speed_request_m_s") is not None:
                v_requested.append(s["entry_speed_request_m_s"])
                (v_applied if s.get("entry_speed_applied") else v_requested_kept).append(s["entry_speed_request_m_s"])
        per_episode.append((r.get("kappa_mean", np.nan), r["delta_v_m_s"]))
    plan_decisions = sum(kinds[k] for k in ("keep", "later", "earlier"))
    k = np.asarray(per_episode, dtype=float)
    return {
        "first_plan_request_s_mean": float(np.mean(first_req)) if first_req else None,
        "first_plan_effective_s_mean": float(np.mean(first_eff)) if first_eff else None,
        "decisions_by_kind": dict(kinds),
        "with_plan_rates": {name: kinds[name] / plan_decisions for name in ("keep", "later", "earlier")}
        if plan_decisions else None,
        "new_plans_by_trigger": dict(triggers),
        "deviation_replans_per_episode": triggers.get("deviation", 0) / max(len(rows), 1),
        "kappa_out_of_descent": _quantiles(kappa_out), "kappa_in_descent": _quantiles(kappa_in),
        "entry_speed_requested": _quantiles(v_requested),
        "entry_speed_applied_new_plans": _quantiles(v_applied),
        "entry_speed_requested_without_effect": _quantiles(v_requested_kept),
        "entry_speed_applied_share_of_decisions": len(v_applied) / max(len(v_requested), 1),
        "corr_episode_kappa_delta_v": float(np.corrcoef(k[:, 0], k[:, 1])[0, 1])
        if len(k) > 2 and np.std(k[:, 0]) > 0 else None,
    }


def _quantiles(values) -> dict | None:
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
            "p50": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def critic_stats(rows: dict[int, dict]) -> dict | None:
    gaps, later, earlier = defaultdict(list), [], []
    for r in rows.values():
        log = r.get("decision_log", [])
        if not log or log[0].get("q") is None:
            return None
        rewards = [s["r"] for s in log]
        g, to_go = 0.0, []
        for x in reversed(rewards):
            g = x + GAMMA * g
            to_go.append(g)
        to_go = to_go[::-1]
        for k, s in enumerate(log):
            q = s["q"]
            gap = q["taken"] - to_go[k]
            group = "first" if k == 0 else ("new_plan" if s.get("trigger") else _kind(s))
            gaps[group].append(gap)
            gaps["all"].append(gap)
            if "keep" in q:
                later.append(q["later15"] - q["keep"])
                earlier.append(q["earlier15"] - q["keep"])
    return {"q_minus_g_by_kind": {k: _quantiles(v) for k, v in gaps.items()},
            "q_later15_minus_keep": _quantiles(later), "q_earlier15_minus_keep": _quantiles(earlier)}


def _windows(values, at, width=1000):
    out = defaultdict(list)
    for v, t in zip(values, at):
        if v is not None and np.isfinite(v):
            out[int(t // width) * width].append(v)
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def training_stats(run: Path) -> dict:
    out: dict = {}
    decisions = list(_jsonl(run / "decisions.jsonl"))
    by = defaultdict(Counter)
    kappa = defaultdict(list)
    cf = defaultdict(Counter)
    tau_diff = defaultdict(list)
    cf_kind = defaultdict(lambda: defaultdict(Counter))
    v_applied = defaultdict(list)
    for d in decisions:
        if d.get("v_applied") and d.get("v_req") is not None:
            v_applied[int(d["t"] // 1000) * 1000].append(d["v_req"])
        w = int(d["t"] // 1000) * 1000
        by[w][d.get("req") or "unknown"] += 1
        if d.get("trig"):
            by[w]["replan_" + d["trig"]] += 1
        by[w]["decisions"] += 1
        kappa[w].append(d["kappa"])
        if "cf" in d:
            cf[w]["decisions"] += 1
            cf[w]["open"] += int(d["cf"]["w"])
            k = cf_kind[w][d["cf"]["kind"]]
            k["candidate"] += int(d["cf"].get("tau_eff_alt") is not None or d["cf"].get("v_plan_alt") is not None)
            k["physically_distinct"] += int(d["cf"].get("alt_status") is not None and not any(
                r.startswith(("same_effective_tau", "same_entry_speed")) for r in d["cf"]["reasons"]))
            k["gate_valid"] += int(d["cf"]["w"])
            k["decisions"] += 1
            for reason in d["cf"]["reasons"]:
                cf[w][reason.split(":")[0]] += 1
            if d["cf"]["w"] and d["cf"].get("tau_eff_alt") is not None:
                tau_diff[w].append(abs(float(d["cf"]["tau_eff_alt"]) - float(d["tau_eff"])))
    out["behaviour_per_1k"] = {w: {**c, "kappa_mean": float(np.mean(kappa[w])),
                                   "entry_speed_applied": _quantiles(v_applied.get(w, []))}
                               for w, c in sorted(by.items())}
    if cf_kind:
        out["cf_by_kind_per_1k"] = {w: {k: dict(c) for k, c in kinds.items()} for w, kinds in sorted(cf_kind.items())}
        total = defaultdict(Counter)
        for kinds in cf_kind.values():
            for k, c in kinds.items():
                total[k].update(c)
        out["cf_by_kind_total"] = {k: dict(c) for k, c in total.items()}
    if cf:
        out["cf_gate_per_1k"] = {w: {**c, "open_rate": c["open"] / c["decisions"],
                                     "mean_abs_tau_eff_alt_minus_real": float(np.mean(tau_diff[w])) if tau_diff[w] else None}
                                 for w, c in sorted(cf.items())}
    episodes = list(_jsonl(run / "episodes.jsonl"))
    ev = defaultdict(Counter)
    for e in episodes:
        ev[int(e["num_timesteps"] // 1000) * 1000][e.get("event", "?")] += 1
    out["behaviour_outcomes_per_1k"] = {w: dict(c) for w, c in sorted(ev.items())}
    fix = run / "fix_stats.json"
    if not fix.exists():
        marks = sorted(run.glob("checkpoints/*_fix_stats.json"), key=lambda p: int(p.stem.split("_")[1]))
        fix = marks[-1] if marks else fix
    if fix.exists():
        s = json.loads(fix.read_text())
        at = s.get("audit_decisions", [])
        out["critic_audit_per_1k"] = {name: _windows(s.get(name, []), at) for name in (
            "q_minus_g_plan", "q_minus_g_no_plan", "q_minus_g_descent", "band_q_spread")}
        if s.get("probe_decisions"):
            p = s["probe_decisions"]
            out["grad_probe_per_1k"] = {
                "mc_over_td": _windows(list(np.asarray(s["g_mc"]) / np.asarray(s["g_td"])), p),
                "eq_over_td": _windows(list(np.asarray(s["g_eq"]) / np.asarray(s["g_td"])), p)}
    cfs = run / "cf_stats.json"
    if not cfs.exists():
        marks = sorted(run.glob("checkpoints/*_cf_stats.json"), key=lambda p: int(p.stem.split("_")[1]))
        cfs = marks[-1] if marks else cfs
    if cfs.exists():
        s = json.loads(cfs.read_text())
        if s.get("grad_update"):
            ratio = np.asarray(s["grad_norm_aux"]) / np.asarray(s["grad_norm_real"])
            out["cf_grad_share"] = _quantiles(list(ratio))
        if s.get("valid_fraction"):
            out["cf_valid_fraction_in_batches"] = float(np.mean(s["valid_fraction"]))
    heartbeat = run / "heartbeat.json"
    if heartbeat.exists():
        h = json.loads(heartbeat.read_text())
        out["wall_s_per_decision"] = h["wall_s"] / max(h["decisions"], 1)
        out["heartbeat"] = h
    return out


def readout(m0: Path, m2: Path, rules: dict[str, Path], pure: Path | None) -> dict:
    result: dict = {"runs": {"m0": str(m0), "m2": str(m2)}, "curve1": {}, "curve2": {}, "curve3": {}, "curve4": {}}
    references = {name: _rows_dir(path) for name, path in rules.items()}
    if pure is not None:
        references["pure"] = _rows_dir(pure)
    for method, run in (("m0", m0), ("m2", m2)):
        marks = sorted((int(p.name) for p in (run / "dev_eval").glob("*") if p.name.isdigit()))
        all_rows = {n: _rows_dir(run / "dev_eval" / str(n)) for n in marks}
        zero = all_rows.get(0, {})
        result["curve1"][method] = {}
        result["curve2"][method] = {}
        result["curve3"][method] = {}
        for n in marks:
            rows = all_rows[n]
            subset = sorted(rows)
            result["curve1"][method][n] = {
                "this": outcome_table(rows), "zero_k_same_openings": outcome_table(zero, subset),
                **{name: outcome_table(ref, subset) for name, ref in references.items()},
            }
            result["curve2"][method][n] = action_stats(rows)
            result["curve3"][method][n] = critic_stats(rows)
        training = training_stats(run)
        result["curve2"][method]["training"] = {k: training.get(k) for k in ("behaviour_per_1k", "behaviour_outcomes_per_1k")}
        result["curve3"][method]["training"] = {k: training.get(k) for k in ("critic_audit_per_1k", "grad_probe_per_1k")}
        result["curve4"][method] = {k: training.get(k) for k in (
            "cf_gate_per_1k", "cf_by_kind_per_1k", "cf_by_kind_total", "cf_grad_share", "cf_valid_fraction_in_batches", "wall_s_per_decision", "heartbeat")}
    for name, ref in references.items():
        result["curve2"][f"reference_{name}"] = action_stats(ref) if name != "pure" else None
    return result


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--m2", type=Path, required=True)
    parser.add_argument("--rules", default="", help="name=dir,... (fixed-rule dev evaluations)")
    parser.add_argument("--pure", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rules = {k: Path(v) for k, v in (item.split("=", 1) for item in args.rules.split(",") if item)}
    result = readout(args.m0, args.m2, rules, args.pure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=float))
    for method in ("m0", "m2"):
        for n, row in result["curve1"][method].items():
            t = row["this"]
            if t.get("episodes"):
                print(method, n, t["episodes"], "capture", t["capture"], "timeout", t["timeout"], "unsafe", t["unsafe"],
                      "return", round(t["mean_discounted_return"], 2), "dv", round(t["mean_delta_v_m_s"], 3))
    return result


if __name__ == "__main__":
    main()
