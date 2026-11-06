"""U2 offline precheck (stage A of ``docs/final_method/U2_PROBE_SPEC.md``): no training.

    python -B -m experiments.u2_probe_audit run --openings 294000-294047,13000000-13000047 --workers 4 --output DIR
    python -B -m experiments.u2_probe_audit summary --input DIR --output SUMMARY.json

``run``: fixed-intent and random-intent episodes in the probing environment (``reduced_probe``). At every decision state
it records the probe vector and per-probe details, the same prediction for the executed intent (audit only), the real
time of the next mode switch (entry accepted or descent), the probe's and the step's wall time, and on every fifth
decision a re-probe with the same and with fresh planner instances (A1). At every real 2 s entry check it re-computes
the probe's own entry rule on the real state and compares it with ``_try_bridge`` (A2), and at every descent check the
probe's gate with the executor's. ``summary``: the pre-registered checks A1-A5 and the report-only readings.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

POLICIES = {"c_10_0": (1.0, 0.0), "c_10_-5": (1.0, -0.5), "c_6_5": (0.6, 0.5), "rand": None}
URGENT = (140.0, 200.0)
HORIZON_S = 48.0


def block(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def episode(job) -> dict:
    opening, policy, out = job
    file = Path(out) / f"{policy}_{opening}.json"
    if file.exists():
        return json.loads(file.read_text(encoding="utf-8"))
    import torch; torch.set_num_threads(1)
    from env.t1_p_stage import relative_inertial
    from env.t2_probe import ExecutorProbe, executor_gate
    from env.t2_stage import OBS_DIM, T2StageEnv, observe
    from experiments.t2 import mission_profile

    env = T2StageEnv(openings=[opening], shaping=False, obs_profile="reduced_probe", mission=mission_profile(URGENT))
    checks = {"entry": 0, "entry_mismatch": [], "gate": 0, "gate_mismatch": [], "descent": 0, "descent_mismatch": []}
    timeline = []                       # (time after each 2 s truth step, mode)
    audit = {}

    def state():
        e = env.env; t = e.target_state; q, v = relative_inertial(e)
        return q, v, t.rotation.copy(), t.omega.copy()

    orig_try = env._try_bridge

    def try_bridge(m):
        q, v, rot, om = state()
        g = executor_gate(q, v, rot, env.env.config.task, audit["p"].descent_radius_m)
        from env.t2_stage import BRIDGE_ENVELOPE_MARGIN_M_S, BRIDGE_RANGE_M
        real_gate = not (m.range_m > BRIDGE_RANGE_M or m.radial_envelope_margin_m_s < BRIDGE_ENVELOPE_MARGIN_M_S)
        mine_gate = not (g["range_m"] > BRIDGE_RANGE_M or g["radial_margin_m_s"] < BRIDGE_ENVELOPE_MARGIN_M_S)
        checks["gate"] += 1
        if real_gate != mine_gate:
            checks["gate_mismatch"].append({"t": env.env.time_s, "real": real_gate, "probe": mine_gate, "range": m.range_m,
                                            "margin_real": m.radial_envelope_margin_m_s, "margin_probe": g["radial_margin_m_s"]})
        t_left = float(env.mission["deadline_s"]) - float(env.env.time_s)
        accepted = orig_try(m)
        if real_gate:
            hit = audit["p"].entry_check(q, v, rot, om, t_left)
            mine = hit is not None and hit["kind"] == "entry"
            checks["entry"] += 1
            if mine != accepted:
                checks["entry_mismatch"].append({"t": env.env.time_s, "real": accepted, "probe": mine, "range": g["range_m"],
                                                 "kind": None if hit is None else hit["kind"]})
        return accepted

    env._try_bridge = try_bridge
    orig_adv = env._advance

    def advance():
        r, event, info = orig_adv()
        timeline.append((float(env.env.time_s), env.generator.mode if env.generator is not None else None, event))
        return r, event, info

    env._advance = advance
    obs, _ = env.reset()
    audit["p"] = ExecutorProbe(env); audit["x"] = ExecutorProbe(env)
    g0 = env.generator
    orig_desc = g0.descending

    def descending(e, metrics=None):
        real = orig_desc(e, metrics)
        q, v, rot, _ = state()
        mine = executor_gate(q, v, rot, env.env.config.task, audit["p"].descent_radius_m)["descending"]
        checks["descent"] += 1
        if real != mine:
            checks["descent_mismatch"].append({"t": env.env.time_s, "real": real, "probe": mine})
        return real

    g0.descending = descending
    rng = np.random.default_rng([int(opening), 7])
    decisions, k, done, event = [], 0, False, None
    t_env = t_probe = 0.0
    while not done:
        e = env.env; q, v, rot, om = state()
        a = np.asarray(POLICIES[policy] if POLICIES[policy] is not None else rng.uniform(-1.0, 1.0, 2), dtype=np.float64)
        full = observe(e, q, v, e.time_s, e._delta_v, env.mission, rot, om, env.last)
        rec = {"k": k, "t": float(e.time_s), "range": float(np.linalg.norm(q)), "mode": env.generator.mode, "a": a.tolist(),
               "probe": np.asarray(obs[OBS_DIM:], float).round(6).tolist(), "obs": np.asarray(obs[:OBS_DIM], float).round(6).tolist(),
               "full": np.asarray(full, float).round(6).tolist(),
               "rows": [{x: d.get(x) for x in ("event", "kind", "t", "dv", "end", "solves", "holds")} for d in (env.last_probe or [])],
               "probe_s": float(env.probe_wall_s)}
        x = audit["x"].probe_intent(a, q, v, rot, om, float(e.time_s), float(e._delta_v), env.mission, env.geometry)
        rec["exec_probe"] = {y: x.get(y) for y in ("event", "kind", "t", "dv", "end")}
        if k % 5 == 0:
            same, _ = env.prober.probe(env)
            fresh, _ = ExecutorProbe(env).probe(env)
            ref = np.asarray(obs[OBS_DIM:], np.float32)
            rec["det"] = {"same": bool(np.array_equal(same, ref)), "fresh": bool(np.array_equal(fresh, ref)),
                          "max_diff": float(max(np.abs(same - ref).max(), np.abs(fresh - ref).max()))}
        decisions.append(rec)
        started = time.perf_counter()
        obs, _, term, trunc, info = env.step(a)
        t_step = time.perf_counter() - started
        t_env += t_step - (env.probe_wall_s if not (term or trunc) else env.probe_wall_s)
        t_probe += env.probe_wall_s
        rec["step_s"] = float(t_step - env.probe_wall_s)
        done, event, k = bool(term or trunc), info["event"], k + 1
    for rec in decisions:                       # the real next mode switch after each decision
        nxt = [t for t, mode, _ in timeline if t > rec["t"] + 1e-6 and mode in ("bridge", "descent")]
        rec["real_switch_s"] = (nxt[0] - rec["t"]) if nxt else None
    res = {"opening": opening, "policy": policy, "event": event, "time_s": float(env.env.time_s), "mission": env.mission,
           "decisions": decisions, "checks": checks, "t_env_s": t_env, "t_probe_s": t_probe,
           "end_t": float(env.env.time_s)}
    file.write_text(json.dumps(res), encoding="utf-8")
    return res


def cmd_run(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    jobs = [(o, p, str(out)) for o in block(a.openings) for p in POLICIES]
    if a.limit:
        jobs = jobs[: int(a.limit)]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(int(a.workers), maxtasksperchild=4) as pool:
        for i, r in enumerate(pool.imap_unordered(episode, jobs)):
            print(i + 1, len(jobs), r["policy"], r["opening"], r["event"], round(r["t_probe_s"] / max(r["t_env_s"], 1e-9), 2), flush=True)


# ---------------------------------------------------------------------- summary
def _logistic(X, y, Xt, l2=1.0, iters=300):
    import torch
    torch.manual_seed(0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    A = torch.tensor((X - mu) / sd, dtype=torch.float64); B = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(A.shape[1], dtype=torch.float64, requires_grad=True); b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, B) + l2 * (w @ w) / len(B)
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(torch.tensor((Xt - mu) / sd) @ w + b).numpy()


def _auc(y, p):
    order = np.argsort(p); y = np.asarray(y)[order]
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return None
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summary(root: Path) -> dict:
    eps = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(root.glob("*.json"))]
    eps = [e for e in eps if "decisions" in e]
    D = [(e, d) for e in eps for d in e["decisions"]]
    out = {"episodes": len(eps), "decisions": len(D), "policies": sorted({e["policy"] for e in eps})}
    # A1 determinism
    det = [d["det"] for _, d in D if "det" in d]
    out["A1"] = {"rechecked": len(det), "same_identical": sum(x["same"] for x in det), "fresh_identical": sum(x["fresh"] for x in det),
                 "max_diff": max([x["max_diff"] for x in det], default=0.0)}
    out["A1"]["pass"] = out["A1"]["same_identical"] == len(det) and out["A1"]["fresh_identical"] == len(det)
    # A2 rule consistency
    ch = {k: sum(e["checks"][k] for e in eps) for k in ("entry", "gate", "descent")}
    mm = {k: [x for e in eps for x in e["checks"][k + "_mismatch"]] for k in ("entry", "gate", "descent")}
    out["A2"] = {"checks": ch, "mismatches": {k: len(v) for k, v in mm.items()}, "examples": {k: v[:10] for k, v in mm.items()},
                 "entry_agreement": 1.0 - len(mm["entry"]) / max(ch["entry"], 1)}
    out["A2"]["pass"] = not mm["gate"] and not mm["descent"] and out["A2"]["entry_agreement"] >= 0.995
    # A3 non-collapse
    def informative(d):
        pairs = {(r["event"], round(r["t"], 3)) for r in d["rows"]}
        return len(pairs) > 1
    live = [(e, d) for e, d in D if d["rows"]]
    bins = {"gt12": lambda r: r > 12, "10_12": lambda r: 10 < r <= 12, "8_10": lambda r: 8 < r <= 10, "lt8": lambda r: r <= 8}
    out["A3"] = {"states": len(live), "informative_share": float(np.mean([informative(d) for _, d in live])) if live else 0.0,
                 "all_no_event_share": float(np.mean([all(r["event"] == 0 for r in d["rows"]) for _, d in live])) if live else 0.0,
                 "all_event_share": float(np.mean([all(r["event"] == 1 for r in d["rows"]) for _, d in live])) if live else 0.0,
                 "by_range": {k: {"n": sum(f(d["range"]) for _, d in live),
                                  "informative_share": float(np.mean([informative(d) for _, d in live if f(d["range"])] or [np.nan]))}
                              for k, f in bins.items()}}
    out["A3"]["pass"] = out["A3"]["informative_share"] >= 0.10
    # A4 cost
    te, tp = sum(e["t_env_s"] for e in eps), sum(e["t_probe_s"] for e in eps)
    ps = [d["probe_s"] for _, d in live]
    out["A4"] = {"env_s": te, "probe_s": tp, "ratio": tp / max(te, 1e-9), "probe_median_s": float(np.median(ps)) if ps else None,
                 "probe_p95_s": float(np.percentile(ps, 95)) if ps else None,
                 "solves_per_decision_median": float(np.median([sum(r["solves"] for r in d["rows"]) for _, d in live])) if live else None}
    out["A4"]["pass"] = out["A4"]["ratio"] <= 1.5
    # A5 executed-intent prediction vs real execution (fixed-intent episodes, 48 s)
    def agree(rows, horizon):
        y = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
        if len(y) == 0:
            return {"n": 0}
        tpr = float(np.mean(p[y == 1] == 1)) if (y == 1).any() else None
        tnr = float(np.mean(p[y == 0] == 0)) if (y == 0).any() else None
        bal = None if tpr is None or tnr is None else (tpr + tnr) / 2
        return {"n": int(len(y)), "real_positive": int(y.sum()), "predicted_positive": int(p.sum()), "tpr": tpr, "tnr": tnr,
                "balanced_accuracy": bal, "accuracy": float(np.mean(y == p))}
    def pairs(sel, horizon):
        rows, terr = [], []
        for e, d in sel:
            if d["t"] + horizon > e["end_t"] + 1e-6 and d["real_switch_s"] is None and e["event"] not in ("timeout", "resource_exhausted"):
                continue                       # censored: the episode ended inside the horizon without a switch for another reason
            real = d["real_switch_s"] is not None and d["real_switch_s"] <= horizon + 1e-6
            pred = d["exec_probe"]["event"] == 1 and d["exec_probe"]["t"] <= horizon + 1e-6
            rows.append((int(real), int(pred), d["range"]))
            if real and pred:
                terr.append(d["exec_probe"]["t"] - d["real_switch_s"])
        return rows, terr
    fixed = [(e, d) for e, d in live if e["policy"] != "rand"]
    rows48, terr48 = pairs(fixed, HORIZON_S)
    out["A5"] = {**agree(rows48, HORIZON_S), "time_error_s": {"n": len(terr48), "median": float(np.median(terr48)) if terr48 else None,
                                                              "mean_abs": float(np.mean(np.abs(terr48))) if terr48 else None},
                 "by_range": {k: agree([r for r in rows48 if f(r[2])], HORIZON_S) for k, f in bins.items()}}
    out["A5"]["pass"] = (out["A5"].get("balanced_accuracy") or 0.0) >= 0.6
    rows16, terr16 = pairs(live, 16.0)
    out["report_16s_all_policies"] = {**agree(rows16, 16.0), "time_error_median_s": float(np.median(terr16)) if terr16 else None}
    # report: small-sample prediction of the real switch within the next stage (16 s)
    X, y, grp = [], [], []
    for e, d in live:
        if d["t"] + 16.0 > e["end_t"] + 1e-6 and d["real_switch_s"] is None:
            continue
        X.append((d["obs"], [d["full"][i] for i in (8, 17, 19)], d["probe"], d["a"]))
        y.append(int(d["real_switch_s"] is not None and d["real_switch_s"] <= 16.0 + 1e-6)); grp.append(e["opening"])
    y, grp = np.array(y), np.array(grp)
    feats = {"obs": lambda x: x[0] + x[3], "obs+U1": lambda x: x[0] + x[1] + x[3], "obs+U2": lambda x: x[0] + x[2] + x[3],
             "obs+U1+U2": lambda x: x[0] + x[1] + x[2] + x[3]}
    ops = np.array(sorted(set(grp))); rs = np.random.default_rng(0); rs.shuffle(ops)
    folds = np.array_split(ops, 5)
    rep = {"n": int(len(y)), "positive": int(y.sum())}
    for name, f in feats.items():
        Xa = np.array([f(x) for x in X], float)
        res = {}
        for n_train in (250, 500, 1000, None):
            ll, au = [], []
            for fo in folds:
                te = np.isin(grp, fo); tr_idx = np.where(~te)[0]
                if n_train is not None:
                    tr_idx = np.random.default_rng(1).choice(tr_idx, size=min(n_train, len(tr_idx)), replace=False)
                p = np.clip(_logistic(Xa[tr_idx], y[tr_idx], Xa[te]), 1e-6, 1 - 1e-6)
                ll.append(float(-np.mean(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))); au.append(_auc(y[te], p))
            res[str(n_train or "all")] = {"logloss": float(np.mean(ll)), "auc": float(np.mean([v for v in au if v is not None]))}
        rep[name] = res
    out["report_event_prediction"] = rep
    out["stop"] = not (out["A3"]["pass"] and out["A4"]["pass"] and out["A5"]["pass"])
    out["defects"] = not (out["A1"]["pass"] and out["A2"]["pass"])
    return out


def cmd_summary(a):
    res = summary(Path(a.input))
    Path(a.output).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k].get("pass") if isinstance(res[k], dict) and "pass" in res[k] else res[k]
                      for k in ("episodes", "decisions", "A1", "A2", "A3", "A4", "A5", "stop", "defects")}, indent=1))


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest="cmd", required=True)
    r = s.add_parser("run"); r.add_argument("--openings", required=True); r.add_argument("--workers", type=int, default=4)
    r.add_argument("--output", required=True); r.add_argument("--limit", type=int)
    m = s.add_parser("summary"); m.add_argument("--input", required=True); m.add_argument("--output", required=True)
    a = p.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
