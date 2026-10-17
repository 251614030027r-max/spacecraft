"""Upper review numbers for the T1-L three-arm seed-0 40k delivery (collab 68680eb)."""
import glob, json, os, sys
import numpy as np
from scipy import stats

Z, OUT, CHOICE = sys.argv[1], sys.argv[2], sys.argv[3]
MARKS = (0, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
ARMS = ("b2", "n0", "ndelta")
RULE = "rule_k1.0_v0.02"


def rows(d):
    return {int(os.path.basename(f)[:-5]): json.load(open(f)) for f in glob.glob(d + "/*.json")
            if os.path.basename(f)[:-5].isdigit()}


def paired(a, b, key="discounted_return"):
    o = sorted(set(a) & set(b))
    x = np.array([a[k][key] - b[k][key] for k in o])
    m, s = float(x.mean()), float(x.std(ddof=1))
    h = float(stats.t.ppf(0.975, len(x) - 1) * s / np.sqrt(len(x)))
    return {"n": len(x), "mean": m, "sd": s, "ci95": [m - h, m + h], "better": int((x > 0).sum()), "worse": int((x < 0).sum())}


def behaviour(R):
    v = [x for r in R.values() for x in (r["entry_speed_applied"] if isinstance(r["entry_speed_applied"], list)
                                          else [r["entry_speed_applied"]]) if x is not None]
    return {"first_delay_s": float(np.mean([r["first_delay_s"] for r in R.values()])),
            "kappa_mean": float(np.mean([r["kappa_mean"] for r in R.values()])),
            "entry_speed_applied_mean": float(np.mean(v)) if v else None,
            "new_plans": float(np.mean([r["new_plans"] for r in R.values()])),
            "req_later": float(np.mean([r["req_later"] for r in R.values()])),
            "time_s": float(np.mean([r["time_s"] for r in R.values()])),
            "delta_v_m_s": float(np.mean([r["delta_v_m_s"] for r in R.values()]))}


ev = {a: {n: rows(f"{Z}/{a}_s0/dev_eval/{n}") for n in MARKS} for a in ARMS}
rules = {os.path.basename(d): rows(d) for d in sorted(glob.glob(f"{Z}/rules/rule_*"))}
out = {"source": "collab 68680eb, T1_L_THREE_ARM_S0_FINAL.zip sha256 eae15a98089316b6ddbfc21745e6e6cc568d4bb8f26e05e68305f4ec342bf5df",
       "science": "166244bd4d08713473e17354626f10e40b496e01", "strong_reference": RULE, "paired": {}, "behaviour": {},
       "failures": {}, "training": {}}
for n in MARKS:
    p = {}
    for a in ARMS:
        p[f"{a}-0k"] = paired(ev[a][n], ev[a][0])
        p[f"{a}-{RULE}"] = paired(ev[a][n], rules[RULE])
    p["n0-b2"] = paired(ev["n0"][n], ev["b2"][n])
    p["ndelta-n0"] = paired(ev["ndelta"][n], ev["n0"][n])
    if n == 40000:
        for a in ("n0", "ndelta"):
            p[f"{a}-{RULE}:time_s"] = paired(ev[a][n], rules[RULE], "time_s")
            p[f"{a}-{RULE}:delta_v"] = paired(ev[a][n], rules[RULE], "delta_v_m_s")
    out["paired"][n] = p
    out["behaviour"][n] = {a: behaviour(ev[a][n]) for a in ARMS}
    out["failures"][n] = {a: [[o, r["event"], r.get("violations"), r.get("end_phase")] for o, r in sorted(ev[a][n].items())
                              if r["event"] != "success"] for a in ARMS}
out["behaviour"]["rules"] = {k: behaviour(R) for k, R in rules.items()}
out["rules_return"] = {k: float(np.mean([r["discounted_return"] for r in R.values()])) for k, R in rules.items()}
for a in ARMS:
    fix = json.load(open(f"{Z}/{a}_s0/fix_stats.json"))
    at, q = np.asarray(fix["audit_decisions"]), np.asarray(fix["q_minus_g_plan"], dtype=float)
    later = {}
    with open(f"{Z}/{a}_s0/decisions.jsonl") as h:
        for line in h:
            d = json.loads(line)
            if (d.get("rv") or {}).get("due") and d.get("req") in ("keep", "later", "earlier"):
                b = int((d["t"] - 1) // 5000) * 5000
                c = later.setdefault(b, [0, 0])
                c[0] += d["req"] == "later"
                c[1] += 1
    out["training"][a] = {
        "replay_q_minus_g_plan_by_5k": {b: float(np.nanmean(q[(at > b) & (at <= b + 5000)])) for b in range(0, 40000, 5000)
                                        if ((at > b) & (at <= b + 5000)).any()},
        "later_share_at_reviews_with_plan_by_5k": {b: c[0] / c[1] for b, c in sorted(later.items())}}
o = sorted(rules[RULE])
best = {x: max(R[x]["discounted_return"] for R in rules.values()) for x in o}
gain = np.array([best[x] - rules[RULE][x]["discounted_return"] for x in o])
winners = [max(rules, key=lambda k: rules[k][x]["discounted_return"]) for x in o]
out["rule_oracle"] = {
    "definition": "per opening, the best of the 6 fixed rules, chosen after the fact on the same 16 openings: an upper "
                  "bound of state-dependent switching inside this rule family only",
    "oracle_minus_strong": {"mean": float(gain.mean()), "median": float(np.median(gain)), "max": float(gain.max()),
                            "openings_where_another_rule_wins": int((gain > 1e-9).sum())},
    "winner_counts": {k: winners.count(k) for k in sorted(set(winners))},
    **{f"{a}_40k_minus_oracle_mean": float(np.mean([ev[a][40000][x]["discounted_return"] - best[x] for x in o]))
       for a in ("n0", "ndelta")}}
out["trainer_choice_preview"] = json.load(open(CHOICE))
out["trainer_choice_preview"]["note"] = ("experiments.t1_l_s_trainer_choice (science 672ba79) on the delivered package, decisions.jsonl "
                                         "decompressed from decisions.jsonl.gz; the run order applies it on the lower's original run directories")
json.dump(out, open(OUT, "w"), indent=1, default=float)
