"""T1-O V2 section 2: diagnosis of the existing P1 / P2 screen data (no new episode).

Reads ``evidence/T1O_SCREEN/<p>_per_opening.csv``, the episode JSON of ``<p>_episodes.tar.gz``
and the frozen screening openings. Per profile, on the 16-opening subset where F1, F3 and F7
were all run (paired): outcomes, time, delta-v and the discounted return at fuel weights
5 / 10 / 15 / 20 (exact re-score from per-step delta-v); the strongest uniform choice and the
per-opening best of the three; time / delta-v crossings (one candidate faster, the other
cheaper, both safe captures); deadline clipping, re-plans by trigger, recoveries and holds;
and the per-opening return differences against the observable initial geometry.

    python -B -m experiments.t1o_diagnose --profile p2 --episodes <dir with F1/F3/F7> --output diagnose_p2.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

WEIGHTS = (5, 10, 15, 20)
POLICIES = ("F1", "F3", "F7")
FEATURES = ("range_m", "latitude_deg", "phase_deg", "rho_perp_m", "port_los_angle_deg", "port_latitude_deg",
            "omega_deg_s", "fov_margin_deg")


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)


def spearman(x, y) -> float:
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def diagnose(profile: str, episodes: Path, csv_path: Path, openings_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    by: dict[str, dict[int, dict]] = {}
    for r in rows:
        by.setdefault(r["policy"], {})[int(r["opening"])] = r
    sub = sorted(set(by["F3"]) & set(by["F7"]) & set(by["F1"]))
    feats = {o["seed"]: o["features"] for o in json.loads(openings_path.read_text())["openings"]}
    f = lambda p, s, k: float(by[p][s][k])  # noqa: E731
    out: dict = {"profile": profile, "paired_openings": sub, "per_weight": {}}
    for lam in WEIGHTS:
        key = f"J_{lam}"
        means = {p: float(np.mean([f(p, s, key) for s in sub])) for p in POLICIES}
        best = max(means, key=means.get)
        per_best = {s: max(POLICIES, key=lambda p: f(p, s, key)) for s in sub}
        oracle = float(np.mean([f(per_best[s], s, key) for s in sub]))
        diffs = {p: [f(p, s, key) - f("F1", s, key) for s in sub] for p in ("F3", "F7")}
        out["per_weight"][lam] = {
            "mean_return": means, "strongest_uniform": best, "oracle3_minus_strongest": oracle - means[best],
            "per_opening_best_counts": dict(Counter(per_best.values())),
            **{f"{p}_minus_F1": {"mean": float(np.mean(d)), "better": int(sum(x > 0 for x in d)),
                                 "better_by_1": int(sum(x >= 1.0 for x in d)), "worse_by_1": int(sum(x <= -1.0 for x in d))}
               for p, d in diffs.items()},
        }
    outcome = {p: dict(Counter(by[p][s]["event"] for s in sub)) for p in POLICIES}
    out["outcomes"] = outcome
    out["mean_time_s"] = {p: float(np.mean([f(p, s, "time_s") for s in sub])) for p in POLICIES}
    out["mean_delta_v"] = {p: float(np.mean([f(p, s, "delta_v_m_s") for s in sub])) for p in POLICIES}
    cross = []
    for s in sub:
        ok = [p for p in POLICIES if by[p][s]["event"] == "success"]
        for i, a in enumerate(ok):
            for b in ok[i + 1:]:
                dt, ddv = f(b, s, "time_s") - f(a, s, "time_s"), f(b, s, "delta_v_m_s") - f(a, s, "delta_v_m_s")
                if (dt > 0) != (ddv > 0) and abs(dt) >= 2.0 and abs(ddv) >= 0.02:
                    cross.append({"opening": s, "pair": f"{a}-{b}", "dt_s": dt, "ddv_m_s": ddv})
    out["time_delta_v_crossings"] = cross
    out["openings_with_crossing"] = sorted({c["opening"] for c in cross})
    events: dict = {}
    for p in POLICIES:
        deadline, trig, kinds = Counter(), Counter(), Counter()
        steps = 0
        for s in sub:
            ep = json.loads((episodes / p / f"{s}.json").read_text())
            for st in ep["decision_log"]:
                steps += 1
                if st["deadline"] != "none":
                    deadline[st["deadline"]] += 1
                if st["trigger"]:
                    trig[st["trigger"]] += 1
                kinds[st["status"].split(":")[0]] += 1
        events[p] = {"steps": steps, "deadline": dict(deadline), "new_plans_by_trigger": dict(trig),
                     "status": dict(kinds)}
    out["plan_events"] = events
    strata = {}
    for p in ("F3", "F7"):
        d = [f(p, s, "J_15") - f("F1", s, "J_15") for s in sub]
        dtime = [f(p, s, "time_s") - f("F1", s, "time_s") for s in sub]
        ddv = [f(p, s, "delta_v_m_s") - f("F1", s, "delta_v_m_s") for s in sub]
        corr = {k: spearman([feats[s][k] for s in sub], d) for k in FEATURES}
        corr["abs_sin_latitude"] = spearman([abs(np.sin(np.deg2rad(feats[s]["latitude_deg"]))) for s in sub], d)
        corr["sin_phase"] = spearman([np.sin(np.deg2rad(feats[s]["phase_deg"])) for s in sub], d)
        corr["cos_phase"] = spearman([np.cos(np.deg2rad(feats[s]["phase_deg"])) for s in sub], d)
        corr["tau_min_anchor_s"] = spearman([f("F1", s, "first_tau_min_feasible_s") for s in sub], d)
        better = [{"opening": s, "dJ15": d[i], "dt_s": dtime[i], "ddv_m_s": ddv[i],
                   **{k: feats[s][k] for k in ("range_m", "latitude_deg", "phase_deg", "omega_deg_s")},
                   "tau_min_anchor_s": f("F1", s, "first_tau_min_feasible_s")}
                  for i, s in enumerate(sub) if d[i] > 0]
        strata[p] = {"spearman_dJ15_vs": corr, "better_than_F1_at_15": better,
                     "dJ15_by_phase_bin": _bins(sub, d, lambda s: int(feats[s]["phase_deg"] // 90)),
                     "dJ15_by_latitude_bin": _bins(sub, d, lambda s: min(int(abs(np.sin(np.deg2rad(feats[s]["latitude_deg"]))) * 4), 3)),
                     "delta_v_saved_when_better": [b["ddv_m_s"] for b in better]}
    out["strata"] = strata
    return out


def _bins(sub, values, key):
    groups: dict[int, list[float]] = {}
    for s, v in zip(sub, values):
        groups.setdefault(key(s), []).append(v)
    return {int(k): {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(groups.items())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=("p1", "p2"), required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = diagnose(args.profile, args.episodes, Path(f"evidence/T1O_SCREEN/{args.profile}_per_opening.csv"),
                      Path(f"experiments/t1o_screen_{args.profile}_openings.json"))
    args.output.write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
