"""Read-only analysis for the pre-registered ECC formal experiment."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from math import ceil, floor, pi, radians
from pathlib import Path
import re

import numpy as np

REFERENCE = "SAC_C"
ECC = "ECC_CERT_LOOP"
PRESCREEN_POLICIES = ("rally", "c_10_0", "c_10_-5", "c_6_5", "b_10")
PRESCREEN_BUDGETS = (2.0, 1.8, 1.6)
CONDITIONS = ("urgent", "economy", "natural", "resource")
GROUP = re.compile(r"(.+)_s(\d+)_(urgent|economy|natural|resource)(?:_(cert|nocert|rigid))?")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def opening_rows(paths):
    result = {}
    for path in paths:
        row = read_json(path)
        opening = row["opening"]
        if opening in result or row["event"] in ("error", "numerical_failure"):
            raise ValueError(f"Duplicate or invalid episode in {path}")
        result[opening] = row
    return result


def success(row):
    return row["event"] == "success"


def distribution(values):
    values = list(values)
    if not values:
        return {"n": 0}
    q = np.asarray(values, dtype=float)
    return {"n": len(q), "mean": float(q.mean()), "median": float(np.median(q)),
            "p05": float(np.quantile(q, .05)), "p95": float(np.quantile(q, .95)),
            "min": float(q.min()), "max": float(q.max())}


def capture_summary(rows):
    n = len(rows)
    k = sum(map(success, rows))
    if not n:
        return {"n": 0, "captures": 0, "capture_rate": None, "capture_interval": None}
    p, z = k / n, 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / (1 + z * z / n)
    return {"n": n, "captures": k, "capture_rate": p,
            "capture_interval": [max(0., center - radius), min(1., center + radius)]}


def paired_bootstrap(strata, *, median=False):
    """Resample paired effects within each seed, retaining the original stratum sizes."""
    arrays = [np.asarray(rows, dtype=float) for rows in strata if len(rows)]
    if not arrays:
        return {"n": 0, "mean": None, "median": None, "interval": None}
    values = np.concatenate(arrays)
    rng = np.random.default_rng(20261110)
    samples = np.concatenate([v[rng.integers(len(v), size=(10_000, len(v)))] for v in arrays], axis=1)
    estimator = np.median(samples, axis=1) if median else samples.mean(axis=1)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "interval": np.quantile(estimator, [.025, .975]).tolist()}


def paired(candidate, baseline):
    a_keys, b_keys = set(candidate), set(baseline)
    if a_keys != b_keys:
        raise ValueError(f"Paired key mismatch: {len(a_keys - b_keys)} / {len(b_keys - a_keys)}")
    by_seed = defaultdict(list)
    rescue, lost = [], []
    for seed, opening in sorted(candidate):
        a, b = candidate[seed, opening], baseline[seed, opening]
        delta = int(success(a)) - int(success(b))
        if delta == 1:
            rescue.append([seed, opening])
        elif delta == -1:
            lost.append([seed, opening])
        row = {"capture": delta, "return": a["return"] - b["return"]}
        if success(a) and success(b):
            row.update(time=a["time_s"] - b["time_s"], delta_v=a["delta_v"] - b["delta_v"])
        by_seed[seed].append(row)

    def aggregate(groups):
        return {key: paired_bootstrap([[row[key] for row in rows if key in row] for rows in groups])
                for key in ("capture", "return", "time", "delta_v")}

    return {"per_seed": {str(seed): aggregate([rows]) for seed, rows in by_seed.items()},
            "combined": aggregate(list(by_seed.values())), "rescued": rescue, "lost": lost}


def primary(comparison):
    if comparison is None:
        return "not_established"
    interval = comparison["combined"]["capture"]["interval"]
    positive = sum(row["capture"]["mean"] > 0 for row in comparison["per_seed"].values())
    return "significant_increase" if interval and interval[0] > 0 and positive >= 2 else "not_established"


def summary(rows):
    rows = list(rows)
    terminal, margins = defaultdict(list), defaultdict(list)
    timing = []
    for row in rows:
        if success(row):
            for key, value in (row.get("terminal") or {}).items():
                if key != "in_cone" and value is not None:
                    terminal[key].append(value)
        for key, value in (row.get("min_margins") or {}).items():
            if value is not None:
                margins[key].append(value)
        timing.extend(d["cert"]["wall_s"] for d in row.get("decisions", ())
                      if "wall_s" in (d.get("cert") or {}))
    return {**capture_summary(rows), "return": distribution(r["return"] for r in rows),
            "events": dict(Counter(r["event"] for r in rows)),
            "mission_types": dict(Counter("urgent" if r["mission"]["urgent"] else "economy"
                                          for r in rows if "mission" in r)),
            "terminal_success": {k: distribution(v) for k, v in terminal.items()},
            "min_margins": {k: distribution(v) for k, v in margins.items()},
            "cert_wall_s": distribution(timing),
            "entry_losses": sum(r.get("entries_lost", 0) for r in rows),
            "entry_recoveries": sum(len(r.get("window_recoveries", ())) for r in rows)}


def n_geo(row):
    from env.t2_stage import (BRIDGE_ANGLE_DEG, BRIDGE_DEADLINE_MARGIN_S, opportunity_windows,
                              spin_frame, window_half_width)

    initial = row.get("initial")
    if not initial:
        return None
    q, v, rotation, omega, axis = [np.asarray(initial[k]) for k in ("q", "v", "rotation", "omega", "axis")]
    frame = spin_frame(rotation, omega, q, axis)
    deadline = row["mission"]["deadline_s"] - BRIDGE_DEADLINE_MARGIN_S
    timetable = opportunity_windows(q, v, rotation, omega, axis)
    arrival = timetable["t_arr"]
    if arrival > deadline:
        return 0
    width = window_half_width(radians(frame["lam_deg"]), radians(frame["beta_deg"]), radians(BRIDGE_ANGLE_DEG))
    if width is None:
        return 0
    rate = frame["rate_rad_s"]
    if rate <= 1e-9:
        return int(width >= abs((frame["phase_rad"] + pi) % (2 * pi) - pi))
    if width == pi:
        return 1
    half = width / rate
    first, period = frame["time_to_pass_s"], timetable["period"]
    return max(0, floor((deadline + half - first) / period) - ceil((arrival - half - first) / period) + 1)


def geometric_strata(candidate, baseline):
    if not candidate or set(candidate) != set(baseline):
        return {}
    opening_count = {}
    for (_, opening), row in candidate.items():
        if opening not in opening_count:
            opening_count[opening] = n_geo(row)
    groups = [(label, {o for o, count in opening_count.items() if count is not None and min(count, 2) == i})
              for i, label in enumerate(("0", "1", ">=2"))]
    groups = [(label, opens) for label, opens in groups if opens]
    i = 0
    while len(groups) > 1 and i < len(groups):
        if len(groups[i][1]) < 10:
            j = i + 1 if i + 1 < len(groups) else i - 1
            lo, hi = sorted((i, j))
            groups[lo:hi + 1] = [(groups[lo][0] + "+" + groups[hi][0], groups[lo][1] | groups[hi][1])]
            i = max(0, lo - 1)
        else:
            i += 1
    return {name: {"openings": sorted(opens), "effect": paired(
        {k: v for k, v in candidate.items() if k[1] in opens},
        {k: v for k, v in baseline.items() if k[1] in opens})} for name, opens in groups}


def mechanism(rows):
    strata = defaultdict(list)
    samples = []
    for (seed, opening), row in sorted(rows.items()):
        initial = row.get("initial")
        if not initial:
            continue
        period = 2 * pi / max(np.linalg.norm(initial["omega"]), 1e-9)
        for i, decision in enumerate(row.get("decisions", ())):
            cert = decision.get("cert")
            if cert is None or not cert.get("cands"):
                continue
            feasible = [c for c in cert["cands"] if c["event"] == 1
                        and c["dv"] + c["dv_entry"] <= row["budget_m_s"]]
            early = min(feasible, key=lambda c: c["t"]) if feasible else None
            eta = (decision["remaining_s"] - 45.0 - early["t"]) / period if early else None
            key = "uncovered" if eta is None else "<0" if eta < 0 else "[0,1)" if eta < 1 else ">=1"
            trace = {"seed": seed, "opening": opening, "decision": i, "eta": eta,
                     "event_kind": early["kind"] if early else None, "modified": bool(cert["intervened"]),
                     "qualified": sum(c["certified"] for c in cert["cands"]),
                     "remaining_dv": decision["remaining_dv"]}
            strata[key].append(trace)
            samples.append(trace)
    return {"strata": {key: {"n": len(items), "modification_rate": np.mean([t["modified"] for t in items]).item(),
                             "qualified_fraction": sum(t["qualified"] for t in items) / (9 * len(items)),
                             "event_types": dict(Counter(t["event_kind"] or "uncovered" for t in items))}
                       for key, items in strata.items()}, "decisions": samples}


def curves(root):
    result = {}
    for path in sorted((root / "B").glob("TRAIN_*/episodes.json")):
        rows = read_json(path)
        if not rows:
            continue
        manifest = read_json(path.with_name("manifest.json"))
        name = path.parent.name.removeprefix("TRAIN_")
        x = np.asarray([r["steps"] for r in rows])
        y = np.asarray([r["return_true"] for r in rows])
        late = x > manifest["steps"] - 1000
        last = x > manifest["steps"] - 500
        earlier = late & ~last

        def urgent_capture(mask):
            urgent = [success(r) for r, keep in zip(rows, mask) if keep and r["mission"]["urgent"]]
            return float(np.mean(urgent)) if urgent else None

        p, q = urgent_capture(earlier), urgent_capture(last)
        result[name] = {"steps": x.tolist(), "true_return_smoothed": [
            float(y[max(0, i - 49):i + 1].mean()) for i in range(len(y))],
            "true_return_slope_per_1000_steps": float(1000 * np.polyfit(x[late], y[late], 1)[0])
            if len(set(x[late])) >= 2 else None,
            "urgent_capture_change_last_two_500": q - p if p is not None and q is not None else None,
            "validation": {}}
    for folder in sorted((root / "V").glob("*_step*")):
        name, _, step = folder.name.rpartition("_step")
        result.setdefault(name, {"validation": {}})["validation"][step] = capture_summary(
            list(opening_rows(sorted(folder.glob("*.json"))).values()))
    return result


def readout(root):
    stages = defaultdict(lambda: defaultdict(dict))
    for folder in sorted((root / "C").iterdir()) if (root / "C").is_dir() else ():
        match = GROUP.fullmatch(folder.name)
        if not match:
            continue
        method, seed, condition, switch = match.groups()
        name = f"{method}_{switch}" if switch else method
        stages[condition][name].update({(int(seed), o): row for o, row in
                                        opening_rows(sorted(folder.glob("*.json"))).items()})
    fixed = defaultdict(dict)
    for folder in sorted((root / "S").iterdir()) if (root / "S").is_dir() else ():
        policy, _, condition = folder.name.rpartition("_")
        if condition in CONDITIONS:
            fixed[condition][policy] = opening_rows(sorted(folder.glob("*.json")))
    conditions = {}
    for condition in CONDITIONS:
        groups = stages[condition]

        def compare(a, b):
            return paired(groups[a], groups[b]) if groups.get(a) and groups.get(b) else None

        fixed_comp = {}
        for policy, rows in fixed[condition].items():
            matches = defaultdict(list)
            for (_, o), row in groups.get(ECC, {}).items():
                if o in rows:
                    matches[o].append(row)
            if matches and set(matches) == set(rows):
                fixed_comp[policy] = {
                    "unique_openings": len(matches),
                    "capture": paired_bootstrap([[np.mean([success(r) for r in v]) - success(rows[o])
                                                  for o, v in sorted(matches.items())]]),
                    "return": paired_bootstrap([[np.mean([r["return"] for r in v]) - rows[o]["return"]
                                                 for o, v in sorted(matches.items())]]),
                }
        conditions[condition] = {
            "learned": {method: {"pooled": summary(rows.values()), "per_seed": {
                str(seed): summary(v for (s, _), v in rows.items() if s == seed) for seed in sorted({s for s, _ in rows})}}
                        for method, rows in groups.items()},
            "fixed": {name: summary(rows.values()) for name, rows in fixed[condition].items()},
            "ecc_vs_fixed": fixed_comp,
            "ecc_vs_sac": compare(ECC, REFERENCE), "ecc_vs_td3": compare(ECC, "TD3_C"),
            "attribution_2x2": {m: compare(m, REFERENCE) for m in (ECC, REFERENCE, "EACH_C", "SAC_CERT_LOOP")},
            "switches": {"ecc_cert": compare(ECC, ECC + "_nocert"), "each_cert": compare("EACH_C_cert", "EACH_C"),
                         "sac_cert": compare("SAC_C_cert", REFERENCE),
                         "rotation_guidance": compare(ECC, "ECC_CERT_LOOP_NOGEO"),
                         "replanning": compare(ECC, ECC + "_rigid")},
            "n_geo": geometric_strata(groups.get(ECC, {}), groups.get(REFERENCE, {})),
            "mechanism": mechanism(groups.get(ECC, {})),
        }
    return {"primary_verdict": primary(conditions["urgent"]["ecc_vs_sac"]),
            "conditions": conditions, "curves": curves(root)}


def prescreen(rows, budget):
    openings = set(range(295100, 295148))
    if any(set(rows.get(p, {})) != openings for p in (*PRESCREEN_POLICIES, "gstar")):
        raise ValueError("Each of the six fixed policies must have all 48 prescreen episodes")
    five = [row for name in PRESCREEN_POLICIES for row in rows[name].values()]
    if any(row["mission"]["budget_m_s"] != budget or row["mission"]["urgent"]
           for group in rows.values() for row in group.values()):
        raise ValueError("Prescreen mission must be economic with the requested budget")
    activated = sum(r["event"] == "resource_exhausted" or r["delta_v"] >= .9 * budget for r in five)
    captures = sum(any(success(rows[p][o]) for p in PRESCREEN_POLICIES) for o in openings)
    qualifies = activated / len(five) >= .10 and captures / len(openings) >= .80
    return {"decision": "freeze_budget" if qualifies else "cancel" if budget == PRESCREEN_BUDGETS[-1] else "next_budget",
            "budget_m_s": budget, "activated": activated, "runs": len(five), "best_captures": captures,
            "openings": len(openings), "activation_rate": activated / len(five),
            "best_capture_rate": captures / len(openings), "gstar_reference": summary(rows["gstar"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("formal", "prescreen"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name == "prescreen":
            p.add_argument("--budget", type=float, choices=PRESCREEN_BUDGETS, required=True)
    args = parser.parse_args()
    if args.command == "formal":
        result = readout(args.root)
    else:
        result = prescreen({p: opening_rows(sorted(args.root.glob(f"{p}_*.json")))
                            for p in (*PRESCREEN_POLICIES, "gstar")}, args.budget)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result.get("primary_verdict", result.get("decision")))


if __name__ == "__main__":
    main()
