"""T1-O V2 pre-registration (research plan "T1-O V2", 2026-10-17): Gates A-E on profile P2.

Frozen in this file, with the opening sets, before the first new complete episode. P2 =
14-18 m, 2.1-2.7 deg/s (``env.t1_o_task``); everything physical, G* v0.6 and the shared MPC
unchanged; 6 s reviews with plan-event reviews; first-plan delay after the feasible anchor
(0-90 s range, the effective range is audited), +-30 s in-flight shifts, keep band, the 240 s
planning deadline. Reward: time -0.05 / s, +100 / -150 / -100, gamma 0.999 per 2 s step, fuel
weight lambda in {5, 10, 15, 20} chosen on the discovery set (exact per-step re-score).

Opening sets (``experiments.t1o_openings``, geometry only):
  discovery16  ``experiments/t1o_p2_discovery_openings.json``     298300-298398
  confirm32    ``experiments/t1o_p2_confirmation_openings.json``  298400-298504
  dev64        ``experiments/t1o_p2_development_openings.json``   298600-298690
  dev16 = the first opening of every cell of dev64; dev32 = the first two of every cell.

Gate A (discovery16 x the 10 candidates ``GATE_A``; every candidate differs only in its first
review; afterwards a live plan is kept and a review without one asks for the earliest feasible
arrival). Physical layer first, the return second. Per fuel weight:
  A1 (weight independent) at least 14 / 16 openings have a safe capture by some candidate;
     every opening where several candidates fail by the same mechanism is listed;
  A2 at least 4 openings with two safe captures forming a time / delta-v non-dominated pair
     (|dt| >= 2 s and |d delta-v| >= 0.02 m/s, one faster and costlier);
  A3 with U = the strongest uniform candidate (highest mean return over the 16):
     (i) >= 4 openings where another candidate beats U by >= 1.0; (ii) >= 4 openings where U
     beats another *safely captured* candidate by >= 1.0; (iii) on the openings where U
     captured safely, mean(per-opening best safe capture - U) >= 2.5;
  A4 kappa (K0-A0, K30-A30) or the entry speed (V0-A0, V30-A30, Vmax-V30): >= 4 openings with
     both captured safely, |dJ| >= 1.0 and |dt| >= 2 s or |d delta-v| >= 0.02 m/s (fresh
     episodes are bitwise repeatable; the repeat check ``repeatability`` is reported); the
     share of openings preferring each end is reported (one end everywhere = no state value);
  no arbitrage: no delayed candidate with more timeouts than A0 and a higher mean return.
The smallest weight passing A2-A4 without arbitrage is chosen; A1 failing or no weight passing
is LEARNING_SPACE_NO_GO (no confirmation set, no training, no new profile). Requests clipped by
the 240 s deadline are counted and not taken as effective samples.

Gate B (after A, on the discovery trajectories of U at the chosen weight):
  in flight: per opening the first review with t >= 20 s, a live feasible plan (kept or new),
  not descending, remaining >= 30 s and deadline slack (horizon - remaining) >= 15 s; at most
  16 states. From one snapshot with fresh solvers, branches keep and +15 s (-15 s when +15 s
  would pass the deadline), each continued to the end with U's protocol; a third fresh keep
  branch is not needed (fresh branches are bitwise equal); the keep branch vs. the original
  continuation is the numerical repeatability. Real effect: |dJ| >= 1.0 and > 10 x the
  repeatability. PASS: >= 8 states with two safe distinct outcomes and |dJ| >= 1.0, and the
  preferred branch not the same in all of them.
  near field: per opening the first review in G*'s descent mode, at most 16; branches kappa
  1.0 and 0.7 to the end; reported with the same effect rule (descriptive).
  G*: requested -> accepted / clipped / slipped / kept / recovered, and the share of legal
  requests mapped onto the same plan, from every Gate A / C episode.

G* predicted-cost rule (non-learning baseline): at the first review, for every Gate A
candidate with U's kappa, the earliest anchor for its entry speed plus its delay is planned
by the same G* planner (read only); predicted cost = 0.05 x adopted arrival + lambda x planned
delta-v; the cheapest is executed (ties: the smaller delay). Planning CPU is recorded.

Simple state rule (Gate C): features = the initial SAC observation components range_10m,
sin_latitude, cos_latitude, sin_phase, cos_phase, rho_perp_10m, port_los_180deg,
port_latitude_90deg, omega_2p7deg_s, fov_margin_50deg, radial_closing_0p3m_s,
tangential_speed_0p5m_s (no feasible-anchor time: it is not observed). Rules: constant, one
split, or a root split with each child split again; at most two features; thresholds at
midpoints of the discovery values; every leaf >= 3 openings and outputs one Gate A candidate
(the one with the highest summed return in the leaf). Objective: summed discovery return at
the chosen weight; ties: fewer leaves, then feature order, then threshold order.

Gate C (confirm32, opened once): U, the state rule, the G* rule and A0 when U is not A0.
PASS: state rule - U paired mean >= 1.5 with 95 % t-interval lower bound > 0, no more unsafe
than U. The G* rule is compared paired: if it matches or beats the state rule in return,
safety and on-line planning, report "existing planning solves the task" (no four-arm run);
otherwise the pilot may proceed. Zero step: the untrained O-R3 model (seed 0, 256 x 3) and the
state rule on dev32; required: state rule - 0k >= 3.0 mean return or >= 3 more captures.

Gate D (pilot, after A-C and E's eligibility tests): O-R3, seed 0, trainer A, 12k real steps,
dev16 at 0 / 3k / 6k / 9k / 12k. PASS: paired mean return at 9k and at 12k above 0k; replay
audit Q - G (live plan) mean over 6k-12k <= +50; unsafe + timeout at 9k and at 12k each <= the
0k count + 2. Review-action spread, adoption, later share and MC agreement are reported.

Gate E (after lambda is frozen): MPC counterfactual credibility. Calibration snapshots from
the Gate A trajectories, validation snapshots from the Gate C trajectories (independent);
far-field kinds and the near-field kappa alternative; e_r tolerance = p95 of the executed
branch's e_r on calibration (other gates unchanged); on validation: accepted pairs with a true
2 s reward difference >= 0.10 and repeatability < 1/3 of it: >= 20 pairs and sign accuracy
>= 80 %; far-field credible distinct >= 10 % of far-field review candidates; >= 20 credible
near-field kappa samples.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments.t1o_screen import CANDIDATES

GATE_A = ("A0", "A15", "A30", "A45", "A60", "K0", "K30", "V0", "V30", "Vmax")
WEIGHTS = (5.0, 10.0, 15.0, 20.0)
DELAYED = ("A15", "A30", "A45", "A60")
DIMENSION_PAIRS = {"kappa": (("A0", "K0"), ("A30", "K30")),
                   "entry_speed": (("A0", "V0"), ("A30", "V30"), ("V30", "Vmax"))}
STATE_FEATURES = ("range_10m", "sin_latitude", "cos_latitude", "sin_phase", "cos_phase", "rho_perp_10m",
                  "port_los_180deg", "port_latitude_90deg", "omega_2p7deg_s", "fov_margin_50deg",
                  "radial_closing_0p3m_s", "tangential_speed_0p5m_s")
MIN_LEAF = 3


def J(row: dict, lam: float) -> float:
    return float(row["discounted_return_by_fuel_weight"][str(lam)])


def safe(row: dict) -> bool:
    return row["event"] == "success"


def physically_distinct(a: dict, b: dict) -> bool:
    return abs(a["time_s"] - b["time_s"]) >= 2.0 or abs(a["delta_v_m_s"] - b["delta_v_m_s"]) >= 0.02


def load(root: Path, seeds: list[int], names=GATE_A) -> dict[str, dict[int, dict]]:
    return {c: {s: json.loads((root / c / f"{s}.json").read_text()) for s in seeds} for c in names}


# --------------------------------------------------------------------------- Gate A
def gate_a(rows: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    out: dict = {"A1": {}, "per_weight": {}}
    safe_any = [s for s in seeds if any(safe(rows[c][s]) for c in GATE_A)]
    multi_fail = {s: {c: [rows[c][s]["event"], rows[c][s]["end_phase"], rows[c][s]["hold_steps"]] for c in GATE_A}
                  for s in seeds if sum(not safe(rows[c][s]) for c in GATE_A) >= 2}
    clipped = {c: sum(any(st["deadline"] != "none" for st in rows[c][s]["decision_log"]) for s in seeds) for c in GATE_A}
    out["A1"] = {"openings_with_a_safe_capture": len(safe_any), "pass": len(safe_any) >= 14,
                 "openings_with_several_failures": multi_fail, "episodes_with_deadline_events": clipped}
    pareto = []
    for s in seeds:
        ok = [c for c in GATE_A if safe(rows[c][s])]
        pairs = [(a, b) for a, b in combinations(ok, 2)
                 if abs(rows[a][s]["time_s"] - rows[b][s]["time_s"]) >= 2.0
                 and abs(rows[a][s]["delta_v_m_s"] - rows[b][s]["delta_v_m_s"]) >= 0.02
                 and (rows[a][s]["time_s"] - rows[b][s]["time_s"]) * (rows[a][s]["delta_v_m_s"] - rows[b][s]["delta_v_m_s"]) < 0]
        if pairs:
            pareto.append({"opening": s, "pairs": len(pairs)})
    for lam in WEIGHTS:
        means = {c: float(np.mean([J(rows[c][s], lam) for s in seeds])) for c in GATE_A}
        u = max(means, key=means.get)
        beats_u = [s for s in seeds if any(safe(rows[c][s]) and J(rows[c][s], lam) >= J(rows[u][s], lam) + 1.0
                                           for c in GATE_A if c != u)]
        u_beats = [s for s in seeds if safe(rows[u][s]) and any(
            safe(rows[c][s]) and J(rows[u][s], lam) >= J(rows[c][s], lam) + 1.0 for c in GATE_A if c != u)]
        valid = [s for s in seeds if safe(rows[u][s])]
        gap = float(np.mean([max(J(rows[c][s], lam) for c in GATE_A if safe(rows[c][s])) - J(rows[u][s], lam)
                             for s in valid])) if valid else float("nan")
        best_counts: dict[str, int] = {}
        for s in valid:
            b = max((c for c in GATE_A if safe(rows[c][s])), key=lambda c: J(rows[c][s], lam))
            best_counts[b] = best_counts.get(b, 0) + 1
        dims = {}
        for dim, pairs in DIMENSION_PAIRS.items():
            hits, prefer = [], {"first": 0, "second": 0}
            for s in seeds:
                hit = False
                for a, b in pairs:
                    ra, rb = rows[a][s], rows[b][s]
                    if safe(ra) and safe(rb) and physically_distinct(ra, rb) and abs(J(ra, lam) - J(rb, lam)) >= 1.0:
                        hit = True
                        prefer["first" if J(ra, lam) > J(rb, lam) else "second"] += 1
                if hit:
                    hits.append(s)
            dims[dim] = {"openings": len(hits), "preferences": prefer}
        timeouts = {c: sum(rows[c][s]["event"] == "timeout" for s in seeds) for c in GATE_A}
        arbitrage = [c for c in DELAYED if timeouts[c] > timeouts["A0"] and means[c] > means["A0"]]
        checks = {"A2_pareto": len(pareto) >= 4, "A3_i": len(beats_u) >= 4, "A3_ii": len(u_beats) >= 4,
                  "A3_iii_gap": bool(valid) and gap >= 2.5,
                  "A4_dimension": max(d["openings"] for d in dims.values()) >= 4, "no_arbitrage": not arbitrage}
        out["per_weight"][str(lam)] = {
            "mean_return": means, "uniform_best": u, "openings_beating_U_by_1": beats_u, "openings_U_beats_by_1": u_beats,
            "U_safe_openings": len(valid), "best_minus_U_mean": gap, "per_opening_best": best_counts,
            "dimensions": dims, "timeouts": timeouts, "arbitrage": arbitrage, "checks": checks, "passes": all(checks.values())}
    out["pareto_openings"] = pareto
    passing = [lam for lam in WEIGHTS if out["per_weight"][str(lam)]["passes"]]
    out["fuel_weight"] = passing[0] if (passing and out["A1"]["pass"]) else None
    out["verdict"] = "PASS" if out["fuel_weight"] is not None else "LEARNING_SPACE_NO_GO"
    return out


# --------------------------------------------------------------------------- simple state rule
def _leaf(rows, seeds, lam):
    totals = {c: sum(J(rows[c][s], lam) for s in seeds) for c in GATE_A}
    best = max(GATE_A, key=lambda c: (totals[c], -GATE_A.index(c)))
    return best, totals[best]


def _thresholds(values):
    v = sorted(set(values))
    return [0.5 * (a + b) for a, b in zip(v[:-1], v[1:])]


def fit_state_rule(rows: dict[str, dict[int, dict]], seeds: list[int], lam: float) -> dict:
    from env.t1_task_review import REVIEW_OBSERVATION_NAMES

    idx = {f: REVIEW_OBSERVATION_NAMES.index(f) for f in STATE_FEATURES}
    x = {s: rows["A0"][s]["initial_observation"] for s in seeds}
    best = {"form": "constant", "candidate": _leaf(rows, seeds, lam)[0], "objective": _leaf(rows, seeds, lam)[1],
            "leaves": 1}

    def better(obj, leaves, key):
        return (obj, -leaves) > (best["objective"], -best["leaves"]) and key is not None

    for i, f1 in enumerate(STATE_FEATURES):
        for t1 in _thresholds([x[s][idx[f1]] for s in seeds]):
            lo = [s for s in seeds if x[s][idx[f1]] <= t1]
            hi = [s for s in seeds if x[s][idx[f1]] > t1]
            if len(lo) < MIN_LEAF or len(hi) < MIN_LEAF:
                continue
            (cl, ol), (ch, oh) = _leaf(rows, lo, lam), _leaf(rows, hi, lam)
            if better(ol + oh, 2, f1):
                best = {"form": "split", "feature": f1, "threshold": t1, "low": cl, "high": ch,
                        "objective": ol + oh, "leaves": 2}
            for f2 in STATE_FEATURES:
                if f2 == f1:
                    continue
                children = {}
                for side, group in (("low", lo), ("high", hi)):
                    opts = [(None, _leaf(rows, group, lam))]
                    for t2 in _thresholds([x[s][idx[f2]] for s in group]):
                        a = [s for s in group if x[s][idx[f2]] <= t2]
                        b = [s for s in group if x[s][idx[f2]] > t2]
                        if len(a) >= MIN_LEAF and len(b) >= MIN_LEAF:
                            (ca, oa), (cb, ob) = _leaf(rows, a, lam), _leaf(rows, b, lam)
                            opts.append((t2, (ca, cb, oa + ob)))
                    children[side] = max(opts, key=lambda o: (o[1][-1], o[0] is None))
                leaves = sum(1 if t is None else 2 for t, _ in children.values())
                obj = sum(v[-1] for _, v in children.values())
                if leaves > 2 and better(obj, leaves, f2):
                    best = {"form": "two_level", "feature": f1, "threshold": t1, "feature2": f2,
                            "children": {side: ({"leaf": v[0]} if t is None else
                                                {"threshold": t, "low": v[0], "high": v[1]})
                                         for side, (t, v) in children.items()},
                            "objective": obj, "leaves": leaves}
    return best


def apply_state_rule(rule: dict, observation) -> str:
    from env.t1_task_review import REVIEW_OBSERVATION_NAMES

    value = lambda f: float(observation[REVIEW_OBSERVATION_NAMES.index(f)])  # noqa: E731
    if rule["form"] == "constant":
        return rule["candidate"]
    side = "low" if value(rule["feature"]) <= rule["threshold"] else "high"
    if rule["form"] == "split":
        return rule[side]
    child = rule["children"][side]
    if "leaf" in child:
        return child["leaf"]
    return child["low"] if value(rule["feature2"]) <= child["threshold"] else child["high"]


# --------------------------------------------------------------------------- G* predicted-cost rule
def gstar_rule_choice(profile: str, opening: int, kappa: float, lam: float) -> dict:
    """Read-only planning at the first review; returns the chosen candidate and the costs."""

    from time import process_time

    from env.t1_gstar import free_body_rotations
    from env.t1_o_task import t1o_review_env

    env = t1o_review_env(profile, lam)
    env.reset(options={"opening": opening})
    g, e = env.generator, env.env
    q0 = e.chaser_state.position - e.target_state.position
    v0 = e.chaser_state.rotation @ e.chaser_state.velocity - e.target_state.rotation @ e.target_state.velocity
    started = process_time()
    costs = {}
    for name in GATE_A:
        delay, k, v_in = CANDIDATES[name]
        if abs(k - kappa) > 1e-9:
            continue
        tau_min, _ = g.earliest_feasible(e, v_in)
        if tau_min is None:
            continue
        tau = tau_min + delay
        history = free_body_rotations(e.target_state.rotation, e.target_state.omega, e.target_parameters.inertia, tau + 1.0)
        plan = g.planner.solve(q0, v0, tau, history, None, entry_speed_m_s=v_in)
        if plan["feasible"]:
            costs[name] = {"tau_s": tau, "planned_delta_v": plan["delta_v_m_s"],
                           "cost": e.config.reward.time_cost_per_s * tau + lam * plan["delta_v_m_s"]}
    choice = min(costs, key=lambda c: (costs[c]["cost"], CANDIDATES[c][0])) if costs else "A0"
    return {"opening": opening, "choice": choice, "costs": costs, "planning_cpu_s": process_time() - started}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("gate-a")
    a.add_argument("--openings", type=Path, required=True)
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("state-rule")
    r.add_argument("--openings", type=Path, required=True)
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--fuel-weight", type=float, required=True)
    r.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("gstar-rule")
    g.add_argument("--openings", type=Path, required=True)
    g.add_argument("--kappa", type=float, required=True)
    g.add_argument("--fuel-weight", type=float, required=True)
    g.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    doc = json.loads(args.openings.read_text())
    if args.command == "gate-a":
        result = gate_a(load(args.root, doc["seeds"]), doc["seeds"])
    elif args.command == "state-rule":
        result = fit_state_rule(load(args.root, doc["seeds"]), doc["seeds"], args.fuel_weight)
    else:
        result = {"rows": [gstar_rule_choice(doc["profile"], s, args.kappa, args.fuel_weight) for s in doc["seeds"]]}
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: v for k, v in result.items() if k in ("verdict", "fuel_weight", "form", "candidate", "feature")}))


if __name__ == "__main__":
    main()
