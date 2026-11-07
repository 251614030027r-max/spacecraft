"""T1-O V2 Gates B and C and the zero-step check (rules: ``experiments.t1o_v2`` docstring).

* ``gate-b``: in-flight keep / +-15 s branches and near-field kappa 1.0 / 0.7 branches from
  snapshots of the uniform candidate's discovery trajectories; every branch starts from the
  same synced state with fresh solvers and is flown to the end with the uniform protocol
  (the kappa branches keep their kappa); the keep branch against the original continuation is
  the numerical repeatability.
* ``gate-c``: the per-opening candidate of each arm (uniform, state rule, G* rule, A0) on the
  confirmation set, the episodes (shared where two arms pick the same candidate) and the
  paired readout.
* ``zero-step``: an untrained model checkpoint and the state rule on dev32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.t1o_screen import BASE_FUEL, CANDIDATES, FUEL_WEIGHTS, FixedCandidate, run as run_candidates
from experiments.t1o_v2 import apply_state_rule, gstar_rule_choice

GAMMA = 0.999


def _scores(rewards: list[float], dvs: list[float]) -> dict[str, float]:
    return {str(l): float(sum(GAMMA ** i * (r - (l - BASE_FUEL) * dv) for i, (r, dv) in enumerate(zip(rewards, dvs))))
            for l in FUEL_WEIGHTS}


class _Continue(FixedCandidate):
    """The uniform protocol after the branching review: keep, earliest without a plan; kappa
    (and entry speed) as given."""

    def __init__(self, name: str, kappa: float | None = None) -> None:
        super().__init__(name)
        self.first_done = True
        if kappa is not None:
            self.kappa = float(kappa)


def _fly(env, first_action: np.ndarray, policy: _Continue, k0: int) -> dict:
    """From the current state: ``first_action`` now, then ``policy`` to the end; discounted from
    step ``k0`` of the episode (the branch's own return-to-go is reported, discounted to it)."""

    rewards, dvs, k = [], [], 0
    action = first_action
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        dvs.append(float(info["delta_v_step_m_s"]))
        k += 1
        if terminated or truncated:
            s = info["episode_summary"]
            return {"event": s["event"], "time_s": s["time_s"], "delta_v_m_s": s["delta_v_m_s"],
                    "violations": s["violations"], "return_to_go": _scores(rewards, dvs), "steps": k}
        action = policy.action(env)


def _snapshot_branches(profile: str, real, actions: list[np.ndarray], policies: list[_Continue]) -> list[dict]:
    from env.t1_o_task import t1o_env_config, t1o_review_env
    from env.t1_option_cf import ReviewCounterfactualPredictor

    out = []
    k0 = real.env.decision
    for action, policy in zip(actions, policies):
        branch = ReviewCounterfactualPredictor(None, t1o_env_config(profile), None,
                                               shadow_factory=lambda: t1o_review_env(profile))
        branch.sync(real, fresh_solvers=True)
        out.append(_fly(branch.shadow, action, policy, k0))
    return out


def gate_b_opening(args) -> dict:
    """Both Gate B snapshots of one discovery opening (in flight and near field)."""

    import torch

    torch.set_num_threads(1)
    from env.t1_o_task import t1o_review_env

    profile, opening, uniform = args
    result = {"opening": opening}
    for mode in ("in_flight", "near_field"):
        real = t1o_review_env(profile)
        real.reset(options={"opening": opening})
        policy = FixedCandidate(uniform)
        found = None
        while True:
            g, e = real.generator, real.env
            if real._due and policy.first_done:
                remaining = g.remaining_s(e)
                horizon = float(e.config.max_time_s) - g.deadline_margin_s - float(e.time_s)
                descending = g._descending(e, e.metrics())
                if (mode == "in_flight" and e.time_s >= 20.0 and g.plan is not None and not descending
                        and remaining >= 30.0 and horizon - remaining >= 15.0
                        and g.last_status.startswith(("kept", "optimal"))):
                    found = {"t": e.time_s, "remaining_s": remaining, "slack_s": horizon - remaining}
                    break
                if mode == "near_field" and descending:
                    found = {"t": e.time_s}
                    break
            action = policy.action(real)
            _, _, terminated, truncated, info = real.step(action)
            if info["review"]["reviewed"]:
                policy.executed(real, info)
            if terminated or truncated:
                break
        if found is None:
            result[mode] = None
            continue
        base = policy.action(real)  # U's own action at this review (keep with a live plan)
        if mode == "in_flight":
            alt = base.copy()
            alt[0] = real.action_for_shift(15.0)  # +15 s (the slack is >= 15 s by selection)
            keep, moved = _snapshot_branches(profile, real, [base, alt], [_Continue(uniform), _Continue(uniform)])
            original = _fly(real, base, _Continue(uniform), real.env.decision)
            result[mode] = {**found, "shift_s": 15.0, "keep": keep, "shift": moved, "original": original}
        else:
            c = real.config
            a_hi, a_lo = base.copy(), base.copy()
            a_hi[1] = 2.0 * (1.0 - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
            a_lo[1] = 2.0 * (0.7 - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
            hi, lo = _snapshot_branches(profile, real, [a_hi, a_lo], [_Continue(uniform, 1.0), _Continue(uniform, 0.7)])
            result[mode] = {**found, "kappa_1.0": hi, "kappa_0.7": lo}
    return result


def gate_b(profile: str, openings: dict, uniform: str, lam: float, workers: int = 4) -> dict:
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(workers) as pool:
        per = pool.map(gate_b_opening, [(profile, s, uniform) for s in openings["seeds"]], chunksize=1)
    rows = {mode: [{"opening": r["opening"], **r[mode]} for r in per if r[mode] is not None]
            for mode in ("in_flight", "near_field")}
    return {"uniform": uniform, "fuel_weight": lam, **rows, "summary": summarise_b(rows, lam)}


def summarise_b(rows: dict, lam: float) -> dict:
    key = str(lam)
    out = {}
    states = rows["in_flight"][:16]
    effects, prefer = [], {"keep": 0, "shift": 0}
    noise = [abs(r["keep"]["return_to_go"][key] - r["original"]["return_to_go"][key]) for r in states]
    for r in states:
        a, b = r["keep"], r["shift"]
        d = b["return_to_go"][key] - a["return_to_go"][key]
        n = abs(r["keep"]["return_to_go"][key] - r["original"]["return_to_go"][key])
        real = (a["event"] == "success" and b["event"] == "success" and abs(d) >= 1.0 and abs(d) > 10.0 * n
                and (abs(a["time_s"] - b["time_s"]) >= 2.0 or abs(a["delta_v_m_s"] - b["delta_v_m_s"]) >= 0.02))
        effects.append({"opening": r["opening"], "dJ_shift_minus_keep": d, "repeatability": n, "real_effect": real})
        if real:
            prefer["shift" if d > 0 else "keep"] += 1
    count = sum(e["real_effect"] for e in effects)
    out["in_flight"] = {"states": len(states), "real_effects": count, "preferences": prefer,
                        "repeatability_max": max(noise) if noise else None, "effects": effects,
                        "pass": count >= 8 and min(prefer.values()) > 0}
    near = rows["near_field"][:16]
    neff = []
    for r in near:
        a, b = r["kappa_1.0"], r["kappa_0.7"]
        d = b["return_to_go"][key] - a["return_to_go"][key]
        neff.append({"opening": r["opening"], "dJ_0.7_minus_1.0": d, "both_safe": a["event"] == b["event"] == "success",
                     "dt_s": b["time_s"] - a["time_s"], "ddv": b["delta_v_m_s"] - a["delta_v_m_s"]})
    out["near_field"] = {"states": len(near), "effects": neff,
                         "openings_abs_dJ_ge_1": sum(e["both_safe"] and abs(e["dJ_0.7_minus_1.0"]) >= 1.0 for e in neff)}
    return out


# --------------------------------------------------------------------------- Gate C
def gate_c_assignments(profile: str, openings: dict, uniform: str, rule: dict, kappa: float, lam: float) -> dict:
    from env.t1_o_task import t1o_review_env

    arms: dict[str, dict[int, str]] = {"uniform": {}, "state_rule": {}, "gstar_rule": {}}
    if uniform != "A0":
        arms["A0"] = {}
    gstar = {}
    for s in openings["seeds"]:
        env = t1o_review_env(profile)
        obs, _ = env.reset(options={"opening": s})
        arms["uniform"][s] = uniform
        arms["state_rule"][s] = apply_state_rule(rule, obs)
        gstar[s] = gstar_rule_choice(profile, s, kappa, lam)
        arms["gstar_rule"][s] = gstar[s]["choice"]
        if "A0" in arms:
            arms["A0"][s] = "A0"
    return {"arms": arms, "gstar": gstar}


def paired(a: list[float], b: list[float]) -> dict:
    from scipy import stats

    d = np.asarray(a) - np.asarray(b)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    h = float(stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)))
    return {"n": len(d), "mean": m, "sd": sd, "ci95": [m - h, m + h], "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


def gate_c_readout(assign: dict, root: Path, lam: float) -> dict:
    key = str(lam)
    rows = {}
    for arm, picks in assign["arms"].items():
        rows[arm] = {s: json.loads((root / c / f"{s}.json").read_text()) for s, c in picks.items()}
    seeds = sorted(assign["arms"]["uniform"])
    J = {arm: [r[s]["discounted_return_by_fuel_weight"][key] for s in seeds] for arm, r in rows.items()}
    out = {"fuel_weight": lam, "outcomes": {arm: {e: sum(r[s]["event"] == e for s in seeds) for e in ("success", "unsafe", "timeout")}
                                             for arm, r in rows.items()},
           "mean_return": {arm: float(np.mean(v)) for arm, v in J.items()},
           "state_rule_minus_uniform": paired(J["state_rule"], J["uniform"]),
           "gstar_rule_minus_uniform": paired(J["gstar_rule"], J["uniform"]),
           "state_rule_minus_gstar_rule": paired(J["state_rule"], J["gstar_rule"]),
           "gstar_planning_cpu_s_mean": float(np.mean([g["planning_cpu_s"] for g in assign["gstar"].values()]))}
    p = out["state_rule_minus_uniform"]
    out["pass"] = (p["mean"] >= 1.5 and p["ci95"][0] > 0.0
                   and out["outcomes"]["state_rule"]["unsafe"] <= out["outcomes"]["uniform"]["unsafe"])
    return out


def run_assignments(profile: str, assign: dict, root: Path, workers: int) -> None:
    by_candidate: dict[str, set[int]] = {}
    for picks in assign["arms"].values():
        for s, c in picks.items():
            by_candidate.setdefault(c, set()).add(s)
    for c, seeds in sorted(by_candidate.items()):
        run_candidates(profile, c, sorted(seeds), root / c, workers)


# --------------------------------------------------------------------------- zero step
def model_episode(profile: str, model, opening: int) -> dict:
    from env.t1_o_task import t1o_review_env

    env = t1o_review_env(profile)
    obs, _ = env.reset(options={"opening": opening})
    rewards, dvs = [], []
    while True:
        action = np.asarray(model.predict(obs, deterministic=True)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        dvs.append(float(info["delta_v_step_m_s"]))
        if terminated or truncated:
            s = info["episode_summary"]
            return {"opening": opening, "event": s["event"], "time_s": s["time_s"], "delta_v_m_s": s["delta_v_m_s"],
                    "discounted_return_by_fuel_weight": _scores(rewards, dvs)}


def dev_subset(dev64: dict, per_cell: int) -> list[int]:
    """dev16 (per_cell 1) / dev32 (per_cell 2): the first openings of every dev64 cell."""

    cells: dict[tuple[int, int], list[int]] = {}
    for o in dev64["openings"]:
        cells.setdefault((o["latitude_bin"], o["phase_bin"]), []).append(o["seed"])
    return sorted(s for seeds in cells.values() for s in sorted(seeds)[:per_cell])


def _model_job(args) -> None:
    import torch

    torch.set_num_threads(1)
    from train.t1_tcpf import TCPFSAC

    profile, checkpoint, opening, out = args
    model = TCPFSAC.load(checkpoint, device="cpu")
    row = model_episode(profile, model, opening)
    Path(out, f"{opening}.json").write_text(json.dumps(row, default=float))
    print(opening, row["event"], round(row["time_s"], 1), flush=True)


def eval_model(profile: str, checkpoint: Path, seeds: list[int], output_dir: Path, workers: int) -> None:
    import multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)
    todo = [(profile, str(checkpoint), s, str(output_dir)) for s in seeds if not (output_dir / f"{s}.json").exists()]
    if todo:
        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            pool.map(_model_job, todo, chunksize=1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("gate-b")
    b.add_argument("--profile", default="p2")
    b.add_argument("--openings", type=Path, required=True)
    b.add_argument("--uniform", required=True)
    b.add_argument("--fuel-weight", type=float, required=True)
    b.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("gate-c")
    c.add_argument("--profile", default="p2")
    c.add_argument("--openings", type=Path, required=True)
    c.add_argument("--uniform", required=True)
    c.add_argument("--rule", type=Path, required=True)
    c.add_argument("--fuel-weight", type=float, required=True)
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("eval-model")
    m.add_argument("--profile", default="p2")
    m.add_argument("--checkpoint", type=Path, required=True)
    m.add_argument("--dev64", type=Path, required=True)
    m.add_argument("--per-cell", type=int, choices=(1, 2), required=True, help="1: dev16, 2: dev32")
    m.add_argument("--output-dir", type=Path, required=True)
    m.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "eval-model":
        seeds = dev_subset(json.loads(args.dev64.read_text()), args.per_cell)
        eval_model(args.profile, args.checkpoint, seeds, args.output_dir, args.workers)
        return
    doc = json.loads(args.openings.read_text())
    if args.command == "gate-b":
        result = gate_b(args.profile, doc, args.uniform, args.fuel_weight)
    else:
        rule = json.loads(args.rule.read_text())
        kappa = CANDIDATES[args.uniform][1]
        assign = gate_c_assignments(args.profile, doc, args.uniform, rule, kappa, args.fuel_weight)
        (args.root).mkdir(parents=True, exist_ok=True)
        (args.root / "assignments.json").write_text(json.dumps(assign, indent=1, default=float))
        run_assignments(args.profile, assign, args.root, args.workers)
        result = gate_c_readout(assign, args.root, args.fuel_weight)
    args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result.get("summary", {k: result.get(k) for k in ("pass", "mean_return")}), default=float)[:2000])


if __name__ == "__main__":
    main()
