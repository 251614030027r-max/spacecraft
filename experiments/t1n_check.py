"""T1-N technical review (Research Plan R2 sec. 9.4): physics, interface and signal checks.

Development openings only (block 292000-293999, T1-N technical checks; never the formal block).
Nothing here is a performance gate: it locates errors and measures what R2 asks the upper to
report (t = 0 legality, the planner's response to neutral requests, closed-loop non-degeneracy,
the safety of a rejection, shadow-prediction information and cost).

    python -B -m experiments.t1n_check openings --first 292000 --count 2000 --output o.json
    python -B -m experiments.t1n_check feasibility --first 292000 --count 48 --output f.json --workers 4
    python -B -m experiments.t1n_check closed --first 292000 --count 16 --policies P,... --output-dir D --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

TAU_GRID_S = tuple(float(t) for t in range(20, 201, 5))


# --------------------------------------------------------------------------- #
def opening_record(opening: int) -> dict[str, Any]:
    from env.t1_env import T1Env
    from env.t1_n_task import t1n_env_config

    env = T1Env(t1n_env_config())
    env.reset(opening)
    m = env.metrics()
    f = env.features(m)
    t, c = env.target_state, env.chaser_state
    q = c.position - t.position
    v = c.rotation @ c.velocity - t.rotation @ t.velocity
    r = float(np.linalg.norm(q))
    u = q / r
    radial = float(v @ u)
    tangential = float(np.linalg.norm(v - radial * u))
    h = env.anchor.h_axis
    omega_inertial = t.rotation @ t.omega
    body_z = t.rotation[:, 2]
    port = t.rotation @ env.config.task.approach_axis
    return {
        "opening": opening, "violations": dict(m.violations), "range_m": r,
        "radial_speed_m_s": radial, "tangential_speed_m_s": tangential, "speed_m_s": float(np.linalg.norm(v)),
        "radial_envelope_margin_m_s": float(m.radial_envelope_margin_m_s),
        "fov_margin_deg": float(np.degrees(m.fov_margin_rad)),
        "tumble_deg_s": float(np.degrees(np.linalg.norm(t.omega))),
        "spin_offset_body_z_deg": float(np.degrees(np.arccos(np.clip(t.omega[2] / np.linalg.norm(t.omega), -1, 1)))),
        "omega_to_h_deg": float(np.degrees(np.arccos(np.clip(omega_inertial @ h / np.linalg.norm(omega_inertial), -1, 1)))),
        "body_z_to_h_deg": float(np.degrees(np.arccos(np.clip(body_z @ h, -1, 1)))),
        "port_latitude_deg": float(np.degrees(np.arcsin(np.clip(port @ h, -1, 1)))),
        "los_latitude_deg": f["latitude_deg"], "phase_deg": f["phase_deg"],
        # angular rate of the chaser about the target from the tangential speed vs the port sweep
        "tangential_rate_over_spin": float(tangential / r / np.linalg.norm(t.omega)),
    }


def port_latitude_band(opening: int, horizon_s: float = 240.0, step_s: float = 1.0) -> dict[str, float]:
    """Torque-free prediction of the port direction's latitude about H over the horizon."""

    from env.t1_env import T1Env
    from env.t1_gstar import free_body_rotations
    from env.t1_n_task import t1n_env_config

    env = T1Env(t1n_env_config())
    env.reset(opening)
    t = env.target_state
    times, rotations, _ = free_body_rotations(t.rotation, t.omega, env.target_parameters.inertia, horizon_s, step_s)
    h = env.anchor.h_axis
    lat = [float(np.degrees(np.arcsin(np.clip((r @ env.config.task.approach_axis) @ h, -1, 1)))) for r in rotations]
    return {"port_latitude_min_deg": min(lat), "port_latitude_max_deg": max(lat),
            "port_latitude_swing_deg": max(lat) - min(lat)}


def cmd_openings(args) -> None:
    rows = [opening_record(k) | (port_latitude_band(k) if i < args.band_count else {})
            for i, k in enumerate(range(args.first, args.first + args.count))]
    illegal = [r for r in rows if r["violations"]]

    def stats(key):
        x = np.array([r[key] for r in rows if key in r])
        return {"min": float(x.min()), "p05": float(np.percentile(x, 5)), "median": float(np.median(x)),
                "p95": float(np.percentile(x, 95)), "max": float(x.max())}

    keys = ("range_m", "radial_speed_m_s", "tangential_speed_m_s", "speed_m_s", "radial_envelope_margin_m_s",
            "fov_margin_deg", "tumble_deg_s", "spin_offset_body_z_deg", "omega_to_h_deg", "body_z_to_h_deg",
            "port_latitude_deg", "los_latitude_deg", "tangential_rate_over_spin",
            "port_latitude_min_deg", "port_latitude_max_deg", "port_latitude_swing_deg")
    summary = {"openings": [args.first, args.first + args.count - 1], "count": len(rows),
               "t0_illegal": len(illegal), "t0_illegal_openings": [r["opening"] for r in illegal],
               "stats": {k: stats(k) for k in keys}}
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


# --------------------------------------------------------------------------- #
def feasibility_record(opening: int) -> dict[str, Any]:
    """Planner response at t = 0: feasibility and planned transfer delta-v over the neutral tau grid."""

    from env.t1_n_review import T1NReviewEnv

    env = T1NReviewEnv(openings=[opening], earliest_summary=False)
    env.reset()
    e, g = env.env, env.generator
    t, c = e.target_state, e.chaser_state
    q0 = c.position - t.position
    v0 = c.rotation @ c.velocity - t.rotation @ t.velocity
    from env.t1_gstar import free_body_rotations
    from env.t1_n_task import T1N_ENTRY_SPEED_M_S

    history = free_body_rotations(t.rotation, t.omega, e.target_parameters.inertia, max(TAU_GRID_S) + 1.0)
    started = perf_counter()
    grid = []
    for tau in TAU_GRID_S:
        res = g.planner.solve(q0, v0, tau, history, None, entry_speed_m_s=T1N_ENTRY_SPEED_M_S)
        grid.append({"tau_s": tau, "feasible": bool(res["feasible"]), "delta_v_m_s": res["delta_v_m_s"]})
    wall = perf_counter() - started
    earliest, _ = g.earliest_feasible(e, T1N_ENTRY_SPEED_M_S)
    feas = [r for r in grid if r["feasible"]]
    # feasible intervals on the grid (a gap of one grid step or more separates them)
    intervals, current = [], None
    for r in grid:
        if r["feasible"]:
            current = [r["tau_s"], r["tau_s"]] if current is None else [current[0], r["tau_s"]]
        elif current is not None:
            intervals.append(current)
            current = None
    if current is not None:
        intervals.append(current)
    dv = [r["delta_v_m_s"] for r in feas]
    local_minima = sum(1 for i in range(1, len(grid) - 1)
                       if grid[i]["feasible"] and grid[i - 1]["feasible"] and grid[i + 1]["feasible"]
                       and grid[i]["delta_v_m_s"] < grid[i - 1]["delta_v_m_s"]
                       and grid[i]["delta_v_m_s"] < grid[i + 1]["delta_v_m_s"])
    return {"opening": opening, "grid": grid, "feasible_fraction": len(feas) / len(grid),
            "first_feasible_s": feas[0]["tau_s"] if feas else None, "earliest_grid_s": earliest,
            "intervals": intervals, "dv_min": min(dv) if dv else None, "dv_max": max(dv) if dv else None,
            "tau_at_dv_min": feas[int(np.argmin(dv))]["tau_s"] if dv else None,
            "dv_local_minima": local_minima, "solve_wall_s": wall / len(grid)}


def cmd_feasibility(args) -> None:
    openings = list(range(args.first, args.first + args.count))
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(feasibility_record, openings))
    frac = np.array([r["feasible_fraction"] for r in rows])
    first = np.array([r["first_feasible_s"] if r["first_feasible_s"] is not None else np.nan for r in rows])
    summary = {
        "openings": [openings[0], openings[-1]], "tau_grid_s": [TAU_GRID_S[0], TAU_GRID_S[-1], 5.0],
        "feasible_fraction": {"mean": float(frac.mean()), "min": float(frac.min()), "max": float(frac.max())},
        "no_feasible_request": int(sum(r["first_feasible_s"] is None for r in rows)),
        "first_feasible_s": {"median": float(np.nanmedian(first)), "p05": float(np.nanpercentile(first, 5)),
                             "p95": float(np.nanpercentile(first, 95))},
        "openings_with_gap_in_feasible_set": int(sum(len(r["intervals"]) > 1 for r in rows)),
        "openings_with_interior_dv_minimum": int(sum(r["dv_local_minima"] > 0 for r in rows)),
        "tau_at_dv_min_minus_first_s": [float(r["tau_at_dv_min"] - r["first_feasible_s"]) for r in rows
                                        if r["tau_at_dv_min"] is not None],
        "dv_range_m_s": [[r["dv_min"], r["dv_max"]] for r in rows],
        "solve_wall_s_mean": float(np.mean([r["solve_wall_s"] for r in rows])),
    }
    Path(args.output).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("dv_range_m_s", "tau_at_dv_min_minus_first_s")},
                     indent=1))


# --------------------------------------------------------------------------- #
# Closed-loop fixed policies (truth). Policies act only at review states.
#   fixed:<a>         no-plan requests use the neutral action a, live plans are kept
#   earliest:<k>      request the observed earliest feasible arrival + k s (0 = G*-earliest rule on
#                     the conditional G*), keep live plans
#   random:<seed>     uniform a at every review (a stand-in for an untrained policy)
#   reject            always ask tau_low (20 s) without a plan and a +30 s shift with one
#   oldgstar          the T1 v0.6 G* with its own earliest search (T1-L interface, delay 0,
#                     kappa 1.0, v_in 0.02) -- the full G*-earliest reference
# --------------------------------------------------------------------------- #


def _policy(name: str, env):
    from env.t1_n_review import N_DUE_INDEX, N_HAS_PLAN_INDEX, action_for_tau

    kind, _, arg = name.partition(":")
    rng = np.random.default_rng(int(arg) if kind == "random" else 0)

    def act(obs):
        if obs[N_DUE_INDEX] < 0.5:
            return np.zeros(1)
        has_plan = obs[N_HAS_PLAN_INDEX] > 0.5
        if kind == "fixed":
            return np.zeros(1) if has_plan else np.array([float(arg)])
        if kind == "earliest":
            if has_plan:
                return np.zeros(1)
            earliest = env.earliest_feasible_summary()
            if earliest is None:
                return np.array([1.0])
            return np.array([action_for_tau(earliest + float(arg), env.env.time_s)])
        if kind == "random":
            return rng.uniform(-1.0, 1.0, size=1)
        if kind == "reject":
            return np.array([1.0]) if has_plan else np.array([-1.0])
        raise ValueError(name)

    return act


def run_episode(job: tuple[int, str, str]) -> dict[str, Any]:
    opening, policy, out_dir = job
    path = Path(out_dir) / f"{policy.replace(':', '_')}_{opening}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = perf_counter()
    steps = []
    if policy == "oldgstar":
        from env.t1_n_task import t1n_env_config
        from env.t1_task_review import T1LReviewEnv

        env = T1LReviewEnv(env_config=t1n_env_config(), openings=[opening], delay_max_s=90.0)
        obs, _ = env.reset()
        action = np.array([-1.0, 1.0, -1.0])  # delay 0, kappa 1.0, v_in 0.02; keep with a live plan
        done = False
        while not done:
            a = action.copy()
            if env.generator.remaining_s(env.env) > 0.0:
                a[0] = 0.0
            obs, reward, term, trunc, info = env.step(a)
            x = info["execution"]
            steps.append({"t": env.env.time_s, "r": reward, "dv": info["delta_v_step_m_s"], "status": x["plan_status"],
                          "tau_req": x["tau_request_s"], "tau_eff": x["tau_effective_s"],
                          "reviewed": info["review"]["reviewed"]})
            done = term or trunc
    else:
        from env.t1_n_review import T1NReviewEnv

        env = T1NReviewEnv(openings=[opening])
        obs, _ = env.reset()
        act = _policy(policy, env)
        done = False
        while not done:
            obs, reward, term, trunc, info = env.step(act(obs))
            x, rv = info["execution"], info["review"]
            steps.append({"t": env.env.time_s, "r": reward, "dv": info["delta_v_step_m_s"], "status": x["plan_status"],
                          "tau_req": x["tau_request_s"], "tau_eff": x["tau_effective_s"], "reviewed": rv["reviewed"],
                          "outcome": rv.get("outcome"), "mode": rv.get("mode")})
            done = term or trunc
    summary = info.get("episode_summary", {})
    record = {"opening": opening, "policy": policy, "event": info["event"], "return": float(sum(s["r"] for s in steps)),
              "time_s": env.env.time_s, "delta_v_m_s": float(sum(s["dv"] for s in steps)),
              "violations": summary.get("violations", {}), "end_phase": summary.get("end_phase"),
              "max_range_m": summary.get("max_range_m"), "force_saturated_fraction": summary.get("force_saturated_fraction"),
              "statuses": {k: sum(1 for s in steps if s["status"].startswith(k))
                           for k in ("optimal", "kept", "hold", "descent", "recovery")},
              "rejections": sum(1 for s in steps if s.get("outcome") == "rejected" and s["reviewed"]),
              "first_tau_effective_s": next((s["tau_eff"] for s in steps if s["status"].startswith("optimal")), None),
              "wall_s": perf_counter() - started, "steps": steps}
    path.write_text(json.dumps(record))
    return record


def cmd_closed(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    openings = list(range(args.first, args.first + args.count))
    jobs = [(k, p, str(out)) for p in args.policies.split(",") for k in openings]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_episode, jobs))
    table = {}
    for p in args.policies.split(","):
        rs = [r for r in rows if r["policy"] == p]
        table[p] = {"success": sum(r["event"] == "success" for r in rs), "unsafe": sum(r["event"] == "unsafe" for r in rs),
                    "timeout": sum(r["event"] == "timeout" for r in rs), "n": len(rs),
                    "return_mean": float(np.mean([r["return"] for r in rs])),
                    "time_mean_s": float(np.mean([r["time_s"] for r in rs])),
                    "delta_v_mean": float(np.mean([r["delta_v_m_s"] for r in rs])),
                    "wall_s_mean": float(np.mean([r["wall_s"] for r in rs]))}
    (out / "table.json").write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


def main(argv: list[str] | None = None) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("openings")
    p.add_argument("--first", type=int, default=292000)
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--band-count", type=int, default=200)
    p.add_argument("--output", required=True)
    p = sub.add_parser("feasibility")
    p.add_argument("--first", type=int, default=292000)
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True)
    p = sub.add_parser("closed")
    p.add_argument("--first", type=int, default=292000)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--policies", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p = sub.add_parser("rank")
    p.add_argument("--first", type=int, default=292100)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reviews", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", required=True)
    p = sub.add_parser("timing")
    p.add_argument("--opening", type=int, default=292001)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"timing": cmd_timing, "openings": cmd_openings, "feasibility": cmd_feasibility, "closed": cmd_closed, "rank": cmd_rank}[args.cmd](args)



# --------------------------------------------------------------------------- #
# Ranking-signal diagnostic (research-layer ruling 2026-10-18, item 4.3): G* plan value, G* + MPC
# 2 s correction, and the long-run truth difference at the same review states
# --------------------------------------------------------------------------- #


def _earliest_continuation(obs, env):
    from env.t1_n_review import N_HAS_PLAN_INDEX, action_for_tau

    if obs[N_HAS_PLAN_INDEX] > 0.5:
        return np.zeros(1)
    earliest = env.earliest_feasible_summary()
    return np.array([1.0]) if earliest is None else np.array([action_for_tau(earliest, env.env.time_s)])


def rank_record(job: tuple[int, int, int, str]) -> dict[str, Any]:
    """At the first ``reviews`` review states (outside the descent) of a random-policy episode:
    the ranking signal of the executed action and its alternative (``env.t1_n_cf``), and both truth
    branches flown to the end (continuation: keep; earliest feasible at a no-plan review)."""

    opening, seed, reviews, out_dir = job
    path = Path(out_dir) / f"rank_{opening}_{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    from env.t1_n_cf import RollingPredictor, T1NShadowConfig, alternative_action, rank_signal
    from env.t1_n_review import T1NReviewEnv

    env = T1NReviewEnv(openings=[opening])
    obs, _ = env.reset()
    cfg = T1NShadowConfig()
    pred = RollingPredictor(env_config=env.env.config, config=cfg)
    rng = np.random.default_rng(seed * 1_000_003 + opening)
    mass = env.env.chaser_parameters.mass
    rows: list[dict[str, Any]] = []
    done = False
    while not done and len(rows) < reviews:
        action = np.zeros(1)
        if env._due:
            action = rng.uniform(-1.0, 1.0, size=1)
            alt, kind = alternative_action(env, action)
            if alt is not None:
                row: dict[str, Any] = {"t": env.env.time_s, "kind": kind, "a": float(action[0]), "alt": float(alt[0])}
                preds = {}
                for name, act in (("real", action), ("alt", alt)):
                    pred.sync(env, fresh_solvers=True)
                    p = pred.rollout(act, 1)
                    preds[name] = p
                    pred.sync(env, fresh_solvers=True)
                    started = perf_counter()
                    te = pred.truth_branch(act, None, _earliest_continuation)
                    row[f"truth_{name}"] = {"return": te["discounted_return"], "first": te["rewards"][0],
                                            "event": te["event"], "steps": te["steps"], "wall_s": perf_counter() - started}
                    row[f"pred_{name}"] = {"first": p["rewards"][0] if p["rewards"] else None, "outcome": p["outcome"],
                                           "tau_eff": p["tau_effective_s"], "violation": p["violation"],
                                           "wall_s": p["wall_s"]}
                error = None if preds["real"]["rewards"] == [] else abs(preds["real"]["rewards"][0] - row["truth_real"]["first"])
                near = env.env.time_s + 2.0 > env.env.config.max_time_s - cfg.end_margin_s
                r = rank_signal(cfg, preds["real"], preds["alt"], env.env.config, mass, error, near)
                row.update({"signal": r["signal"], "plan_only": r["plan_only"], "weight": r["weight"],
                            "reasons": r["reasons"], "mpc_correction_real": r["mpc_correction_real"],
                            "mpc_correction_alt": r["mpc_correction_alt"], "first_reward_error": error,
                            "truth_diff": row["truth_alt"]["return"] - row["truth_real"]["return"],
                            "truth_first_diff": row["truth_alt"]["first"] - row["truth_real"]["first"]})
                rows.append(row)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
    record = {"opening": opening, "seed": seed, "rows": rows}
    path.write_text(json.dumps(record, default=float))
    return record


def summarise_rank(records: list[dict[str, Any]], min_truth: float = 0.5) -> dict[str, Any]:
    rows = [r for rec in records for r in rec["rows"]]
    have = [r for r in rows if r["signal"] is not None]
    sig = [r for r in have if abs(r["truth_diff"]) > min_truth]

    def agree(key, rs):
        return int(sum(np.sign(r[key]) == np.sign(r["truth_diff"]) for r in rs))

    weighted = [r for r in sig if r["weight"] > 0.0]
    flips = [r for r in have if np.sign(r["signal"]) != np.sign(r["plan_only"])]
    out = {
        "pairs": len(rows), "pairs_with_both_plans": len(have),
        "pairs_with_weight": int(sum(r["weight"] > 0.0 for r in rows)),
        "weight_fraction": float(np.mean([r["weight"] > 0.0 for r in rows])) if rows else None,
        "weight_mean": float(np.mean([r["weight"] for r in rows])) if rows else None,
        "reasons": {k: int(sum(any(x.startswith(k) for x in r["reasons"]) for r in rows))
                    for k in ("alt_", "real_", "no_plan", "same_effective_tau", "e_r1", "small_signal", "near_horizon")},
        "pairs_truth_identifiable": len(sig),
        "sign_agreement_plan_only": agree("plan_only", sig), "sign_agreement_combined": agree("signal", sig),
        "sign_agreement_combined_weighted_pairs": [agree("signal", weighted), len(weighted)],
        "weighted_agreement_combined": float(sum(r["weight"] * (np.sign(r["signal"]) == np.sign(r["truth_diff"]))
                                                 for r in weighted) / max(sum(r["weight"] for r in weighted), 1e-12))
        if weighted else None,
        "rank_flips_by_mpc_correction": len(flips),
        "rank_flips_that_match_truth": int(sum(np.sign(r["signal"]) == np.sign(r["truth_diff"]) for r in flips)),
        "mpc_correction_of_D_abs_median": float(np.median([abs(r["signal"] - r["plan_only"]) for r in have])) if have else None,
        "plan_only_D_abs_median": float(np.median([abs(r["plan_only"]) for r in have])) if have else None,
        "corr_plan_only_truth": float(np.corrcoef([r["plan_only"] for r in have], [r["truth_diff"] for r in have])[0, 1])
        if len(have) > 2 else None,
        "corr_combined_truth": float(np.corrcoef([r["signal"] for r in have], [r["truth_diff"] for r in have])[0, 1])
        if len(have) > 2 else None,
        "abs_error_plan_only_vs_truth_median": float(np.median([abs(r["plan_only"] - r["truth_diff"]) for r in have])) if have else None,
        "abs_error_combined_vs_truth_median": float(np.median([abs(r["signal"] - r["truth_diff"]) for r in have])) if have else None,
        "mpc_first_step_error_median": float(np.median([abs(r[f"pred_{b}"]["first"] - r[f"truth_{b}"]["first"])
                                                        for r in rows for b in ("real", "alt") if r[f"pred_{b}"]["first"] is not None])) if rows else None,
        "plan_first_step_error_median": float(np.median([abs((r[f"pred_{b}"]["first"] - r[f"mpc_correction_{b}"]) - r[f"truth_{b}"]["first"])
                                                         for r in have for b in ("real", "alt")])) if have else None,
        "by_kind": {k: {"n": sum(r["kind"] == k for r in sig),
                        "plan_only": agree("plan_only", [r for r in sig if r["kind"] == k]),
                        "combined": agree("signal", [r for r in sig if r["kind"] == k])}
                    for k in ("request_shift", "keep_vs_shift", "replan_vs_keep")},
        "truth_abs_diff_median": float(np.median([abs(r["truth_diff"]) for r in rows])) if rows else None,
        "shadow_wall_per_branch_s": float(np.median([r[f"pred_{b}"]["wall_s"] for r in rows for b in ("real", "alt")])) if rows else None,
    }
    return out


def cmd_rank(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, args.seed, args.reviews, str(out)) for k in range(args.first, args.first + args.count)]
    with ProcessPoolExecutor(args.workers) as pool:
        records = list(pool.map(rank_record, jobs))
    summary = summarise_rank(records)
    (out / "rank_summary.json").write_text(json.dumps({"summary": summary,
                                                       "rows": [dict(opening=rec["opening"], **r) for rec in records
                                                                for r in rec["rows"]]}, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


def cmd_timing(args) -> None:
    """Serial wall per real 2 s step, plain interface vs the ranking-signal interface (same actions)."""

    from env.t1_n_cf import T1NReviewCFEnv
    from env.t1_n_review import T1NReviewEnv

    results = {}
    for label, factory in (("plain", lambda: T1NReviewEnv(openings=[args.opening])),
                           ("rank_signal", lambda: T1NReviewCFEnv(openings=[args.opening]))):
        env = factory()
        env.reset()
        rng = np.random.default_rng(3)
        walls, reviews, overhead = [], 0, 0.0
        for _ in range(args.steps):
            a = rng.uniform(-1.0, 1.0, size=1)
            started = perf_counter()
            _, _, term, trunc, info = env.step(a)
            walls.append(perf_counter() - started)
            reviews += int(info["review"]["reviewed"])
            cf = info.get("counterfactual")
            overhead += cf["overhead_wall_s"] if cf else 0.0
            if term or trunc:
                break
        results[label] = {"steps": len(walls), "wall_per_step_s": float(np.mean(walls)), "reviews": reviews,
                          "shadow_wall_per_step_s": overhead / len(walls)}
    base = results["plain"]["wall_per_step_s"]
    for v in results.values():
        v["relative_to_plain"] = v["wall_per_step_s"] / base
    Path(args.output).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
