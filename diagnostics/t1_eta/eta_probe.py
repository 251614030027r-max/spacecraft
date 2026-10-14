"""Stage B of the execution-trade-off review: is the MPC input weight eta_u worth learning?

No training, no frozen-code change: a fixed eta_u is ``T1EnvConfig.input_weight = 0.01 * eta``
(eta = 1 is the frozen configuration itself). The task intent is held at the
earliest-feasible rule (keep the plan, kappa 0.7) so that eta is isolated from the SAC
value errors of P1-P3. Pre-registered thresholds: collab reviews/T1_EXEC_TRADEOFF_STAGE_A_REVIEW_20261014.md.

    # B1 + B3: 2 s closed loop and 2 s shadow prediction per eta at 4 states per opening
    python -B -m diagnostics.t1_eta.eta_probe b1 --openings 295001,... --etas 0.3,1,3,10,30 --output-dir <dir> --workers 8
    # B2: whole episodes (rule intent) with a fixed eta, a phase schedule, or Pure with a fixed eta
    python -B -m diagnostics.t1_eta.eta_probe b2 --mode fixed --eta 10 --openings <16> --output-dir <dir> --workers 8
    python -B -m diagnostics.t1_eta.eta_probe b2 --mode schedule --schedule plan_start=3,tracking=10,descent=1 ...
    python -B -m diagnostics.t1_eta.eta_probe b2 --mode pure --eta 10 ...
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from diagnostics.t1_rootcause.common import live_plan, mode_action
from env.t1_env import T1Env, T1EnvConfig
from env.t1_gym import T1GStarEnv

BASE_INPUT_WEIGHT = T1EnvConfig().input_weight  # 0.01, frozen


def env_config(eta: float) -> T1EnvConfig:
    return replace(T1EnvConfig(), input_weight=BASE_INPUT_WEIGHT * float(eta))


def rule(env: T1GStarEnv) -> np.ndarray:
    return mode_action("E", env, None)


def phase(env: T1GStarEnv) -> str:
    """Pre-registered phase classes of the schedule (decided before the step)."""

    g, e = env.generator, env.env
    assert g is not None
    if g._descending(e, e.metrics()):
        return "descent"
    return "tracking" if live_plan(env) else "plan_start"


def _margins(env: T1GStarEnv) -> dict:
    m = env.env.metrics()
    return {"union_margin_m": float(m.union_margin_m), "fov_margin_deg": float(np.degrees(m.fov_margin_rad)),
            "radial_envelope_margin_m_s": float(m.radial_envelope_margin_m_s),
            "total_speed_margin_m_s": float(m.total_speed_margin_m_s),
            "axial_envelope_margin_m_s": float(m.axial_envelope_margin_m_s)}


# ---------------------------------------------------------------------- B1 / B3
def select_decisions(log: list[dict]) -> dict[str, int]:
    """Pre-registered: plan_init = 1; first_descent = first 'descent' decision d;
    mid_approach = d // 2; max_saturation = argmax force-saturated fraction over 2 <= k < d."""

    d = next((x["k"] for x in log if x["status"] == "descent"), None)
    if d is None or d < 4:
        return {}
    sat = max(range(2, d), key=lambda k: log[k]["force_saturated_fraction"])
    picks = {"plan_init": 1, "mid_approach": d // 2, "max_saturation": sat, "first_descent": d}
    return picks


def b1_opening(opening: int, etas: list[float], output_dir: str) -> None:
    from env.t1_counterfactual import CounterfactualPredictor

    path = Path(output_dir) / f"b1_{opening}.json"
    if path.exists():
        return
    started = perf_counter()
    # pass 1: the rule trajectory at eta = 1, to select the decisions
    env = T1GStarEnv()
    env.reset(options={"opening": opening})
    log = []
    for k in range(10_000):
        _, _, terminated, truncated, info = env.step(rule(env))
        x = info["execution"]
        log.append({"k": k, "status": x["plan_status"], "force_saturated_fraction": x["force_saturated_fraction"]})
        if terminated or truncated:
            break
    picks = select_decisions(log)
    # pass 2: replay and branch at each selected decision
    env = T1GStarEnv()
    env.reset(options={"opening": opening})
    closed = {eta: CounterfactualPredictor(env_config=env_config(eta)) for eta in etas}
    predicted = {eta: CounterfactualPredictor(env_config=env_config(eta)) for eta in etas}
    states = {}
    order = sorted(picks.items(), key=lambda kv: kv[1])
    k = 0
    for name, target in order:
        while k < target:
            env.step(rule(env))
            k += 1
        action = rule(env)
        rows = {}
        for eta in etas:
            sh = closed[eta]
            sh.sync(env)
            dv0 = sh.shadow.env._delta_v
            obs, reward, terminated, truncated, info = sh.shadow.step(action)
            x = info["execution"]
            rows[eta] = {
                "real": {"reward": reward, "delta_v_m_s": sh.shadow.env._delta_v - dv0, "event": info["event"],
                         "first_wrench": None if sh.shadow.first_wrench is None else sh.shadow.first_wrench.tolist(),
                         "force_saturated_fraction": x["force_saturated_fraction"],
                         "max_slack_solved": x["max_slack_solved"], "plan_deviation_end_m": x["plan_deviation_end_m"],
                         "plan_status": x["plan_status"], "margins_end": _margins(sh.shadow)},
            }
            p = predicted[eta]
            p.sync(env)
            pr = p.predict(action)
            c = env.env.config
            rows[eta]["predicted"] = {
                "reward": pr.reward,
                "delta_v_m_s": (-pr.reward - c.reward.time_cost_per_s * c.dt_s * c.decision_steps)
                / c.reward.delta_v_cost_per_m_s,
                "saturation": pr.saturation, "solve_slack": pr.solve_slack, "plan_status": pr.plan_status}
        states[name] = {"k": target, "time_s": env.env.time_s, "phase": phase(env), "by_eta": rows}
    out = {"opening": opening, "picks": picks, "states": states, "etas": etas,
           "rule_log_decisions": len(log), "wall_s": perf_counter() - started}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    print("b1", opening, picks, "wall %.0f s" % out["wall_s"], flush=True)


# ---------------------------------------------------------------------- B2
def b2_episode(mode: str, eta: float | None, schedule: dict[str, float] | None, opening: int, output_dir: str) -> None:
    from env.t1_counterfactual import CounterfactualPredictor

    tag = f"{mode}_{eta}" if mode != "schedule" else "schedule_" + "_".join(f"{k}{v}" for k, v in sorted(schedule.items()))
    path = Path(output_dir) / f"b2_{tag}_{opening}.json"
    if path.exists():
        return
    started = perf_counter()
    if mode == "pure":
        env = T1Env(env_config(eta))
        env.reset(opening)
        gamma, ret, k = env.config.reward.gamma, 0.0, 0
        while True:
            reward, terminated, truncated, info = env.step(pure=True)
            ret += gamma**k * reward
            k += 1
            if terminated or truncated:
                break
        summary = env.summary()
        result = {"event": info["event"], "discounted_return": ret, **summary,
                  "violations": dict(info.get("violations", {}))}
    else:
        etas = sorted(set(schedule.values())) if mode == "schedule" else [eta]
        envs = {e: CounterfactualPredictor(env_config=env_config(e)).shadow for e in etas}
        active = envs[etas[0]] if mode == "fixed" else envs[schedule["plan_start"]]
        active.reset(options={"opening": opening})
        gamma, ret, k, used = active.env.config.reward.gamma, 0.0, 0, []
        while True:
            if mode == "schedule":
                want = envs[schedule[phase(active)]]
                if want is not active:
                    p = CounterfactualPredictor.__new__(CounterfactualPredictor)
                    p.shadow = want
                    p.sync(active)
                    active = want
                used.append(phase(active))
            _, reward, terminated, truncated, info = active.step(rule(active))
            ret += gamma**k * reward
            k += 1
            if terminated or truncated:
                break
        s = info["episode_summary"]
        result = {"event": s["event"], "discounted_return": ret, "time_s": s["time_s"], "delta_v_m_s": s["delta_v_m_s"],
                  "decisions": s["decisions"], "min_margins": s["min_margins"], "violations": s["violations"],
                  "end_phase": s["end_phase"], "phases_used": {ph: used.count(ph) for ph in set(used)}}
    result.update({"mode": mode, "eta": eta, "schedule": schedule, "opening": opening, "wall_s": perf_counter() - started})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    print("b2", tag, opening, result["event"], round(result["discounted_return"], 2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("b1", "b2"))
    parser.add_argument("--openings", required=True)
    parser.add_argument("--etas", default="0.3,1,3,10,30")
    parser.add_argument("--mode", choices=("fixed", "schedule", "pure"))
    parser.add_argument("--eta", type=float)
    parser.add_argument("--schedule", help="plan_start=..,tracking=..,descent=..")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    openings = [int(o) for o in args.openings.split(",")]
    if args.command == "b1":
        etas = [float(e) for e in args.etas.split(",")]
        jobs, fn = [(o, etas, str(args.output_dir)) for o in openings], b1_opening
    else:
        schedule = ({k: float(v) for k, v in (kv.split("=") for kv in args.schedule.split(","))}
                    if args.mode == "schedule" else None)
        jobs, fn = [(args.mode, args.eta, schedule, o, str(args.output_dir)) for o in openings], b2_episode
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(fn, jobs)
    else:
        for job in jobs:
            fn(*job)


if __name__ == "__main__":
    main()
