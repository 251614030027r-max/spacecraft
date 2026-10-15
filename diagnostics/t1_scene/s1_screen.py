"""T1-S1 decision-space screen: fixed policies only (no SAC), frozen MPC / G* / reward / truth.

T1-S1 changes only the initial distribution: range 12-20 m, inertial speed <= 0.18 m/s
(T1: 9-15 m, 0.10 m/s). Development openings 296000-296007 (new block; 296000-296003
screen, 296004-296007 locked confirmation).

Policy: the first decision requests tau0 (earliest feasible, or an absolute 70 s / 90 s);
afterwards a live plan is kept (a0 = 0) and with no live plan the earliest feasible
arrival is requested (a0 = -1), as the rule does; kappa is fixed for the episode.

    python -B -m diagnostics.t1_scene.s1_screen --tau earliest --kappa 0.7 --openings 296000,... --output-dir <d> --workers 4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from env.t1_env import T1EnvConfig
from env.t1_gstar import _plan_position
from env.t1_gym import T1GStarEnv
from env.t1_task import T1InitialDistribution

S1 = replace(T1EnvConfig(), initial=replace(T1InitialDistribution(), range_m=(12.0, 20.0), inertial_speed_max_m_s=0.18))
NEW_PLAN = ("optimal", "recovery")


def a0_for_first(env: T1GStarEnv, tau: str) -> float:
    if tau == "earliest":
        return -1.0
    return float(tau) / (0.5 * env.config.tau_max_s) - 1.0  # absolute map 0..tau_max


def a1_for(env: T1GStarEnv, kappa: float) -> float:
    c = env.config
    return 2.0 * (kappa - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0


def episode(tau: str, kappa: float, opening: int, output_dir: str, scene: str = "S1") -> None:
    path = Path(output_dir) / f"{scene}_tau{tau}_k{kappa}_{opening}.json"
    if path.exists():
        return
    started = perf_counter()
    env = T1GStarEnv(env_config=S1 if scene == "S1" else T1EnvConfig())
    obs0, _ = env.reset(options={"opening": opening})
    g = env.generator
    gamma, ret, k, log, accepted = env.env.config.reward.gamma, 0.0, 0, [], []
    while True:
        e = env.env
        remaining = g.remaining_s(e)
        deviation = None
        if g.plan is not None:
            q = e.chaser_state.position - e.target_state.position
            deviation = float(np.linalg.norm(_plan_position(g.plan, e.time_s - g.plan_start_s, g.planner.mass) - q))
        usable = deviation is not None and deviation <= g.replan_deviation_m and remaining > 0.0
        a0 = a0_for_first(env, tau) if k == 0 else (0.0 if remaining > 0.0 else -1.0)
        action = np.array([a0, a1_for(env, kappa)])
        tau_req, kappa_eff = env.intent(action)
        t0, dv0 = float(e.time_s), e._delta_v
        _, reward, terminated, truncated, info = env.step(action)
        x = info["execution"]
        status = x["plan_status"]
        trigger = None
        if status.startswith(NEW_PLAN):
            trigger = "first" if k == 0 else "expired" if remaining <= 0.0 else "deviation" if not usable else "other"
            accepted.append({"k": k, "trigger": trigger, "tau_request_s": tau_req, "tau_s": float(g.plan["tau_s"]),
                             "arrival_abs_s": float(g.plan_start_s + g.plan["tau_s"])})
        log.append({"k": k, "t_s": t0, "status": status, "trigger": trigger, "tau_request_s": tau_req,
                    "kappa": kappa_eff, "slip_s": x["slip_s"], "delta_v_m_s": e._delta_v - dv0, "reward": reward,
                    "range_m": float(e.metrics().range_m)})
        ret += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    s = info["episode_summary"]
    descent = [d for d in log if d["status"] == "descent"]
    result = {
        "scene": scene, "tau": tau, "kappa": kappa, "opening": opening, "initial_observation": obs0.tolist(),
        "initial_range_m": float(obs0[0] * 10.0),
        "event": s["event"], "discounted_return": ret, "time_s": s["time_s"], "delta_v_m_s": s["delta_v_m_s"],
        "min_margins": s["min_margins"], "violations": s["violations"], "end_phase": s["end_phase"],
        "end_range_m": s["end_range_m"], "max_range_m": s["max_range_m"],
        "first_tau_request_s": accepted[0]["tau_request_s"] if accepted else None,
        "first_tau_effective_s": accepted[0]["tau_s"] if accepted else None,
        "triggers": {t: sum(a["trigger"] == t for a in accepted) for t in ("first", "deviation", "expired", "other")},
        "slips": sum(d["slip_s"] > 0 for d in log), "holds": sum(d["status"].startswith("hold") for d in log),
        "descent_decisions": len(descent), "first_descent_t_s": descent[0]["t_s"] if descent else None,
        "far_field_delta_v_m_s": float(sum(d["delta_v_m_s"] for d in log if d["status"] != "descent")),
        "descent_delta_v_m_s": float(sum(d["delta_v_m_s"] for d in descent)),
        "accepted_plans": accepted, "decision_log": log, "wall_s": perf_counter() - started,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(path)
    print(scene, "tau", tau, "k", kappa, opening, result["event"], round(ret, 2), "T", s["time_s"],
          "dv %.2f" % s["delta_v_m_s"], "tau_eff", result["first_tau_effective_s"], "triggers", result["triggers"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", required=True, help="earliest | 70 | 90")
    parser.add_argument("--kappa", type=float, default=0.7)
    parser.add_argument("--openings", required=True)
    parser.add_argument("--scene", default="S1", choices=("S1", "T1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(args.tau, args.kappa, int(o), str(args.output_dir), args.scene) for o in args.openings.split(",")]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(episode, jobs)
    else:
        for job in jobs:
            episode(*job)


if __name__ == "__main__":
    main()
