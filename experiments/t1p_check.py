"""T1-P technical checks (T1_P_FAST_TRACK_CONDITIONAL_GO_20261018 sec. 2): correctness, not performance.

Development openings only (292000-293999).

    python -B -m experiments.t1p_check stages --opening 292000 --output stages.json      # 7 stage actions, truth
    python -B -m experiments.t1p_check pipeline --opening 292000 --output pipeline.json  # geometric rule to the end
    python -B -m experiments.t1p_check pair --opening 292000 --pairs 2 --output pair.json  # stage pairs vs truth
    python -B -m experiments.t1p_check timing --opening 292001 --steps 24 --output timing.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

ACTIONS = {"neutral": [0, 0, 0], "r+": [1, 0, 0], "r-": [-1, 0, 0], "t+": [0, 1, 0], "t-": [0, -1, 0],
           "b+": [0, 0, 1], "b-": [0, 0, -1]}


def _truth_stage(env, action, steps: int = 6) -> dict[str, Any]:
    from env.t1_p_stage import relative_inertial

    rec: dict[str, Any] = {"rewards_raw": [], "rewards_train": [], "modes": [], "statuses": [], "min_range_m": np.inf, "violations": {}}
    jumps = []
    obs, r, term, trunc, info = env.step(np.asarray(action, dtype=np.float64))
    req = info["execution"]["request"]
    rec["request"] = req
    for k in range(steps):
        if k > 0:
            if env._due:
                break
            obs, r, term, trunc, info = env.step(np.zeros(3))
        rec["rewards_raw"].append(info["reward_raw"])
        rec["rewards_train"].append(float(r))
        rec["modes"].append(info["execution"]["mode"])
        rec["statuses"].append(info["execution"]["plan_status"])
        rec["min_range_m"] = min(rec["min_range_m"], float(env.env.metrics().range_m))
        jumps.append(env.env.summary()["reference_jump_max_m"])
        if term or trunc:
            rec["violations"] = dict(info.get("violations", {}))
            rec["event"] = info["event"]
            break
    q, v = relative_inertial(env.env)
    rec.update(q_end=q.tolist(), v_end=v.tolist(), reference_jump_max_m=float(max(jumps)) if jumps else None,
               steps=len(rec["rewards_raw"]), event=rec.get("event"), summary_min_margins=env.env.summary()["min_margins"])
    if req is not None:
        rec["goal_error_m"] = float(np.linalg.norm(q - np.asarray(req["q_goal"])))
        rec["goal_velocity_error_m_s"] = float(np.linalg.norm(v - np.asarray(req["v_goal"])))
    return rec


def cmd_stages(args) -> None:
    from env.t1_p_cf import StageShadow
    from env.t1_p_review import T1PReviewEnv

    real = T1PReviewEnv(openings=[args.opening])
    real.reset()
    branch = StageShadow(None, real.env.config)
    out: dict[str, Any] = {"opening": args.opening, "stages": {}}
    for name, a in ACTIONS.items():
        branch.sync(real)
        out["stages"][name] = _truth_stage(branch.shadow, a)
    ends = {k: np.asarray(v["q_end"]) for k, v in out["stages"].items()}
    out["pairwise_end_distance_m"] = {f"{a}|{b}": float(np.linalg.norm(ends[a] - ends[b]))
                                      for a in ends for b in ends if a < b}
    out["repeat_neutral_bitwise"] = None
    branch.sync(real)
    again = _truth_stage(branch.shadow, ACTIONS["neutral"])
    out["repeat_neutral_bitwise"] = again["q_end"] == out["stages"]["neutral"]["q_end"]
    Path(args.output).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: {"steps": v["steps"], "modes": sorted(set(v["modes"])), "statuses": v["statuses"][:1],
                          "goal_error_m": v.get("goal_error_m"), "min_range_m": v["min_range_m"],
                          "event": v["event"], "jump": v["reference_jump_max_m"],
                          "plan_terminal_error_m": (v["request"] or {}).get("terminal_position_error_m")}
                      for k, v in out["stages"].items()}, indent=1, default=float))
    print("min pairwise end distance", min(out["pairwise_end_distance_m"].values()), "repeat", out["repeat_neutral_bitwise"])


def cmd_pipeline(args) -> None:
    from env.t1_p_review import T1PReviewEnv, geometric_rule

    env = T1PReviewEnv(openings=[args.opening])
    env.reset()
    started = perf_counter()
    timeline, done, info = [], False, {}
    raw = 0.0
    while not done:
        a = geometric_rule(env) if env._due else np.zeros(3)
        obs, r, term, trunc, info = env.step(a)
        raw += info["reward_raw"]
        x = info["execution"]
        timeline.append({"t": env.env.time_s, "mode": x["mode"], "status": x["plan_status"], "event": x["generator_event"],
                         "reviewed": info["review"]["reviewed"], "range": float(env.env.metrics().range_m),
                         "r_raw": info["reward_raw"], "r_train": r})
        done = term or trunc
        if args.max_steps and len(timeline) >= args.max_steps:
            break
    s = info.get("episode_summary", {})
    out = {"opening": args.opening, "event": info.get("event"), "return_raw": raw, "time_s": env.env.time_s,
           "delta_v_m_s": env.env.summary()["delta_v_m_s"], "violations": s.get("violations"),
           "modes_seen": sorted({x["mode"] for x in timeline}),
           "first_bridge_s": next((x["t"] for x in timeline if x["mode"] == "bridge"), None),
           "first_descent_s": next((x["t"] for x in timeline if x["mode"] == "descent"), None),
           "events": [x for x in timeline if x["event"]], "wall_s": perf_counter() - started, "timeline": timeline}
    Path(args.output).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("timeline",)}, indent=1, default=float))


def cmd_pair(args) -> None:
    """Rule-driven episode; at the first ``--pairs`` sampled reviews: the stage pair, the real stage,
    and (diagnostics only, never training data) the alternative's truth stage."""

    from env.t1_p_cf import StageShadow, T1PReviewCFEnv
    from env.t1_p_review import geometric_rule

    env = T1PReviewCFEnv(openings=[args.opening])
    env.reset()
    truth = StageShadow(None, env.env.config)
    rows, done = [], False
    pending = None
    while not done and len(rows) < args.pairs:
        a = geometric_rule(env) if env._due else np.zeros(3)
        alt_truth = None
        if env._due:
            from env.t1_p_cf import alternative

            alt, _, _ = alternative(int(env.current_opening), env._review_index, a, env.cf_config)
            if alt is not None:
                truth.sync(env)
                alt_truth = _truth_stage(truth.shadow, alt)
        obs, r, term, trunc, info = env.step(a)
        cf = info["counterfactual"]
        if cf["kind"] == "stage_pair":
            pending = {"t": env.env.time_s - 2.0, "cf": {k: v for k, v in cf.items() if k not in ("z_hat", "a_tilde")},
                       "a": a.tolist(), "a_tilde": cf["a_tilde"].tolist(), "alt_truth": alt_truth,
                       "z_hat_finite": bool(np.all(np.isfinite(cf["z_hat"]))) if cf["z_hat"] is not None else None,
                       "z_hat_due": None if cf["z_hat"] is None else float(cf["z_hat"][-2])}
        if "cf_final" in info and pending is not None:
            pending["final"] = info["cf_final"]
            if pending["alt_truth"] is not None:
                at, gamma = pending["alt_truth"], env.env.config.reward.gamma
                pending["alt_truth_return_train"] = float(sum(gamma ** i * x for i, x in enumerate(at["rewards_train"])))
                pending["alt_pred_minus_truth"] = pending["cf"]["r_hat"] - pending["alt_truth_return_train"]
            rows.append(pending)
            pending = None
        done = term or trunc
    out = {"opening": args.opening, "pairs": rows}
    Path(args.output).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


def cmd_timing(args) -> None:
    from env.t1_p_cf import T1PReviewCFEnv
    from env.t1_p_review import T1PReviewEnv

    results = {}
    for label, factory in (("plain", lambda: T1PReviewEnv(openings=[args.opening])),
                           ("stage_cf", lambda: T1PReviewCFEnv(openings=[args.opening]))):
        env = factory()
        env.reset()
        rng = np.random.default_rng(3)
        walls, reviews, pairs = [], 0, 0
        for _ in range(args.steps):
            a = rng.uniform(-1.0, 1.0, size=3)
            t0 = perf_counter()
            _, _, term, trunc, info = env.step(a)
            walls.append(perf_counter() - t0)
            reviews += int(info["review"]["reviewed"])
            pairs += int((info.get("counterfactual") or {}).get("kind") == "stage_pair")
            if term or trunc:
                break
        results[label] = {"steps": len(walls), "wall_per_step_s": float(np.mean(walls)), "reviews": reviews, "pairs": pairs}
    Path(args.output).write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def main(argv: list[str] | None = None) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("stages", "pipeline", "pair", "timing"):
        p = sub.add_parser(name)
        p.add_argument("--opening", type=int, default=292000)
        p.add_argument("--output", required=True)
        if name == "pipeline":
            p.add_argument("--max-steps", type=int, default=0)
        if name == "pair":
            p.add_argument("--pairs", type=int, default=2)
        if name == "timing":
            p.add_argument("--steps", type=int, default=24)
    args = parser.parse_args(argv)
    {"stages": cmd_stages, "pipeline": cmd_pipeline, "pair": cmd_pair, "timing": cmd_timing}[args.cmd](args)


if __name__ == "__main__":
    main()
