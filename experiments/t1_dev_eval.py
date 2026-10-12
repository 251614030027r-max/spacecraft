"""Development evaluation of M0/M1 checkpoints and of the earliest-feasible G* rule.

Learning-progress monitoring only (design 2026-10-12, sec. 7.1): the openings
are the frozen development evaluation block, disjoint from training
(>= 1e6), R0 (2900xx), R1 (2910xx) and the formal block (>= 300000).
Deterministic policy actions, plain ``T1GStarEnv`` (no counterfactual), serial.

Per episode: outcome, time, delta-v, the discounted return with the training
discount (0.999 per decision, the primary scalar), and how the learned intent
was used (first tau request, kappa, new plans / keeps / slips / holds).

    python -B -m experiments.t1_dev_eval --policy logs/t1/t1_m0_s0/checkpoints/m0_5000_decisions.zip \
        --openings eval/t1/dev_openings.json --output-dir eval/t1/dev/m0_s0/5000
    python -B -m experiments.t1_dev_eval --policy rule:earliest --openings ... --output-dir eval/t1/dev/rule
    python -B -m experiments.t1_dev_eval --watch logs/t1/t1_m1_s0 --openings experiments/t1_dev_openings_295000.json

``--watch RUN_DIR`` evaluates each frozen checkpoint (5k/10k/15k/20k decisions) of a
training run as it appears, into ``RUN_DIR/dev_eval/<decisions>/``, and exits after the
last one (or when the run wrote ABORTED.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from env.t1_gym import T1GStarEnv

KAPPA_RULE = 0.7


def rule_action(env: T1GStarEnv) -> np.ndarray:
    """Earliest feasible arrival, then keep the plan; kappa 0.7 (the R1 rv:0 row in gym terms)."""

    c = env.config
    a1 = 2.0 * (KAPPA_RULE - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
    return np.array([-1.0 if env.generator.remaining_s(env.env) <= 0.0 else 0.0, a1])


def run_episode(env: T1GStarEnv, opening: int, policy) -> dict:
    obs, _ = env.reset(options={"opening": opening})
    gamma = env.env.config.reward.gamma
    ret, k, log = 0.0, 0, []
    while True:
        action = rule_action(env) if policy is None else np.asarray(policy.predict(obs, deterministic=True)[0])
        tau, kappa = env.intent(action)
        obs, reward, terminated, truncated, info = env.step(action)
        x = info["execution"]
        log.append({"tau_request_s": tau, "kappa": kappa, "status": x["plan_status"], "slip_s": x["slip_s"],
                    "tau_effective_s": x["tau_effective_s"]})
        ret += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    statuses = [s["status"] for s in log]
    return {
        **info["episode_summary"], "opening": opening, "discounted_return": ret,
        "numerical_error": info.get("numerical_error"),
        "first_tau_request_s": log[0]["tau_request_s"], "kappa_mean": float(np.mean([s["kappa"] for s in log])),
        "new_plans": sum(s.startswith("optimal") for s in statuses), "kept": statuses.count("kept"),
        "holds": sum(s.startswith("hold") for s in statuses), "slips": sum(s["slip_s"] > 0 for s in log),
        "decision_log": log,
    }


CHECKPOINTS = (5_000, 10_000, 15_000, 20_000)


def evaluate(policy_arg: str, openings: Path, output_dir: Path) -> dict:
    return main([f"--policy={policy_arg}", f"--openings={openings}", f"--output-dir={output_dir}"])


def watch(run_dir: Path, openings: Path, poll_s: float = 60.0) -> None:
    import time

    done = set()
    while len(done) < len(CHECKPOINTS):
        if (run_dir / "ABORTED.json").exists():
            print("run aborted:", (run_dir / "ABORTED.json").read_text(), flush=True)
            return
        for n in CHECKPOINTS:
            zips = sorted((run_dir / "checkpoints").glob(f"*_{n}_decisions.zip"))
            if n not in done and zips:
                time.sleep(5.0)  # let the save finish
                evaluate(str(zips[0]), openings, run_dir / "dev_eval" / str(n))
                done.add(n)
        if len(done) < len(CHECKPOINTS):
            time.sleep(poll_s)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", help="checkpoint .zip, or rule:earliest")
    parser.add_argument("--watch", type=Path, help="training run directory to follow")
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.watch is not None:
        watch(args.watch, args.openings)
        return {}
    if args.output_dir is None or args.policy is None:
        parser.error("--policy and --output-dir are required without --watch")
    policy = None
    if args.policy != "rule:earliest":
        from train.t1_m1 import M1SAC
        from stable_baselines3 import SAC
        try:
            policy = M1SAC.load(args.policy, device="cpu")
        except Exception:  # noqa: BLE001 -- an M0 checkpoint
            policy = SAC.load(args.policy, device="cpu")
    seeds = json.loads(args.openings.read_text())["seeds"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = T1GStarEnv()
    rows = []
    for opening in seeds:
        path = args.output_dir / f"{opening}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            continue
        row = run_episode(env, opening, policy)
        path.write_text(json.dumps(row, default=float))
        rows.append(row)
        print(opening, row["event"], round(row["time_s"], 1), round(row["delta_v_m_s"], 2),
              round(row["discounted_return"], 2), flush=True)
    summary = {
        "policy": args.policy, "episodes": len(rows),
        "clean_capture": sum(r["event"] == "success" for r in rows),
        "mean_discounted_return": float(np.mean([r["discounted_return"] for r in rows])),
        "mean_time_s_fail_as_300": float(np.mean([r["time_s"] if r["event"] == "success" else 300.0 for r in rows])),
        "mean_delta_v_m_s": float(np.mean([r["delta_v_m_s"] for r in rows])),
    }
    summary["failures"] = {}
    for r in rows:
        if r["event"] != "success":
            for key in ([*r.get("violations", {})] if r["event"] == "unsafe" else [r["event"]]):
                summary["failures"][key] = summary["failures"].get(key, 0) + 1
    summary["openings_sha256"] = __import__("hashlib").sha256(args.openings.read_bytes()).hexdigest()
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
