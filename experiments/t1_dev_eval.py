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
    python -B -m experiments.t1_dev_eval --policy rule:earliest_k10 --openings ... --output-dir eval/t1/dev/rule_k10
    python -B -m experiments.t1_dev_eval --policy rule:earliest_k1.0_v0.10 --openings ... --output-dir eval/t1/dev/rule_k10_v10
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
ENTRY_SPEED_RULE = 0.06
#: fixed rules: earliest feasible arrival then keep, at a fixed kappa (and entry speed, m/s);
#: any ``rule:earliest_k<kappa>_v<entry speed>`` (e.g. rule:earliest_k1.0_v0.10) is parsed too
RULES = {"rule:earliest": (0.7, ENTRY_SPEED_RULE), "rule:earliest_k07": (0.7, ENTRY_SPEED_RULE),
         "rule:earliest_k10": (1.0, ENTRY_SPEED_RULE)}


class Rule:
    def __init__(self, kappa: float, entry_speed_m_s: float = ENTRY_SPEED_RULE) -> None:
        self.kappa, self.entry_speed_m_s = float(kappa), float(entry_speed_m_s)


def parse_rule(policy_arg: str) -> Rule | None:
    if policy_arg in RULES:
        return Rule(*RULES[policy_arg])
    if policy_arg.startswith("rule:earliest_k") and "_v" in policy_arg:
        kappa, speed = policy_arg[len("rule:earliest_k"):].split("_v")
        return Rule(float(kappa), float(speed))
    return None


def rule_action(env: T1GStarEnv, kappa: float = KAPPA_RULE, entry_speed_m_s: float = ENTRY_SPEED_RULE) -> np.ndarray:
    """Earliest feasible arrival, then keep the plan; kappa 0.7 (the R1 rv:0 row in gym terms) and
    G*'s nominal entry speed 0.06 m/s unless given."""

    c = env.config
    a1 = 2.0 * (kappa - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
    a2 = (entry_speed_m_s - c.entry_speed_nominal_m_s) / c.entry_speed_span_m_s
    return np.array([-1.0 if env.generator.remaining_s(env.env) <= 0.0 else 0.0, a1, a2])


def critic_values(policy, env: T1GStarEnv, obs: np.ndarray, action: np.ndarray) -> dict | None:
    """min_j Q_j of the taken action and, with a live plan, of keep / earlier 15 s / later 15 s
    (same a1). Read-only; the critic-calibration curve compares q_taken with the realised
    discounted return-to-go of the deterministic episode."""

    critic = getattr(getattr(policy, "policy", None), "critic", None)
    if critic is None:
        return None
    import torch as th

    actions = {"taken": np.asarray(action, dtype=np.float32)}
    if env.generator.remaining_s(env.env) > 0.0:
        rest = [float(v) for v in np.asarray(action).reshape(-1)[1:]]
        actions |= {"keep": np.array([0.0, *rest], np.float32),
                    "earlier15": np.array([env.action_for_shift(-15.0), *rest], np.float32),
                    "later15": np.array([env.action_for_shift(15.0), *rest], np.float32)}
    with th.no_grad():
        o = th.as_tensor(np.repeat(obs[None].astype(np.float32), len(actions), axis=0))
        a = th.as_tensor(np.stack(list(actions.values())))
        q = th.min(th.cat(critic(o, a), dim=1), dim=1)[0].numpy()
    return {name: float(v) for name, v in zip(actions, q)}


def run_episode(env: T1GStarEnv, opening: int, policy) -> dict:
    obs, _ = env.reset(options={"opening": opening})
    gamma = env.env.config.reward.gamma
    ret, k, log = 0.0, 0, []
    while True:
        if policy is None or isinstance(policy, Rule):
            action = (rule_action(env) if policy is None
                      else rule_action(env, policy.kappa, policy.entry_speed_m_s))
        else:
            action = np.asarray(policy.predict(obs, deterministic=True)[0])
        tau, kappa = env.intent(action)
        q = None if policy is None or isinstance(policy, Rule) else critic_values(policy, env, obs, action)
        descent = bool(obs[19] > 0.5)  # descent flag of the state the action was taken in
        obs, reward, terminated, truncated, info = env.step(action)
        x = info["execution"]
        log.append({"tau_request_s": tau, "kappa": kappa, "status": x["plan_status"], "slip_s": x["slip_s"],
                    "tau_effective_s": x["tau_effective_s"], "request_kind": x.get("request_kind"),
                    "trigger": x.get("new_plan_trigger"), "deadline": x.get("deadline"),
                    "deviation_before_m": x.get("deviation_before_m"), "a": [float(v) for v in action],
                    "descent": descent, "r": float(reward), "q": q,
                    "entry_speed_request_m_s": x.get("entry_speed_request_m_s"),
                    "entry_speed_applied": x.get("entry_speed_applied"),
                    "entry_speed_plan_m_s": x.get("entry_speed_plan_m_s")})
        ret += gamma**k * reward
        k += 1
        if terminated or truncated:
            break
    statuses = [s["status"] for s in log]
    return {
        **info["episode_summary"], "opening": opening, "discounted_return": ret,
        "numerical_error": info.get("numerical_error"),
        "first_tau_request_s": log[0]["tau_request_s"], "kappa_mean": float(np.mean([s["kappa"] for s in log])),
        "new_plans": sum(s.startswith(("optimal", "recovery")) for s in statuses),
        "kept": sum(s.startswith("kept") for s in statuses),
        "holds": sum(s.startswith("hold") for s in statuses), "slips": sum(s["slip_s"] > 0 for s in log),
        "first_tau_effective_s": next((s["tau_effective_s"] for s in log if s["status"].startswith(("optimal", "recovery"))), None),
        **{f"req_{k}": sum(s["request_kind"] == k for s in log) for k in ("keep", "later", "earlier", "no_plan")},
        **{f"replan_{k}": sum(s["trigger"] == k for s in log) for k in ("no_plan", "deviation", "deadline", "request")},
        "deadline_clamped": sum(s["deadline"] == "clamped" for s in log),
        "entry_speed_applied": [s["entry_speed_request_m_s"] for s in log if s.get("entry_speed_applied")],
        "decision_log": log,
    }


CHECKPOINTS = (0, 2_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000)


def _write_row(output_dir: Path, opening: int, row: dict) -> None:
    (output_dir / f"{opening}.json").write_text(json.dumps(row, default=float))
    print(opening, row["event"], round(row["time_s"], 1), round(row["delta_v_m_s"], 2),
          round(row["discounted_return"], 2), flush=True)


def _load_policy(policy_arg: str):
    rule = parse_rule(policy_arg)
    if rule is not None:
        return rule
    from stable_baselines3 import SAC

    from train.t1_m1 import M1SAC

    try:
        return M1SAC.load(policy_arg, device="cpu")
    except Exception:  # noqa: BLE001 -- an M0 checkpoint
        return SAC.load(policy_arg, device="cpu")


def _evaluate_one(policy_arg: str, opening: int, output_dir: str) -> None:
    import torch

    torch.set_num_threads(1)
    _write_row(Path(output_dir), opening, run_episode(T1GStarEnv(), opening, _load_policy(policy_arg)))


def evaluate(policy_arg: str, openings: Path, output_dir: Path, workers: int = 1) -> dict:
    return main([f"--policy={policy_arg}", f"--openings={openings}", f"--output-dir={output_dir}",
                 f"--workers={workers}"])


def plateau_reached(summaries: dict[int, dict], minimum: int = 20_000, margin: float = 2.0,
                    patience: int = 2) -> bool:
    """Pre-registered stop rule: after ``minimum`` decisions, a mark improves when its mean
    discounted return exceeds the best of all earlier marks by more than ``margin`` or its
    captures exceed the best earlier count; ``patience`` non-improving marks in a row stop."""
    marks = sorted(summaries)
    misses = 0
    for i, n in enumerate(marks):
        if n <= minimum or i == 0:
            continue
        earlier = [summaries[m] for m in marks[:i]]
        better = (summaries[n]["mean_discounted_return"] > max(e["mean_discounted_return"] for e in earlier) + margin
                  or summaries[n]["clean_capture"] > max(e["clean_capture"] for e in earlier))
        misses = 0 if better else misses + 1
    return misses >= patience


def watch(run_dir: Path, openings: Path, poll_s: float = 60.0, checkpoints=CHECKPOINTS, workers: int = 1,
          stop_on_plateau: bool = False) -> None:
    import time

    CHECKPOINTS = tuple(checkpoints)  # noqa: N806 -- the marks this watcher follows
    done = set()
    summaries: dict[int, dict] = {}
    while len(done) < len(CHECKPOINTS):
        if (run_dir / "finished.json").exists() and not any(
                (run_dir / "checkpoints").glob(f"*_{n}_decisions.zip") for n in CHECKPOINTS if n not in done):
            print("run finished; remaining checkpoints were never written", flush=True)
            return
        if (run_dir / "ABORTED.json").exists():
            print("run aborted:", (run_dir / "ABORTED.json").read_text(), flush=True)
            return
        for n in CHECKPOINTS:
            zips = sorted((run_dir / "checkpoints").glob(f"*_{n}_decisions.zip"))
            if n not in done and zips:
                time.sleep(5.0)  # let the save finish
                summaries[n] = evaluate(str(zips[0]), openings, run_dir / "dev_eval" / str(n), workers)
                done.add(n)
                if stop_on_plateau and plateau_reached(summaries) and not (run_dir / "STOP_REQUESTED").exists():
                    (run_dir / "STOP_REQUESTED").write_text(json.dumps({
                        "rule": "plateau (minimum 20000, margin 2.0, patience 2)", "at_mark": n,
                        "summaries": {m: {k: summaries[m][k] for k in ("clean_capture", "mean_discounted_return")}
                                      for m in sorted(summaries)}}))
                    print(f"plateau reached at {n}: STOP_REQUESTED written", flush=True)
        if len(done) < len(CHECKPOINTS):
            time.sleep(poll_s)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", help="checkpoint .zip, or rule:earliest")
    parser.add_argument("--watch", type=Path, help="training run directory to follow")
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="episodes evaluated in parallel processes")
    parser.add_argument("--first", type=int, default=None, help="only the first n openings of the block")
    parser.add_argument("--checkpoints", default=None, help="with --watch: comma-separated decision marks")
    parser.add_argument("--stop-on-plateau", action="store_true",
                        help="with --watch: write STOP_REQUESTED when plateau_reached (the run order's rule)")
    args = parser.parse_args(argv)
    if args.watch is not None:
        marks = tuple(int(x) for x in args.checkpoints.split(",")) if args.checkpoints else CHECKPOINTS
        watch(args.watch, args.openings, checkpoints=marks, workers=args.workers,
              stop_on_plateau=args.stop_on_plateau)
        return {}
    if args.output_dir is None or args.policy is None:
        parser.error("--policy and --output-dir are required without --watch")
    policy = _load_policy(args.policy) if args.workers <= 1 else None
    seeds = json.loads(args.openings.read_text())["seeds"]
    if args.first is not None:
        seeds = seeds[: args.first]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    todo = [o for o in seeds if not (args.output_dir / f"{o}.json").exists()]
    if args.workers > 1 and todo:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(args.workers, len(todo))) as pool:
            pool.starmap(_evaluate_one, [(args.policy, o, str(args.output_dir)) for o in todo])
    else:
        env = T1GStarEnv()
        for opening in todo:
            _write_row(args.output_dir, opening, run_episode(env, opening, policy))
    rows = [json.loads((args.output_dir / f"{o}.json").read_text()) for o in seeds]
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
