"""Deterministic development evaluation for the T1-L three-arm experiment (plan 2026-10-15).

Policies: a checkpoint (its arm's review period from the run manifest, or ``--review-period``)
or a fixed G* rule ``rule:earliest_k<kappa>_v<entry speed>`` (earliest feasible arrival, i.e.
delay 0, then keep, fixed kappa and entry speed; review period 1). The T1-L review wrapper,
openings ``experiments/t1l_dev_openings_297000.json``, serial or ``--workers`` processes.

Per episode: outcome, time, delta-v, the discounted return (0.999 per 2 s step), violations,
and per step: review or continue, executed action, the anchor (earliest feasible tau, delay,
request), the plan status / trigger / deadline, entry speed, kappa, reward and, at reviews of
a checkpoint, min_j Q of the taken action and of its one-factor neighbours (with a live plan:
keep / -15 s / +15 s; without one: delay -15 s / +15 s).

    python -B -m experiments.t1_l_eval --policy rule:earliest_k1.0_v0.02 --output-dir <dir> --workers 4
    python -B -m experiments.t1_l_eval --watch logs/t1_l_option_20261015/n0_s0 --workers 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from env.t1_l_task import task_env_config
from env.t1_task_review import T1LReviewEnv

OPENINGS = Path("experiments/t1l_dev_openings_297000.json")
CHECKPOINTS = (0, 2_000, 5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000)
RULE_GRID = tuple(f"rule:earliest_k{k}_v{v}" for k in ("0.7", "1.0") for v in ("0.02", "0.06", "0.10"))


class Rule:
    def __init__(self, kappa: float, entry_speed_m_s: float) -> None:
        self.kappa, self.entry_speed_m_s = float(kappa), float(entry_speed_m_s)


def parse_rule(policy: str) -> Rule | None:
    if not policy.startswith("rule:earliest_k"):
        return None
    kappa, speed = policy[len("rule:earliest_k"):].split("_v")
    return Rule(float(kappa), float(speed))


def rule_action(env: T1LReviewEnv, rule: Rule) -> np.ndarray:
    c = env.config
    a1 = 2.0 * (rule.kappa - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
    a2 = (rule.entry_speed_m_s - c.entry_speed_nominal_m_s) / c.entry_speed_span_m_s
    live = env.generator.remaining_s(env.env) > 0.0
    return np.array([0.0 if live else -1.0, a1, a2])  # no plan: delay 0 = earliest feasible


def critic_values(model, env: T1LReviewEnv, obs: np.ndarray, action: np.ndarray) -> dict | None:
    critic = getattr(getattr(model, "policy", None), "critic", None)
    if critic is None:
        return None
    import torch as th

    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    actions = {"taken": a}
    if env.generator.remaining_s(env.env) > 0.0:
        actions |= {"keep": np.array([0.0, a[1], a[2]]),
                    "earlier15": np.array([env.action_for_shift(-15.0), a[1], a[2]]),
                    "later15": np.array([env.action_for_shift(15.0), a[1], a[2]])}
    elif not env.generator._descending(env.env, env.env.metrics()):
        actions |= {"delay_minus15": np.array([max(a[0] - 0.5, -1.0), a[1], a[2]]),
                    "delay_plus15": np.array([min(a[0] + 0.5, 1.0), a[1], a[2]])}
    with th.no_grad():
        o = th.as_tensor(np.repeat(obs[None].astype(np.float32), len(actions), axis=0))
        q = th.min(th.cat(critic(o, th.as_tensor(np.stack(list(actions.values())).astype(np.float32))), dim=1), dim=1)[0]
    return {k: float(v) for k, v in zip(actions, q.numpy())}


def run_episode(env: T1LReviewEnv, opening: int, policy) -> dict:
    obs, _ = env.reset(options={"opening": opening})
    gamma = env.env.config.reward.gamma
    ret, k, log = 0.0, 0, []
    while True:
        due = bool(env._due)
        if isinstance(policy, Rule):
            action = rule_action(env, policy)
        else:
            action = np.asarray(policy.predict(obs, deterministic=True)[0])
        q = critic_values(policy, env, obs, action) if due and not isinstance(policy, Rule) else None
        descent = bool(env.generator._descending(env.env, env.env.metrics()))
        obs, reward, terminated, truncated, info = env.step(action)
        x, rv = info["execution"], info["review"]
        log.append({"t": 2.0 * k, "reviewed": rv["reviewed"], "gap": rv["gap_steps"], "plan_event": rv["plan_event"],
                    "a": [float(v) for v in info["executed_action"]],
                    "tau_min_feasible_s": rv.get("tau_min_feasible_s"), "delay_s": rv.get("delay_s"),
                    "tau_request_s": x["tau_request_s"], "tau_effective_s": x["tau_effective_s"],
                    "status": x["plan_status"], "slip_s": x["slip_s"], "request_kind": x["request_kind"],
                    "trigger": x["new_plan_trigger"], "deadline": x["deadline"],
                    "deviation_before_m": x["deviation_before_m"], "kappa": info["kappa"],
                    "entry_speed_request_m_s": x["entry_speed_request_m_s"],
                    "entry_speed_applied": x["entry_speed_applied"], "entry_speed_plan_m_s": x["entry_speed_plan_m_s"],
                    "descent": descent, "r": float(reward), "q": q})
        ret += gamma ** k * reward
        k += 1
        if terminated or truncated:
            break
    statuses = [s["status"] for s in log]
    reviews = [s for s in log if s["reviewed"]]
    first = log[0]
    return {
        **info["episode_summary"], "opening": opening, "discounted_return": ret,
        "numerical_error": info.get("numerical_error"),
        "reviews": len(reviews), "steps": len(log), "plan_events": sum(s["plan_event"] for s in log),
        "first_tau_min_feasible_s": first["tau_min_feasible_s"], "first_delay_s": first["delay_s"],
        "first_tau_effective_s": first["tau_effective_s"],
        "kappa_mean": float(np.mean([s["kappa"] for s in log])),
        "new_plans": sum(s.startswith(("optimal", "recovery")) for s in statuses),
        "recoveries": sum(s.startswith("recovery") for s in statuses),
        "kept": sum(s.startswith("kept") for s in statuses), "holds": sum(s.startswith("hold") for s in statuses),
        **{f"replan_{n}": sum(s["trigger"] == n for s in log) for n in ("no_plan", "deviation", "deadline", "request")},
        **{f"req_{n}": sum(s["request_kind"] == n for s in reviews) for n in ("keep", "later", "earlier", "no_plan")},
        "entry_speed_applied": [s["entry_speed_request_m_s"] for s in log if s["entry_speed_applied"]],
        "decision_log": log,
    }


def load_policy(policy: str):
    rule = parse_rule(policy)
    if rule is not None:
        return rule
    from train.t1_option_sac import OptionSAC

    manifest = Path(policy).resolve().parent.parent / "manifest.json"
    learner = json.loads(manifest.read_text()).get("learner", {}).get("class", "") if manifest.exists() else ""
    if learner == "train.t1_tcpf.TCPFSAC":
        from train.t1_tcpf import TCPFSAC

        return TCPFSAC.load(policy, device="cpu")
    return OptionSAC.load(policy, device="cpu")


def make_env(task: str, period: int) -> T1LReviewEnv:
    return T1LReviewEnv(env_config=task_env_config(task), review_period_steps=period)


def _evaluate_one(policy: str, period: int, opening: int, output_dir: str, task: str = "t1l") -> None:
    import torch

    torch.set_num_threads(1)
    _write(Path(output_dir), opening, run_episode(make_env(task, period), opening, load_policy(policy)))


def _write(output_dir: Path, opening: int, row: dict) -> None:
    (output_dir / f"{opening}.json").write_text(json.dumps(row, default=float))
    print(opening, row["event"], round(row["time_s"], 1), round(row["delta_v_m_s"], 3),
          round(row["discounted_return"], 2), flush=True)


def summarise(rows: list[dict], policy: str, openings: Path) -> dict:
    out = {
        "policy": policy, "episodes": len(rows),
        "clean_capture": sum(r["event"] == "success" for r in rows),
        "timeout": sum(r["event"] == "timeout" for r in rows), "unsafe": sum(r["event"] == "unsafe" for r in rows),
        "mean_discounted_return": float(np.mean([r["discounted_return"] for r in rows])),
        "mean_time_s_fail_as_300": float(np.mean([r["time_s"] if r["event"] == "success" else 300.0 for r in rows])),
        "mean_delta_v_m_s": float(np.mean([r["delta_v_m_s"] for r in rows])),
        "mean_reviews_per_step": float(np.mean([r["reviews"] / max(r["steps"], 1) for r in rows])),
        "failures": {}, "openings_sha256": hashlib.sha256(openings.read_bytes()).hexdigest(),
    }
    for r in rows:
        if r["event"] != "success":
            for key in ([*r.get("violations", {})] if r["event"] == "unsafe" else [r["event"]]):
                out["failures"][key] = out["failures"].get(key, 0) + 1
    return out


def evaluate(policy: str, period: int, openings: Path, output_dir: Path, workers: int = 1, first: int | None = None,
             task: str = "t1l") -> dict:
    seeds = json.loads(openings.read_text())["seeds"]
    seeds = seeds[:first] if first else seeds
    output_dir.mkdir(parents=True, exist_ok=True)
    todo = [o for o in seeds if not (output_dir / f"{o}.json").exists()]
    if workers > 1 and todo:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            pool.starmap(_evaluate_one, [(policy, period, o, str(output_dir), task) for o in todo])
    elif todo:
        model, env = load_policy(policy), make_env(task, period)
        for o in todo:
            _write(output_dir, o, run_episode(env, o, model))
    rows = [json.loads((output_dir / f"{o}.json").read_text()) for o in seeds]
    summary = summarise(rows, policy, openings) | {"review_period_steps": period, "task": task}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return summary


def watch(run_dir: Path, openings: Path, marks, workers: int, poll_s: float = 60.0) -> None:
    import time

    period = int(json.loads((run_dir / "manifest.json").read_text())["arm_spec"]["review_period_steps"])
    done: set[int] = set()
    while len(done) < len(marks):
        if (run_dir / "ABORTED.json").exists():
            print("run aborted", flush=True)
            return
        for n in marks:
            zips = sorted((run_dir / "checkpoints").glob(f"*_{n}_decisions.zip"))
            if n not in done and zips:
                time.sleep(5.0)
                evaluate(str(zips[0]), period, openings, run_dir / "dev_eval" / str(n), workers)
                done.add(n)
        if (run_dir / "finished.json").exists() and not any(
                (run_dir / "checkpoints").glob(f"*_{n}_decisions.zip") for n in marks if n not in done):
            print("run finished; remaining marks were never written", flush=True)
            return
        if len(done) < len(marks):
            time.sleep(poll_s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy")
    parser.add_argument("--watch", type=Path)
    parser.add_argument("--review-period", type=int, default=None, help="checkpoint policies without --watch")
    parser.add_argument("--openings", type=Path, default=OPENINGS)
    parser.add_argument("--task", default="t1l", choices=("t1l", "t1r"), help="t1r: evaluation-only robustness profile")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--first", type=int, default=None)
    parser.add_argument("--checkpoints", default=None)
    args = parser.parse_args(argv)
    if args.watch is not None:
        marks = tuple(int(x) for x in args.checkpoints.split(",")) if args.checkpoints else CHECKPOINTS
        watch(args.watch, args.openings, marks, args.workers)
        return
    if args.policy is None or args.output_dir is None:
        parser.error("--policy and --output-dir are required without --watch")
    if parse_rule(args.policy) is not None:
        period = 1
    elif args.review_period is not None:
        period = args.review_period
    else:
        parser.error("--review-period is required for a checkpoint outside --watch")
    evaluate(args.policy, period, args.openings, args.output_dir, args.workers, args.first, args.task)


if __name__ == "__main__":
    main()
