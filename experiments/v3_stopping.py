"""Evaluate the learned stopping option; dev-run structural checks; formal readout.

Reading rules: ``docs/STOPPING_METHOD_PREREGISTRATION_20261002.md``.

    evaluate  one row per call on a seed block, one JSON per opening:
              ``pure``     Pure MPC from reset (model-independent);
              ``stopping`` the method: learned task policy, hand off to Pure
                           MPC at the first decision with beta(s) >= 0.5;
              ``learned``  the same task policy with the handoff disabled
                           (reported, never gated)
    devcheck  the three structural checks of the short development run
    readout   per-model counts and the preregistered verdict
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from env.hybrid_env import PrecaptureHybridEnv
from experiments.v3_common import parse_seed_range
from experiments.v3_handoff_scan import ImpulseMeter, _outcome
from experiments.v3_stage_c import _pairs, _scans, _write_json, model_verdict
from train.stopping import StoppingSAC, stopping_configs
from train.train_hybrid import _code_provenance
from train.train_stopping import METHOD
from train.v3_values import sha256_file

DEV_BLOCK = "266000-266047"
FORMAL_BLOCK = "267000-267047"

#: Development-run structural checks (not success rates; nothing is tuned on them).
DEV_MAX_SAME_CORNER = 43        # D1: at most 43/48 openings hand off at k=0, and at most 43/48 never
DEV_MIN_LONG_LEARNED_EVAL = 5   # D2b: >= 5/48 openings fly >= 30 learned decisions deterministically
DEV_LONG_LEARNED = 30           # 30 decisions = 60 s
DEV_TRAIN_TAIL = 0.25           # D2a: last quarter of training episodes ...
DEV_MIN_LONG_LEARNED_TRAIN = 0.25  # ... at least a quarter fly >= 30 learned decisions
DEV_MIN_DISTINCT_K = 3          # D3a: handoffs at k >= 1 occur at >= 3 distinct decisions
DEV_MIN_GAP_RANGE = 1.0         # D3b: median within-episode range of Q_H - Q_C >= 1 reward unit

MIN_PASSING_MODELS = 2


def env_for_stopping_run(run_dir: Path, allow_incomplete: bool) -> tuple[dict[str, Any], PrecaptureHybridEnv]:
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    if manifest.get("method") != METHOD:
        raise ValueError(f"{run_dir} is not a {METHOD} run")
    if not allow_incomplete and manifest.get("status") != "completed":
        raise ValueError(f"run {run_dir} is not completed (status={manifest.get('status')})")
    environment, hybrid = stopping_configs()
    if asdict(hybrid) != manifest["hybrid"]:
        raise ValueError("this checkout builds a different hybrid config from the run's manifest")
    if json.loads(json.dumps(asdict(environment), default=str)) != json.loads(
        json.dumps(manifest["training_environment"], default=str)
    ):
        raise ValueError("this checkout builds a different environment from the run's manifest")
    env = PrecaptureHybridEnv(environment, hybrid)
    if int(env.observation_space.shape[0]) != int(manifest["observation_dimension"]):
        raise ValueError("observation dimension differs from the run's manifest")
    return manifest, env


def load_stopping_model(run_dir: Path, model_name: str = "final_model.zip") -> StoppingSAC:
    torch.set_num_threads(1)
    return StoppingSAC.load(Path(run_dir) / model_name, device="cpu")


def run_stopping_episode(
    env: PrecaptureHybridEnv,
    model: Any,
    seed: int,
    row: str,
    meter: ImpulseMeter,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    """One deterministic episode of ``pure`` / ``stopping`` / ``learned``."""

    observation, _ = env.reset(seed=int(seed))
    meter.force_impulse_n_s = 0.0
    zero = np.zeros(env.action_space.shape, dtype=np.float64)
    branch = "baseline" if row == "pure" else "learned"
    handoff_k = 0 if row == "pure" else None
    threshold = None if model is None else model.stopping_config.deployment_threshold
    trace: dict[str, list[float]] = {"beta": [], "q_handoff": [], "q_continue": []}
    info: dict[str, Any] = {}
    decision = learned_decisions = 0
    while max_decisions is None or decision < max_decisions:
        if branch == "learned":
            values = model.stopping_values(observation)
            for key in trace:
                trace[key].append(values[key])
            if row == "stopping" and values["beta"] >= threshold:
                branch, handoff_k = "baseline", decision
        if branch == "learned":
            action, _ = model.predict(observation, deterministic=True)
            learned_decisions += 1
        else:
            action = zero
        observation, _, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch=branch
        )
        decision += 1
        if terminated or truncated:
            break
    return {
        "seed": int(seed),
        "row": row,
        **_outcome(env, info),
        "decisions": decision,
        "handoff_k": handoff_k,
        "learned_decisions": learned_decisions,
        "equivalent_delta_v_m_s": meter.force_impulse_n_s / float(env.env.chaser_parameters.mass),
        "trace": trace,
    }


def cmd_evaluate(args: argparse.Namespace) -> None:
    manifest, env = env_for_stopping_run(args.run_dir, args.allow_incomplete_run)
    model = None if args.row == "pure" else load_stopping_model(args.run_dir, args.model_name)
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    provenance = {
        "run_dir": str(args.run_dir), "row": args.row,
        "model_sha256": None if model is None else sha256_file(args.run_dir / args.model_name),
        "max_decisions": args.max_decisions, **_code_provenance(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in parse_seed_range(args.seeds):
            path = args.output_dir / f"seed_{seed}.json"
            if path.exists():
                print(f"skip {path} (exists)", flush=True)
                continue
            lock = path.with_suffix(".lock")
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                print(f"skip {path} (claimed)", flush=True)
                continue
            started = perf_counter()
            result = run_stopping_episode(env, model, seed, args.row, meter, args.max_decisions)
            result.update(provenance, wall_clock_s=perf_counter() - started)
            _write_json(path, result)
            lock.unlink()
            print(f"seed {seed}: done", flush=True)
    finally:
        env.close()


# -- development checks -----------------------------------------------------------------


def read_monitor(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        handle.readline()  # Monitor's JSON header
        return list(csv.DictReader(handle))


def dev_checks(monitor_rows: list[dict[str, str]], episodes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    ks = [e["handoff_k"] for e in episodes.values()]
    at_zero = sum(k == 0 for k in ks)
    never = sum(k is None for k in ks)
    mid = sorted(k for k in ks if k is not None and k >= 1)
    long_eval = sum(e["learned_decisions"] >= DEV_LONG_LEARNED for e in episodes.values())
    tail = monitor_rows[int(len(monitor_rows) * (1.0 - DEV_TRAIN_TAIL)):]
    long_train = (
        float(np.mean([int(float(r["learned_decisions"])) >= DEV_LONG_LEARNED for r in tail])) if tail else 0.0
    )
    ranges = []
    for e in episodes.values():
        gap = np.asarray(e["trace"]["q_handoff"]) - np.asarray(e["trace"]["q_continue"])
        if gap.size >= 2:
            ranges.append(float(gap.max() - gap.min()))
    median_range = float(np.median(ranges)) if ranges else 0.0
    d1 = at_zero <= DEV_MAX_SAME_CORNER and never <= DEV_MAX_SAME_CORNER
    d2 = long_train >= DEV_MIN_LONG_LEARNED_TRAIN and long_eval >= DEV_MIN_LONG_LEARNED_EVAL
    d3 = len(set(mid)) >= DEV_MIN_DISTINCT_K and median_range >= DEV_MIN_GAP_RANGE
    return {
        "openings": n,
        "D1_not_collapsed": {"pass": d1, "handoff_at_k0": at_zero, "never_handoff": never,
                             "mid_episode_handoffs": len(mid)},
        "D2_learned_reaches_mid_late": {"pass": d2, "train_tail_episodes": len(tail),
                                        "train_tail_fraction_long": long_train,
                                        "eval_openings_long": long_eval},
        "D3_state_dependent_stopping_value": {"pass": d3, "distinct_mid_handoff_k": len(set(mid)),
                                              "handoff_k_values": mid,
                                              "median_within_episode_gap_range": median_range},
        "all_pass": d1 and d2 and d3,
        "reported_not_gated": {
            "clean_completions": sum(bool(e["clean_completion"]) for e in episodes.values()),
            "train_episodes": len(monitor_rows),
            "train_handoff_fraction": float(np.mean([r["handoff"] == "True" for r in monitor_rows])) if monitor_rows else None,
        },
    }


def cmd_devcheck(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = parse_seed_range(args.seeds)
    episodes = _scans(args.eval_dir)
    missing = sorted(set(expected) - set(episodes))
    if missing:
        raise RuntimeError(f"missing dev openings {missing}")
    result = {
        "preregistration": "docs/STOPPING_METHOD_PREREGISTRATION_20261002.md",
        "run_dir": str(args.run_dir), "seeds": args.seeds,
        **dev_checks(read_monitor(args.run_dir / "train.monitor.csv"), {s: episodes[s] for s in expected}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v)
                      for k, v in result.items() if k.startswith("D") or k == "all_pass"}, indent=1))


# -- formal readout -----------------------------------------------------------------------


def final_verdict(models: dict[str, dict[str, Any]], fidelity_ok: bool) -> str:
    if not fidelity_ok:
        return "FIDELITY_FAIL"
    passing = sum(bool(v["passes"]) for v in models.values())
    return "METHOD_HOLDS" if passing >= MIN_PASSING_MODELS else "METHOD_DOES_NOT_HOLD"


def cmd_readout(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = parse_seed_range(args.seeds)
    pure = _scans(args.pure)
    stopping = {m: _scans(d) for m, d in _pairs(args.stopping).items()}
    learned = {m: _scans(d) for m, d in _pairs(args.learned).items()}
    problems = []

    def check(rows: dict[int, dict], name: str, row: str) -> None:
        missing = sorted(set(expected) - set(rows))
        if missing:
            problems.append(f"{name}: missing seeds {missing}")
        commits = {r.get("code_commit") for r in rows.values()}
        if len(commits) != 1 or None in commits:
            problems.append(f"{name}: {len(commits)} code commits")
        if any(r.get("code_dirty") for r in rows.values()):
            problems.append(f"{name}: dirty checkout")
        if any(r.get("row") != row or r.get("max_decisions") is not None for r in rows.values()):
            problems.append(f"{name}: wrong row or truncated")

    check(pure, "pure", "pure")
    if set(stopping) != set(learned):
        problems.append("stopping and learned rows name different models")
    for m in stopping:
        check(stopping[m], f"stopping {m}", "stopping")
        check(learned.get(m, {}), f"learned {m}", "learned")
        shas = {r.get("model_sha256") for r in stopping[m].values()} | {r.get("model_sha256") for r in learned.get(m, {}).values()}
        if len(shas) != 1:
            problems.append(f"{m}: more than one model file")
    all_commits = {r.get("code_commit") for rows in [pure, *stopping.values(), *learned.values()] for r in rows.values()}
    if len(all_commits) != 1:
        problems.append(f"{len(all_commits)} code commits across rows")
    models = {} if problems else {
        m: model_verdict({s: pure[s] for s in expected}, {s: learned[m][s] for s in expected},
                         {s: stopping[m][s] for s in expected})
        for m in stopping
    }
    result = {
        "preregistration": "docs/STOPPING_METHOD_PREREGISTRATION_20261002.md",
        "seeds": args.seeds, "problems": problems,
        "constants": {"min_passing_models": MIN_PASSING_MODELS},
        "note": "model_verdict's 'hybrid' is the stopping row; 'learned' is the same policy with handoff disabled",
        "verdict": final_verdict(models, not problems),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"verdict": result["verdict"], "problems": len(problems),
                      "models": {m: {k: v[k] for k in ("clean_completions", "violation_episodes", "passes", "efficiency_acceptable")}
                                 for m, v in models.items()}}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-name", default="final_model.zip")
    p.add_argument("--row", choices=("pure", "stopping", "learned"), required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=None, help="tests only")
    p.add_argument("--allow-incomplete-run", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("devcheck")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--eval-dir", required=True, help="stopping-row JSONs on the dev block")
    p.add_argument("--seeds", default=DEV_BLOCK)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_devcheck)

    p = sub.add_parser("readout")
    p.add_argument("--pure", required=True)
    p.add_argument("--stopping", action="append", required=True, help="MODEL=dir")
    p.add_argument("--learned", action="append", required=True, help="MODEL=dir")
    p.add_argument("--seeds", default=FORMAL_BLOCK)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_readout)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
