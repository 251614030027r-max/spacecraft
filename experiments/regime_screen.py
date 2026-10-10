"""Final-regime screening without training (``docs/REGIME_SCREEN_20261006.md``).

    evaluate  one row on one regime: ``pure`` (Pure MPC from reset) or
              ``nominal`` (the scripted smooth nominal on the learned branch,
              full-rate action +1 -- the formal evaluator's ``v3_nominal``)
    readout   per-regime counts and the preregistered selection

Selection never reads a learned or stopping result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from env.hybrid_env import PrecaptureHybridEnv
from experiments.v3_common import parse_seed_range
from experiments.v3_handoff_scan import ImpulseMeter, _outcome
from experiments.v3_stage_c import _scans, _write_json
from train.regimes import REGIMES, STAGE1, regime_environment
from train.stopping import stopping_configs
from train.train_hybrid import _code_provenance

SCREEN_BLOCK = "269000-269047"
#: Stage-1 pass: Pure clean completions in [29, 36] of 48 (60-75%) ...
PURE_CLEAN_MIN, PURE_CLEAN_MAX = 29, 36
#: ... at least two thirds of Pure's failures are time failures (timing, not
#: physical reachability) ...
MIN_TIME_FAILURE_SHARE = 2.0 / 3.0
#: ... and at most two Pure episodes with a truth violation.
MAX_PURE_VIOLATION_EPISODES = 2
#: Stage-2: the interface itself must reach at least this many Pure failures.
MIN_NOMINAL_RESCUE = 3


def regime_env(name: str) -> PrecaptureHybridEnv:
    environment, hybrid = stopping_configs()
    return PrecaptureHybridEnv(regime_environment(environment, REGIMES[name]), hybrid)


def run_row(env: PrecaptureHybridEnv, seed: int, row: str, meter: ImpulseMeter,
            max_decisions: int | None = None) -> dict[str, Any]:
    env.reset(seed=int(seed))
    meter.force_impulse_n_s = 0.0
    action = np.zeros(env.action_space.shape) if row == "pure" else np.ones(env.action_space.shape)
    branch = "baseline" if row == "pure" else "learned"
    info: dict[str, Any] = {}
    decision = 0
    while max_decisions is None or decision < max_decisions:
        _, _, terminated, truncated, info = env.step_with_branch(action, branch=branch)
        decision += 1
        if terminated or truncated:
            break
    return {
        "seed": int(seed), "row": row, **_outcome(env, info), "decisions": decision,
        "equivalent_delta_v_m_s": meter.force_impulse_n_s / float(env.env.chaser_parameters.mass),
    }


def cmd_evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    env = regime_env(args.regime)
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    regime = REGIMES[args.regime]
    provenance = {"regime": args.regime, "tumble_deg_s": regime.tumble_deg_s, "range_min_m": regime.range_min_m,
                  "range_max_m": regime.range_max_m, "speed_max_m_s": regime.speed_max_m_s,
                  "max_decisions": args.max_decisions, **_code_provenance()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in parse_seed_range(args.seeds):
            path = args.output_dir / f"seed_{seed}.json"
            if path.exists():
                continue
            lock = path.with_suffix(".lock")
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            started = perf_counter()
            result = run_row(env, seed, args.row, meter, args.max_decisions)
            result.update(provenance, wall_clock_s=perf_counter() - started)
            _write_json(path, result)
            lock.unlink()
            print(f"{args.regime} {args.row} seed {seed}: done", flush=True)
    finally:
        env.close()


def summarize(rows: dict[int, dict]) -> dict[str, Any]:
    failures = [r for r in rows.values() if not r["clean_completion"]]
    kinds: dict[str, int] = {}
    for r in failures:
        key = "+".join(r["failure"]) or ("violation_only" if not r["zero_violation"] else "other")
        kinds[key] = kinds.get(key, 0) + 1
    clean_times = [r["survival_s"] for r in rows.values() if r["clean_completion"]]
    return {
        "clean": sum(bool(r["clean_completion"]) for r in rows.values()),
        "violation_episodes": sum(not r["zero_violation"] for r in rows.values()),
        "failure_kinds": kinds,
        "time_failure_share": (sum("time_failure" in r["failure"] for r in failures) / len(failures)) if failures else None,
        "qp_zero_fallbacks_total": sum(max(0, r["qp_zero_fallbacks"]) for r in rows.values()),
        "clean_median_time_s": float(np.median(clean_times)) if clean_times else None,
    }


def stage1_pass(summary: dict[str, Any]) -> bool:
    share = summary["time_failure_share"]
    return (PURE_CLEAN_MIN <= summary["clean"] <= PURE_CLEAN_MAX
            and share is not None and share >= MIN_TIME_FAILURE_SHARE
            and summary["violation_episodes"] <= MAX_PURE_VIOLATION_EPISODES)


def select(cells: dict[str, dict[str, Any]]) -> str | None:
    """The passing cell closest to the current mainline (smaller tumble, then
    smaller range_min) among those where the scripted nominal completes
    cleanly on at least MIN_NOMINAL_RESCUE openings that Pure fails."""

    ready = [n for n, c in cells.items()
             if c["passes"] and (c.get("pure_fail_nominal_clean") or 0) >= MIN_NOMINAL_RESCUE]
    if not ready:
        return None
    return sorted(ready, key=lambda n: (REGIMES[n].tumble_deg_s, REGIMES[n].range_min_m))[0]


def cmd_readout(args: argparse.Namespace) -> None:
    expected = parse_seed_range(args.seeds)
    root = Path(args.root)
    cells: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    commits: set = set()
    for name in REGIMES:
        pure = _scans(str(root / name / "pure")) if (root / name / "pure").exists() else {}
        if not pure:
            continue
        nominal = _scans(str(root / name / "nominal")) if (root / name / "nominal").exists() else {}
        for label, rows in (("pure", pure), ("nominal", nominal)):
            if rows and sorted(rows) != expected:
                problems.append(f"{name} {label}: {len(rows)}/{len(expected)} openings")
            for r in rows.values():
                commits.add(r.get("code_commit"))
                if r.get("code_dirty") or r.get("max_decisions") is not None or r.get("regime") != name:
                    problems.append(f"{name} {label}: dirty, truncated or wrong regime")
                    break
        cell = {"pure": summarize(pure)}
        cell["passes"] = stage1_pass(cell["pure"])
        if nominal and sorted(nominal) == expected:
            cell["nominal"] = summarize(nominal)
            cell["pure_fail_nominal_clean"] = sum(
                (not pure[s]["clean_completion"]) and nominal[s]["clean_completion"] for s in expected)
            cell["union_clean"] = sum(pure[s]["clean_completion"] or nominal[s]["clean_completion"] for s in expected)
        cells[name] = cell
    if len(commits) > 1:
        problems.append(f"{len(commits)} code commits across files")
    result = {
        "preregistration": "docs/REGIME_SCREEN_20261006.md", "seeds": args.seeds, "problems": problems,
        "constants": {"pure_clean_range": [PURE_CLEAN_MIN, PURE_CLEAN_MAX],
                      "min_time_failure_share": MIN_TIME_FAILURE_SHARE,
                      "max_pure_violation_episodes": MAX_PURE_VIOLATION_EPISODES,
                      "min_nominal_rescue": MIN_NOMINAL_RESCUE},
        "cells": cells,
        "stage1_passing": [n for n, c in cells.items() if c["passes"]],
        "selected": None if problems else select(cells),
    }
    _write_json(Path(args.output), result)
    print(json.dumps({"problems": problems, "selected": result["selected"], "passing": result["stage1_passing"],
                      "cells": {n: [c["pure"]["clean"], c["pure"]["time_failure_share"], c["pure"]["violation_episodes"],
                                    c.get("pure_fail_nominal_clean")] for n, c in cells.items()}}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--regime", choices=sorted(REGIMES), required=True)
    p.add_argument("--row", choices=("pure", "nominal"), required=True)
    p.add_argument("--seeds", default=SCREEN_BLOCK)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=None, help="tests only")
    p.set_defaults(func=cmd_evaluate)
    p = sub.add_parser("readout")
    p.add_argument("--root", required=True, help="directory holding <regime>/<row>/seed_*.json")
    p.add_argument("--seeds", default=SCREEN_BLOCK)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_readout)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
