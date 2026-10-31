"""Stage B: exact learned -> Pure MPC handoff scan on a frozen V3 policy.

For every seed in the block:

    learned_full   the deterministic policy to the end (Kinf)
    handoff at k   the policy for k decisions, then Pure MPC to the end
                   (``step_with_branch(..., branch="baseline")``: one-way,
                   ``controller.reset()`` at the switch -- the same semantics
                   as M2/M6/M5), for every scanned k

``--scan failures`` scans every decision of the episodes the policy does not
complete cleanly (not completed, or completed with a truth violation)
-- the stage-B gate data; ``--scan successes`` scans its clean completions
(stage B2, accounting for stage C). Episodes of the other kind get their
learned_full outcome recorded and nothing else.

Why snapshots, and why they are exact
-------------------------------------
Re-running the prefix from reset for every k costs O(N^2) decisions per
episode (hundreds of CPU hours on the stage-B block). Instead the learned
episode is run once more and, at each scanned decision boundary, the whole
environment is copied in memory (``copy.deepcopy``) *except* its MPC, which
is swapped for a second controller of the same configuration and reset
before the rollout. A handoff resets the controller anyway, so the copy
starts from exactly the state "the policy for k decisions, then Pure MPC from
scratch" starts from. This is not trusted, it is checked on every scanned
episode: ``--verify-prefix`` re-runs the named k from reset along the
original path (``experiments.v3_common.run_episode`` with
``learned_then_baseline(k)``) and requires bitwise-identical task rewards,
survival time and outcome, and the two learned passes must be bitwise
identical too. Any mismatch raises; nothing is written for that seed.

Output: one JSON per seed in ``--output-dir`` (``seed_<seed>.json``, written
atomically); a seed whose file exists, or that another process has claimed
(``seed_<seed>.lock``), is skipped, so a run can be resumed and several
processes can share one seed list. The reading rules are in
``docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md``;
``experiments.v3_handoff_readout`` applies them.

    python -B -m experiments.v3_handoff_scan --run-dir logs/v3e_262420 --seeds 262000-262047 --scan failures --output-dir eval/v3e/stage_b/262420
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from env.hybrid_env import PrecaptureHybridEnv
from experiments.evaluate_hybrid_policy import VIOLATION_STEP_KEYS
from experiments.v3_common import (
    FAILURE_KEYS,
    learned_then_baseline,
    load_manifest,
    load_policy,
    make_env_for_run,
    parse_seed_range,
    run_episode,
)
from train.mainline import mainline_v3e_configs
from train.train_hybrid import _code_provenance
from train.v3_values import sha256_file

SCHEMA = "v3_handoff_scan/1"


class ImpulseMeter:
    """Accumulates |F| dt of every commanded wrench (the evaluator's Delta-v basis)."""

    def __init__(self, controller: Any, dt_s: float) -> None:
        self.force_impulse_n_s = 0.0
        original = controller.command

        def metered(*args: Any, **kwargs: Any):
            wrench, diagnostics = original(*args, **kwargs)
            self.force_impulse_n_s += float(np.linalg.norm(np.asarray(wrench)[3:])) * dt_s
            return wrench, diagnostics

        controller.command = metered  # instance attribute; the class is untouched


def _outcome(env: PrecaptureHybridEnv, info: dict[str, Any]) -> dict[str, Any]:
    completed = bool(info.get("completed", False))
    zero_violation = all(int(info[key]) == 0 for key in VIOLATION_STEP_KEYS if key in info)
    return {
        "completed": completed,
        "zero_violation": zero_violation,
        "clean_completion": completed and zero_violation,
        "failure": [key for key in FAILURE_KEYS if info.get(key)],
        "survival_s": float(env.env.time_seconds),
        "qp_zero_fallbacks": int(info.get("hybrid_v3_episode_qp_zero_fallbacks", -1)),
    }


def _learned_pass(
    env: PrecaptureHybridEnv,
    policy: Any,
    meter: ImpulseMeter,
    seed: int,
    *,
    on_boundary: Any = None,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    """One learned_full episode. ``on_boundary(k, env, prefix)`` runs before decision k."""

    observation, _ = env.reset(seed=int(seed))
    meter.force_impulse_n_s = 0.0
    task_rewards: list[float] = []
    info: dict[str, Any] = {}
    decision = 0
    while max_decisions is None or decision < max_decisions:
        if on_boundary is not None:
            on_boundary(
                decision,
                env,
                {
                    "force_impulse_n_s": meter.force_impulse_n_s,
                    "task_utility": float(np.sum(task_rewards)),
                },
            )
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch="learned"
        )
        task_rewards.append(float(info["hybrid_integrated_reward_without_shaping"]))
        decision += 1
        if terminated or truncated:
            break
    mass = float(env.env.chaser_parameters.mass)
    return {
        **_outcome(env, info),
        "decisions": decision,
        "task_rewards": task_rewards,
        "task_utility": float(np.sum(task_rewards)),
        "equivalent_delta_v_m_s": meter.force_impulse_n_s / mass,
    }


def _handoff_from(
    env: PrecaptureHybridEnv,
    k: int,
    prefix: dict[str, float],
    controller_b: Any,
    meter_b: ImpulseMeter,
    *,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    """Pure MPC to the end from a copy of ``env`` taken at decision boundary k."""

    snapshot = copy.deepcopy(env, memo={id(env.controller): controller_b})
    controller_b.reset()
    meter_b.force_impulse_n_s = 0.0
    zero = np.zeros(snapshot.action_space.shape, dtype=np.float64)
    time_at_handoff_s = float(snapshot.env.time_seconds)
    task_rewards: list[float] = []
    info: dict[str, Any] = {}
    decision = k
    try:
        while max_decisions is None or decision < max_decisions:
            _, _, terminated, truncated, info = snapshot.step_with_branch(zero, branch="baseline")
            task_rewards.append(float(info["hybrid_integrated_reward_without_shaping"]))
            decision += 1
            if terminated or truncated:
                break
        mass = float(snapshot.env.chaser_parameters.mass)
        return {
            "k": int(k),
            "time_at_handoff_s": time_at_handoff_s,
            **_outcome(snapshot, info),
            "decisions": decision,
            "task_rewards_after_handoff": task_rewards,
            "task_utility_to_go": float(np.sum(task_rewards)),
            "task_utility": prefix["task_utility"] + float(np.sum(task_rewards)),
            "equivalent_delta_v_m_s": (
                prefix["force_impulse_n_s"] + meter_b.force_impulse_n_s
            ) / mass,
        }
    finally:
        # The copy shares ``controller_b``; closing the copy must not touch it.
        snapshot.controller = None  # type: ignore[assignment]


def _prefix_recompute(
    env: PrecaptureHybridEnv, policy: Any, seed: int, k: int, max_decisions: int | None
) -> dict[str, Any]:
    """The original M6 path: from reset, policy for k decisions, then Pure MPC."""

    if max_decisions is None:
        episode = run_episode(env, policy, seed, learned_then_baseline(k))
        return {
            "task_rewards_after_handoff": [float(v) for v in episode["task_rewards"][k:]],
            "completed": episode["completed"],
            "failure": episode["failure"],
            "survival_s": float(env.env.time_seconds),
        }
    # Test-only truncated variant of the same loop.
    observation, _ = env.reset(seed=int(seed))
    zero = np.zeros(env.action_space.shape, dtype=np.float64)
    rewards: list[float] = []
    info: dict[str, Any] = {}
    for decision in range(max_decisions):
        if decision < k:
            action, _ = policy.predict(observation, deterministic=True)
            branch = "learned"
        else:
            action, branch = zero, "baseline"
        observation, _, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch=branch
        )
        rewards.append(float(info["hybrid_integrated_reward_without_shaping"]))
        if terminated or truncated:
            break
    return {
        "task_rewards_after_handoff": rewards[k:],
        "completed": bool(info.get("completed", False)),
        "failure": [key for key in FAILURE_KEYS if info.get(key)],
        "survival_s": float(env.env.time_seconds),
    }


def _verify_ks(spec: str, decisions: int, scanned: list[int]) -> list[int]:
    if not spec:
        return []
    chosen: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if token == "first":
            chosen.add(scanned[0])
        elif token == "middle":
            chosen.add(scanned[len(scanned) // 2])
        elif token == "last":
            chosen.add(scanned[-1])
        elif token:
            chosen.add(int(token))
    return sorted(k for k in chosen if 0 <= k < decisions)


def scan_seed(
    env: PrecaptureHybridEnv,
    env_verify: PrecaptureHybridEnv,
    controller_b: Any,
    meter: ImpulseMeter,
    meter_b: ImpulseMeter,
    policy: Any,
    seed: int,
    *,
    scan: str,
    stride: int,
    verify_prefix: str,
    max_decisions: int | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    first = _learned_pass(env, policy, meter, seed, max_decisions=max_decisions)
    # A failure is anything but a clean completion: a completion with a truth
    # violation is not a success the coordination could be asked to keep.
    wanted = {
        "failures": not first["clean_completion"],
        "successes": first["clean_completion"],
        "all": True,
    }[scan]
    result: dict[str, Any] = {
        "seed": int(seed),
        "learned_full": first,
        "scanned": bool(wanted),
        "stride": int(stride),
        "handoffs": [],
        "verified_prefix_ks": [],
    }
    if not wanted:
        result["wall_clock_s"] = perf_counter() - started
        return result

    scanned = list(range(0, first["decisions"], stride))
    handoffs: list[dict[str, Any]] = []

    def on_boundary(k: int, live_env: PrecaptureHybridEnv, prefix: dict[str, float]) -> None:
        if k % stride == 0:
            handoffs.append(
                _handoff_from(
                    live_env, k, prefix, controller_b, meter_b, max_decisions=max_decisions
                )
            )

    second = _learned_pass(
        env, policy, meter, seed, on_boundary=on_boundary, max_decisions=max_decisions
    )
    for key in ("task_rewards", "survival_s", "completed", "failure", "decisions"):
        if second[key] != first[key]:
            raise RuntimeError(f"seed {seed}: the two learned passes differ in {key}")
    if [row["k"] for row in handoffs] != scanned:
        raise RuntimeError(f"seed {seed}: scanned decisions do not match the learned episode")

    by_k = {row["k"]: row for row in handoffs}
    for k in _verify_ks(verify_prefix, first["decisions"], scanned):
        reference = _prefix_recompute(env_verify, policy, seed, k, max_decisions)
        row = by_k[k]
        for key in ("task_rewards_after_handoff", "completed", "failure", "survival_s"):
            if reference[key] != row[key]:
                raise RuntimeError(
                    f"seed {seed}, k={k}: snapshot handoff differs from the "
                    f"prefix-recompute path in {key}"
                )
        result["verified_prefix_ks"].append(k)

    utility_to_go_learned = np.cumsum(first["task_rewards"][::-1])[::-1]
    for row in handoffs:
        row["learned_task_utility_to_go"] = float(utility_to_go_learned[row["k"]])
    result["handoffs"] = handoffs
    result["wall_clock_s"] = perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="final_model.zip")
    parser.add_argument("--seeds", required=True, help="'262000-262047' or a comma list")
    parser.add_argument("--scan", choices=("failures", "successes", "all"), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--verify-prefix",
        default="first,middle,last",
        help="scanned k re-run from reset along the original path and compared bitwise",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=None, help="tests only")
    parser.add_argument("--allow-incomplete-run", action="store_true", help="smoke tests only")
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    manifest = load_manifest(args.run_dir, require_completed=not args.allow_incomplete_run)
    policy = load_policy(args.run_dir, args.model_name)
    env = make_env_for_run(manifest)
    mainline_environment, mainline_hybrid = mainline_v3e_configs()
    if asdict(env.hybrid_config) != asdict(mainline_hybrid) or asdict(
        env.environment_config
    ) != asdict(mainline_environment):
        raise ValueError("this run is not the canonical V3e mainline configuration")
    env_b = make_env_for_run(manifest)
    env_verify = make_env_for_run(manifest)
    dt_s = float(env.environment_config.dt_s)
    meter = ImpulseMeter(env.controller, dt_s)
    meter_b = ImpulseMeter(env_b.controller, dt_s)
    ImpulseMeter(env_verify.controller, dt_s)
    provenance = {
        "schema": SCHEMA,
        "run_dir": str(args.run_dir),
        "run_name": manifest.get("run_name"),
        "training_code_commit": manifest.get("code_commit"),
        "model_name": args.model_name,
        "model_sha256": sha256_file(Path(args.run_dir) / args.model_name),
        "manifest_sha256": sha256_file(Path(args.run_dir) / "manifest.json"),
        **_code_provenance(),
        "scan": args.scan,
        "stride": args.stride,
        "verify_prefix": args.verify_prefix,
        "max_decisions": args.max_decisions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in parse_seed_range(args.seeds):
            path = args.output_dir / f"seed_{seed}.json"
            if path.exists():
                print(f"skip {path} (exists)", flush=True)
                continue
            # Several processes may share one seed list; the first to create
            # the lock owns the seed. A lock left by a crashed process must be
            # deleted by hand (only when no scan process is running).
            lock = path.with_suffix(".lock")
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                print(f"skip {path} (claimed by another process)", flush=True)
                continue
            result = scan_seed(
                env,
                env_verify,
                env_b.controller,
                meter,
                meter_b,
                policy,
                seed,
                scan=args.scan,
                stride=args.stride,
                verify_prefix=args.verify_prefix,
                max_decisions=args.max_decisions,
            )
            result.update(provenance)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=1))
            os.replace(temporary, path)
            lock.unlink()
            rescued = sum(row["clean_completion"] for row in result["handoffs"])
            print(
                f"seed {seed}: learned completed={result['learned_full']['completed']} "
                f"decisions={result['learned_full']['decisions']} scanned={result['scanned']} "
                f"clean handoffs={rescued}/{len(result['handoffs'])} "
                f"verified={result['verified_prefix_ks']} wall={result['wall_clock_s']:.0f}s",
                flush=True,
            )
    finally:
        env.close()
        env_b.close()
        env_verify.close()


if __name__ == "__main__":
    main()
