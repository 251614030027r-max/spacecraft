"""Real-policy T0b continuity acceptance with legacy 39D checkpoints.

This is the only allowed compatibility path for pre-T0b V3 models: the live
environment remains 42D, while ``policy.predict`` receives a read-only view
with the final applied-direction state removed.  Training and formal
evaluation never use this adapter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=263000)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=35)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--v2-reference-dir",
        type=Path,
        default=Path("eval/adp"),
        help="Directory containing final_v2_nominal_262410..262412.json.",
    )
    return parser.parse_args()


def _v2_fallback_level(directory: Path) -> dict[str, float | int]:
    episodes = steps = qp_failures = zero_fallbacks = 0
    for seed in (262410, 262411, 262412):
        path = directory / f"final_v2_nominal_{seed}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            episodes += 1
            steps += int(record["steps"])
            qp_failures += int(record["qp_infeasible_steps_total"])
            zero_fallbacks += int(record["zero_fallback_steps_total"])
    return {
        "episodes": episodes,
        "control_steps": steps,
        "qp_infeasible_steps": qp_failures,
        "zero_fallback_steps": zero_fallbacks,
        "qp_infeasible_fraction": qp_failures / steps if steps else float("nan"),
        "zero_fallback_fraction": zero_fallbacks / steps if steps else float("nan"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    policy = SAC.load(args.model, device="cpu")
    expected = int(np.prod(policy.observation_space.shape))
    if expected != 39:
        raise ValueError(f"T0b legacy adapter requires a 39D checkpoint, got {expected}D")
    policy.set_random_seed(args.model_seed)
    environment = replace(
        precapture_adaptive_capture_environment_config(),
        cache_target_trajectory=False,
    )
    records: list[dict[str, Any]] = []
    all_jumps: list[float] = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        env = PrecaptureHybridEnv(
            environment_config=environment,
            hybrid_config=PrecaptureHybridConfig(
                horizon_steps=args.horizon,
                waypoint_parametrization="task_state_v3",
                runtime_diagnostics=False,
                include_target_phase_and_time_observation=True,
                include_execution_feedback_observation=True,
                include_staging_direction_observation=True,
            ),
        )
        observation, _ = env.reset(seed=seed)
        if observation.shape != (42,):
            raise RuntimeError(f"T0b environment must be 42D, got {observation.shape}")
        terminated = truncated = False
        jumps: list[float] = []
        limits: list[float] = []
        lags: list[float] = []
        qp_failures = zero_fallbacks = control_steps = 0
        finite = True
        final_info: dict[str, Any] = {}
        try:
            while not (terminated or truncated):
                # Deliberately local to this diagnostic.  No wrapper or env flag
                # can silently permit a 39D model on the 42D production path.
                legacy_observation = observation[:-3]
                action, _ = policy.predict(legacy_observation, deterministic=False)
                observation, reward, terminated, truncated, final_info = env.step(
                    np.asarray(action, dtype=np.float64)
                )
                jump = float(final_info["reference_jump_target_m"])
                limit = float(final_info["reference_jump_target_limit_m"])
                lag = float(final_info["hybrid_v3_reference_direction_lag_rad"])
                jumps.append(jump)
                limits.append(limit)
                lags.append(lag)
                decision_steps = int(final_info["hybrid_control_steps"])
                valid_steps = int(final_info["hybrid_feedback_valid_solve_steps"])
                control_steps += decision_steps
                qp_failures += decision_steps - valid_steps
                zero_fallbacks += int(final_info["hybrid_qp_zero_fallbacks"])
                finite &= bool(
                    np.all(np.isfinite(observation))
                    and np.isfinite(reward)
                    and np.isfinite(jump)
                    and np.isfinite(limit)
                    and np.isfinite(lag)
                )
        finally:
            env.close()
        violations = int(np.count_nonzero(np.asarray(jumps) > np.asarray(limits)))
        all_jumps.extend(jumps)
        records.append(
            {
                "seed": seed,
                "completed": bool(final_info.get("completed", False)),
                "decisions": len(jumps),
                "finite": finite,
                "reference_jump_violations": violations,
                "reference_jump_max_m": float(max(jumps, default=0.0)),
                "reference_jump_p99_m": float(np.percentile(jumps, 99)),
                "reference_jump_limit_min_m": float(min(limits, default=0.0)),
                "direction_lag_max_rad": float(max(lags, default=0.0)),
                "qp_infeasible_steps": qp_failures,
                "zero_fallback_steps": zero_fallbacks,
                "control_steps": control_steps,
            }
        )
    total_steps = sum(row["control_steps"] for row in records)
    total_qp = sum(row["qp_infeasible_steps"] for row in records)
    total_zero = sum(row["zero_fallback_steps"] for row in records)
    return {
        "purpose": "T0b continuity acceptance only; not a performance evaluation",
        "legacy_adapter": "42D observation truncated to first 39D only at policy.predict",
        "model": str(args.model),
        "model_seed": args.model_seed,
        "stochastic_policy": True,
        "episode_seed_range": [args.seed_start, args.seed_start + args.episodes - 1],
        "records": records,
        "summary": {
            "episodes": len(records),
            "finite": all(row["finite"] for row in records),
            "reference_jump_violations": sum(
                row["reference_jump_violations"] for row in records
            ),
            "reference_jump_max_m": max(
                (row["reference_jump_max_m"] for row in records), default=0.0
            ),
            "reference_jump_p99_m": float(
                np.percentile(all_jumps, 99) if all_jumps else 0.0
            ),
            "direction_lag_max_rad": max(
                (row["direction_lag_max_rad"] for row in records), default=0.0
            ),
            "qp_infeasible_steps": total_qp,
            "zero_fallback_steps": total_zero,
            "control_steps": total_steps,
            "qp_infeasible_fraction": total_qp / total_steps if total_steps else 0.0,
            "zero_fallback_fraction": total_zero / total_steps if total_steps else 0.0,
        },
        "v2_reference_level": _v2_fallback_level(args.v2_reference_dir),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial").exists():
        raise FileExistsError(args.output)
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=1))
    partial.replace(args.output)
    summary = result["summary"]
    if not summary["finite"] or summary["reference_jump_violations"]:
        raise SystemExit("T0b continuity acceptance failed")


if __name__ == "__main__":
    main()
