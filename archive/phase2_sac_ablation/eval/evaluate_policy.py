"""Deterministic 100-episode evaluation for a trained clean SAC model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.termination import SuccessThresholds
from env.task import Phase2TaskConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a clean SAC baseline")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_training_manifest(model_path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    try:
        relative = model_path.resolve().relative_to(
            (project_root / "models").resolve()
        )
    except (OSError, ValueError) as error:
        raise ValueError("model must be inside this project's models directory") from error
    if not relative.parts:
        raise ValueError("model path does not identify a run")
    manifest_path = project_root / "logs" / relative.parts[0] / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"training manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "sac":
        raise ValueError("manifest does not describe a SAC model")
    if manifest.get("status") != "completed":
        raise ValueError("only a completed training run can be officially evaluated")
    if not manifest.get("fresh_initialization"):
        raise ValueError("official baseline evaluation requires fresh initialization")
    return manifest


def environment_config_from_manifest(
    manifest: dict[str, Any]
) -> SE3RendezvousConfig:
    values = manifest.get("evaluation_environment")
    if not isinstance(values, dict):
        raise ValueError("manifest is missing evaluation_environment")
    values = dict(values)
    thresholds = values.get("success_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("manifest is missing success thresholds")
    values["success_thresholds"] = SuccessThresholds(**thresholds)
    task = values.get("phase2_task")
    if isinstance(task, dict):
        values["phase2_task"] = Phase2TaskConfig(**task)
    warmup_task = values.get("phase2_warmup_task")
    if isinstance(warmup_task, dict):
        values["phase2_warmup_task"] = Phase2TaskConfig(**warmup_task)
    config = SE3RendezvousConfig(**values)
    if config.curriculum_enabled:
        raise ValueError("official evaluation environment must disable curriculum")
    return config


def _joint_threshold_ratio(
    info: dict[str, Any], config: SE3RendezvousConfig
) -> float:
    t = config.success_thresholds
    if config.phase2_enabled:
        task = config.phase2_task
        return max(
            float(info["attitude_error_rad"]) / task.completion_attitude_rad,
            float(info["angular_velocity_error_rad_s"])
            / task.completion_angular_velocity_rad_s,
            float(info["position_error_m"]) / task.completion_position_m,
            float(info["translational_velocity_error_m_s"])
            / task.completion_speed_m_s,
        )
    return max(
        float(info["attitude_error_rad"]) / t.attitude_rad,
        float(info["angular_velocity_error_rad_s"]) / t.angular_velocity_rad_s,
        float(info["position_error_m"]) / t.position_m,
        float(info["translational_velocity_error_m_s"])
        / t.translational_velocity_m_s,
    )


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate(
    model_path: Path,
    *,
    episodes: int,
    seed: int,
    device: str,
    deterministic: bool = True,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    manifest = load_training_manifest(model_path)
    config = environment_config_from_manifest(manifest)
    env = SE3RendezvousEnv(config)
    model = SAC.load(str(model_path), env=env, device=device)
    records: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            episode_return = 0.0
            force_l1_impulse = 0.0
            torque_l1_impulse = 0.0
            saturated_steps = 0
            maximum_distance = float(info["relative_distance_m"])
            best_ratio = _joint_threshold_ratio(info, config)
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=deterministic)
                action = np.asarray(action, dtype=np.float64)
                saturated_steps += int(np.any(np.abs(action) >= 1.0 - 1.0e-6))
                wrench = env.scale_action(action)
                force_l1_impulse += float(np.sum(np.abs(wrench.force))) * config.dt_s
                torque_l1_impulse += (
                    float(np.sum(np.abs(wrench.torque))) * config.dt_s
                )
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                maximum_distance = max(
                    maximum_distance, float(info["relative_distance_m"])
                )
                best_ratio = min(
                    best_ratio,
                    _joint_threshold_ratio(info, config),
                )
            records.append(
                {
                    "episode": episode,
                    "seed": seed + episode,
                    "return": episode_return,
                    "steps": env.step_count,
                    "time_seconds": env.time_seconds,
                    "completed": bool(info["completed"]),
                    "joint_success": bool(info["joint_success"]),
                    "attitude_success": bool(info["attitude_success"]),
                    "position_success": bool(info["position_success"]),
                    "distance_failure": bool(info["distance_failure"]),
                    "time_failure": bool(info["time_failure"]),
                    "attitude_error_rad": float(info["attitude_error_rad"]),
                    "angular_velocity_error_rad_s": float(
                        info["angular_velocity_error_rad_s"]
                    ),
                    "position_error_m": float(info["position_error_m"]),
                    "translational_velocity_error_m_s": float(
                        info["translational_velocity_error_m_s"]
                    ),
                    "maximum_relative_distance_m": maximum_distance,
                    "best_joint_threshold_ratio": best_ratio,
                    "force_l1_impulse_n_s": force_l1_impulse,
                    "torque_l1_impulse_nm_s": torque_l1_impulse,
                    "action_saturation_fraction": saturated_steps
                    / max(1, env.step_count),
                    **(
                        {
                            "constraint_success": bool(info["constraint_success"]),
                            "corridor_violation_steps": int(
                                info["corridor_violation_steps"]
                            ),
                            "fov_violation_steps": int(info["fov_violation_steps"]),
                            "total_speed_violation_steps": int(
                                info["total_speed_violation_steps"]
                            ),
                            "closing_speed_violation_steps": int(
                                info["closing_speed_violation_steps"]
                            ),
                        }
                        if config.phase2_enabled
                        else {}
                    ),
                }
            )
    finally:
        env.close()

    rate_keys = (
        "completed",
        "joint_success",
        "attitude_success",
        "position_success",
        "distance_failure",
        "time_failure",
    )
    scalar_keys = (
        "return",
        "steps",
        "time_seconds",
        "attitude_error_rad",
        "angular_velocity_error_rad_s",
        "position_error_m",
        "translational_velocity_error_m_s",
        "maximum_relative_distance_m",
        "best_joint_threshold_ratio",
        "force_l1_impulse_n_s",
        "torque_l1_impulse_nm_s",
        "action_saturation_fraction",
    )
    rates = {
        key: mean(float(record[key]) for record in records) for key in rate_keys
    }
    if config.phase2_enabled:
        rates["constraint_success"] = mean(
            float(record["constraint_success"]) for record in records
        )
    acceptance_checks = {
        "at_least_100_episodes": episodes >= 100,
        "completed_rate_at_least_0p80": rates["completed"] >= 0.80,
        "joint_success_rate_at_least_0p80": rates["joint_success"] >= 0.80,
        "position_success_rate_at_least_0p90": rates["position_success"] >= 0.90,
        "distance_failure_rate_at_most_0p05": rates["distance_failure"] <= 0.05,
        "timeout_rate_at_most_0p15": rates["time_failure"] <= 0.15,
    }
    if config.phase2_enabled:
        acceptance_checks["constraint_success_rate_at_least_0p80"] = (
            rates["constraint_success"] >= 0.80
        )
    return {
        "schema_version": 2,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "sac",
        "model": str(model_path.resolve()),
        "episodes": episodes,
        "base_seed": seed,
        "device": device,
        "deterministic": deterministic,
        "environment": asdict(config),
        "training_manifest": manifest,
        "rates": rates,
        "statistics": {
            key: _summary([float(record[key]) for record in records])
            for key in scalar_keys
        },
        "acceptance": {
            "checks": acceptance_checks,
            "passed": all(acceptance_checks.values()),
        },
        "episode_records": records,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            "evaluation output already exists; use a new filename"
        )
    device = resolve_device(args.device)
    result = evaluate(
        args.model,
        episodes=args.episodes,
        seed=args.seed,
        device=device,
        deterministic=not args.stochastic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.write_text(rendered, encoding="utf-8")
    if not args.quiet:
        print(rendered)
    print(f"Evaluation result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
