"""Deterministic evaluation for the canonical Phase-2 Pure SAC path."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

from env.phase2_env import Phase2Mode, phase2_environment_config
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from env.task import Phase2MissionConfig, Phase2TaskConfig
from env.termination import SuccessThresholds


def environment_config_from_manifest(
    manifest: dict[str, Any],
) -> SE3RendezvousConfig:
    values = dict(manifest["evaluation_environment"])
    for key in ("phase2_task", "phase2_warmup_task"):
        if isinstance(values.get(key), dict):
            values[key] = Phase2TaskConfig(**values[key])
    if isinstance(values.get("phase2_mission"), dict):
        values["phase2_mission"] = Phase2MissionConfig(**values["phase2_mission"])
    if isinstance(values.get("success_thresholds"), dict):
        values["success_thresholds"] = SuccessThresholds(
            **values["success_thresholds"]
        )
    config = SE3RendezvousConfig(**values)
    if (
        not config.phase2_enabled
        or not config.phase2_mission_enabled
        or config.curriculum_enabled
    ):
        raise ValueError("manifest is not a canonical Phase-2 mission evaluation")
    return config


def _manifest_for_model(model_path: Path) -> Path:
    resolved = model_path.resolve()
    parts = list(resolved.parts)
    if "models" not in parts:
        raise FileNotFoundError("cannot infer manifest outside the models tree")
    index = parts.index("models")
    if index + 1 >= len(parts):
        raise FileNotFoundError("model path has no run directory")
    return Path(*parts[:index], "logs", parts[index + 1], "manifest.json")


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def evaluate_model(
    model: SAC,
    config: SE3RendezvousConfig,
    *,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    if episodes <= 0 or not config.phase2_enabled:
        raise ValueError("positive episodes and a Phase-2 config are required")
    records: list[dict[str, Any]] = []
    env = SE3RendezvousEnv(config)
    try:
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            min_margins = {name: float("inf") for name in (
                "corridor_axial_margin_m", "corridor_lateral_margin_m",
                "fov_margin_rad", "total_speed_margin_m_s",
                "closing_speed_margin_m_s",
            )}
            first_violation: dict[str, Any] | None = None
            min_position = float(info["position_error_m"])
            min_gate_position = float(info["gate_position_error_m"])
            force_impulse = torque_impulse = 0.0
            saturated = samples = 0
            cycle_times: list[float] = []
            gate_entry: dict[str, float] | None = None
            closest_gate_approach = {
                "time_s": float(info["time_seconds"]),
                "position_error_m": min_gate_position,
                "speed_m_s": float(info["total_speed_m_s"]),
                "fov_angle_rad": float(info["fov_angle_rad"]),
                "attitude_error_rad": float(info["attitude_error_rad"]),
                "angular_velocity_rad_s": float(
                    info["angular_velocity_error_rad_s"]
                ),
            }
            terminated = truncated = False
            while not (terminated or truncated):
                started = perf_counter()
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, _, terminated, truncated, info = env.step(action)
                if info.get("gate_transition", False) and gate_entry is None:
                    gate_entry = {
                        "time_s": float(info["time_seconds"]),
                        "position_error_m": float(info["gate_position_error_m"]),
                        "attitude_error_rad": float(info["attitude_error_rad"]),
                        "speed_m_s": float(info["total_speed_m_s"]),
                        "fov_angle_rad": float(info["fov_angle_rad"]),
                        "angular_velocity_rad_s": float(
                            info["angular_velocity_error_rad_s"]
                        ),
                    }
                cycle_times.append(perf_counter() - started)
                action = np.asarray(action, dtype=np.float64)
                torque_impulse += float(np.linalg.norm(action[:3] * config.max_torque_per_axis_nm)) * config.dt_s
                force_impulse += float(np.linalg.norm(action[3:] * config.max_force_per_axis_n)) * config.dt_s
                saturated += int(np.count_nonzero(np.abs(action) >= 0.95))
                samples += action.size
                min_position = min(min_position, float(info["position_error_m"]))
                current_gate_position = float(info["gate_position_error_m"])
                if current_gate_position < min_gate_position:
                    min_gate_position = current_gate_position
                    closest_gate_approach = {
                        "time_s": float(info["time_seconds"]),
                        "position_error_m": current_gate_position,
                        "speed_m_s": float(info["total_speed_m_s"]),
                        "fov_angle_rad": float(info["fov_angle_rad"]),
                        "attitude_error_rad": float(info["attitude_error_rad"]),
                        "angular_velocity_rad_s": float(
                            info["angular_velocity_error_rad_s"]
                        ),
                    }
                for name in min_margins:
                    value = float(info[name])
                    min_margins[name] = min(min_margins[name], value)
                    if value < 0.0 and first_violation is None:
                        first_violation = {
                            "time_s": float(info["time_seconds"]), "type": name,
                            "margin": value,
                        }
            records.append({
                "episode": episode,
                "seed": seed + episode,
                "completed": bool(info["completed"]),
                "gate_reached": bool(info["gate_reached"]),
                "gate_entry": gate_entry,
                "closest_gate_approach": closest_gate_approach,
                "final_completed": bool(info.get("final_completed", False)),
                "terminal_constraint_failure": bool(
                    info.get("terminal_constraint_failure", False)
                ),
                "phase1_speed_failure": bool(
                    info.get("phase1_speed_failure", False)
                ),
                "premature_entry_failure": bool(
                    info.get("premature_entry_failure", False)
                ),
                "distance_failure": bool(info.get("distance_failure", False)),
                "time_failure": bool(info.get("time_failure", False)),
                "constraint_success": bool(info["constraint_success"]),
                "first_violation": first_violation,
                "minimum_margins": min_margins,
                "minimum_position_error_m": min_position,
                "minimum_gate_position_error_m": min_gate_position,
                "final_position_error_m": float(info["position_error_m"]),
                "final_attitude_error_rad": float(info["attitude_error_rad"]),
                "final_speed_m_s": float(info["total_speed_m_s"]),
                "final_angular_velocity_rad_s": float(info["angular_velocity_error_rad_s"]),
                "force_impulse_n_s": force_impulse,
                "torque_impulse_nm_s": torque_impulse,
                "action_saturation_fraction": saturated / max(1, samples),
                "full_control_cycle_runtime_s": _summary(cycle_times),
            })
    finally:
        env.close()
    return {
        "schema_version": 4,
        "algorithm": "pure_sac",
        "deterministic": deterministic,
        "episodes": episodes,
        "base_seed": seed,
        "environment": asdict(config),
        "rates": {
            "episode_completion": mean(float(r["completed"]) for r in records),
            "gate_acquisition": mean(float(r["gate_reached"]) for r in records),
            "final_completion": mean(float(r["final_completed"]) for r in records),
            "constraint_success": mean(float(r["constraint_success"]) for r in records),
        },
        "episode_records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate canonical Phase-2 Pure SAC")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--mode",
        choices=("phase1_pretrain", "full_mission"),
        default="full_mission",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    manifest_path = args.manifest or _manifest_for_model(args.model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = environment_config_from_manifest(manifest)
    expected = phase2_environment_config(args.mode)
    if (
        config.phase2_task != expected.phase2_task
        or config.phase2_mission != expected.phase2_mission
        or config.phase2_training_mode != expected.phase2_training_mode
        or config.phase2_observation_schema != expected.phase2_observation_schema
    ):
        raise ValueError("manifest differs from the canonical Phase-2 mission")
    config = expected
    model = SAC.load(args.model, device=device)
    result = evaluate_model(model, config, episodes=args.episodes, seed=args.seed, deterministic=not args.stochastic)
    result["model"] = str(args.model.resolve())
    result["training_manifest"] = str(manifest_path.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluation result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
