"""Deterministic evaluation for the canonical Phase-2 Pure SAC path.

This is the shared measurement path for all three methods: ``evaluate_model``
needs nothing but an object with ``.predict(obs, deterministic) -> (action,
state)``. It therefore has to record everything the main table needs, in the
conventions ``eval.metrics`` pins -- completion time, force impulse, worst
constraint margin and *controller* compute -- because a row it cannot fill has
to be filled by a fork, and a forked row is not comparable.

The controller and the environment are timed separately. Timing them together
was over-reporting the controller by about 8.6x on this task (``predict``
1.09 ms against ``env.step`` 8.32 ms, CPU), while ``experiments.evaluate_mpc``
has always timed ``controller.command`` alone; ``full_control_cycle_runtime_s``
is retained as the sum so historical evaluation files still parse, but it is not
the compute column.
"""

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
from eval.metrics import main_table_metrics, summarize
from train.configs import PURE_SAC


# The discount every reported return uses, shared with the scripted validators
# so a model row and the scripted row are on one scale.
GAMMA = PURE_SAC.gamma


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


def _summary(values: list[float]) -> dict[str, float] | None:
    """Per-episode summary, in the one shape ``eval.metrics`` defines."""

    return summarize(values)


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
    # Per-step compute, pooled over the whole evaluation: the per-episode
    # summaries below have already collapsed it, and the main table needs the
    # distribution over steps, not the distribution over episode means.
    all_controller_times: list[float] = []
    all_environment_times: list[float] = []
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
            # Completion needs five conditions at once, held for a full second.
            # Track how close each got on its own, and the best streak, so a
            # zero completion rate says which condition is blocking.
            completion_keys = {
                "position_error_m": "position_error_m",
                "attitude_error_rad": "attitude_error_rad",
                "total_speed_m_s": "total_speed_m_s",
                "angular_velocity_rad_s": "angular_velocity_error_rad_s",
            }
            best_completion = {
                name: float(info[key]) for name, key in completion_keys.items()
            }
            best_completion_streak = 0
            min_position = float(info["position_error_m"])
            min_waypoint_position = float(info["waypoint_position_error_m"])
            force_impulse = torque_impulse = 0.0
            saturated = samples = 0
            controller_times: list[float] = []
            environment_times: list[float] = []
            discounted_return = 0.0
            discount = 1.0
            waypoint_entry: dict[str, float] | None = None
            closest_waypoint_approach = {
                "time_s": float(info["time_seconds"]),
                "position_error_m": min_waypoint_position,
                "speed_m_s": float(info["total_speed_m_s"]),
                "fov_angle_rad": float(info["fov_angle_rad"]),
                "attitude_error_rad": float(info["attitude_error_rad"]),
                "angular_velocity_rad_s": float(
                    info["angular_velocity_error_rad_s"]
                ),
            }
            terminated = truncated = False
            while not (terminated or truncated):
                # Two timers, not one: the controller is what the compute column
                # compares, and the RK45 truth propagation is not part of it.
                started = perf_counter()
                action, _ = model.predict(observation, deterministic=deterministic)
                predicted = perf_counter()
                observation, reward, terminated, truncated, info = env.step(action)
                environment_times.append(perf_counter() - predicted)
                controller_times.append(predicted - started)
                discounted_return += discount * float(reward)
                discount *= GAMMA
                if info.get("waypoint_transition", False) and waypoint_entry is None:
                    waypoint_entry = {
                        "time_s": float(info["time_seconds"]),
                        "position_error_m": float(info["waypoint_position_error_m"]),
                        "attitude_error_rad": float(info["attitude_error_rad"]),
                        "speed_m_s": float(info["total_speed_m_s"]),
                        "fov_angle_rad": float(info["fov_angle_rad"]),
                        "angular_velocity_rad_s": float(
                            info["angular_velocity_error_rad_s"]
                        ),
                    }
                action = np.asarray(action, dtype=np.float64)
                torque_impulse += float(np.linalg.norm(action[:3] * config.max_torque_per_axis_nm)) * config.dt_s
                force_impulse += float(np.linalg.norm(action[3:] * config.max_force_per_axis_n)) * config.dt_s
                saturated += int(np.count_nonzero(np.abs(action) >= 0.95))
                samples += action.size
                min_position = min(min_position, float(info["position_error_m"]))
                for name, key in completion_keys.items():
                    best_completion[name] = min(
                        best_completion[name], float(info[key])
                    )
                best_completion_streak = max(
                    best_completion_streak, int(info["completion_streak"])
                )
                current_waypoint_position = float(info["waypoint_position_error_m"])
                if current_waypoint_position < min_waypoint_position:
                    min_waypoint_position = current_waypoint_position
                    closest_waypoint_approach = {
                        "time_s": float(info["time_seconds"]),
                        "position_error_m": current_waypoint_position,
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
            all_controller_times.extend(controller_times)
            all_environment_times.extend(environment_times)
            records.append({
                "episode": episode,
                "seed": seed + episode,
                # Episode duration, on the same names the scripted validators
                # use, so the reference row and the model rows are one table.
                # For a completed episode this is the completion time.
                "steps": int(info["step_count"]),
                "survival_s": float(info["time_seconds"]),
                "discounted_return": discounted_return,
                "completed": bool(info["completed"]),
                "waypoint_reached": bool(info["waypoint_reached"]),
                "waypoint_entry": waypoint_entry,
                "closest_waypoint_approach": closest_waypoint_approach,
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
                "best_completion_conditions": best_completion,
                "best_completion_streak": best_completion_streak,
                "minimum_position_error_m": min_position,
                "minimum_waypoint_position_error_m": min_waypoint_position,
                "final_position_error_m": float(info["position_error_m"]),
                "final_attitude_error_rad": float(info["attitude_error_rad"]),
                "final_speed_m_s": float(info["total_speed_m_s"]),
                "final_angular_velocity_rad_s": float(info["angular_velocity_error_rad_s"]),
                "force_impulse_n_s": force_impulse,
                "torque_impulse_nm_s": torque_impulse,
                "action_saturation_fraction": saturated / max(1, samples),
                "controller_time_s": _summary(controller_times),
                "environment_step_time_s": _summary(environment_times),
                # Retained so historical evaluation files and any reader that
                # already knows this key still parse. It is the sum of the two
                # above and is *not* the compute column.
                "full_control_cycle_runtime_s": _summary(
                    [
                        controller + environment
                        for controller, environment in zip(
                            controller_times, environment_times
                        )
                    ]
                ),
            })
    finally:
        env.close()
    return {
        "schema_version": 5,
        "algorithm": "pure_sac",
        "deterministic": deterministic,
        "gamma": GAMMA,
        "episodes": episodes,
        "base_seed": seed,
        "environment": asdict(config),
        # The four columns that actually separate the three methods, assembled
        # in the shared conventions rather than re-derived per method.
        "main_table": main_table_metrics(
            records,
            controller_times_s=all_controller_times,
            environment_step_times_s=all_environment_times,
            control_period_s=config.dt_s,
        ),
        "rates": {
            "episode_completion": mean(float(r["completed"]) for r in records),
            "waypoint_acquisition": mean(float(r["waypoint_reached"]) for r in records),
            "final_completion": mean(float(r["final_completed"]) for r in records),
            "constraint_success": mean(float(r["constraint_success"]) for r in records),
            # Failure-mode mix. In full_mission the completion rate alone cannot
            # say whether a policy dies in the corridor or never enters it.
            "terminal_constraint_failure": mean(
                float(r["terminal_constraint_failure"]) for r in records
            ),
            "premature_entry_failure": mean(
                float(r["premature_entry_failure"]) for r in records
            ),
            "phase1_speed_failure": mean(
                float(r["phase1_speed_failure"]) for r in records
            ),
            "distance_failure": mean(float(r["distance_failure"]) for r in records),
            "time_failure": mean(float(r["time_failure"]) for r in records),
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
        choices=("phase1_pretrain", "full_mission", "single_phase"),
        default="single_phase",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--assume-canonical",
        action="store_true",
        help=(
            "skip the manifest comparison and evaluate against the current "
            "canonical config for --mode. Use for a checkpoint trained before a "
            "rename whose manifest no longer loads but whose observation and "
            "action spaces are unchanged (the schema name is verified against "
            "the checkpoint, so a real schema mismatch still fails)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    config = phase2_environment_config(args.mode)
    manifest_path: Path | None = None
    if not args.assume_canonical:
        manifest_path = args.manifest or _manifest_for_model(args.model)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        from_manifest = environment_config_from_manifest(manifest)
        if (
            from_manifest.phase2_task != config.phase2_task
            or from_manifest.phase2_mission != config.phase2_mission
            or from_manifest.phase2_training_mode != config.phase2_training_mode
            or from_manifest.phase2_observation_schema != config.phase2_observation_schema
        ):
            raise ValueError("manifest differs from the canonical Phase-2 mission")
    model = SAC.load(args.model, device=device)
    # The network is only compatible if its observation and action spaces match
    # the current environment; a genuine schema change would surface here rather
    # than as an opaque predict-time shape error.
    reference_env = SE3RendezvousEnv(config)
    try:
        expected_obs = reference_env.observation_space.shape
        expected_act = reference_env.action_space.shape
    finally:
        reference_env.close()
    if (
        model.observation_space.shape != expected_obs
        or model.action_space.shape != expected_act
    ):
        raise ValueError(
            "checkpoint spaces "
            f"{model.observation_space.shape}/{model.action_space.shape} do not "
            f"match the canonical {expected_obs}/{expected_act}; this checkpoint "
            "is not compatible with the current observation schema"
        )
    result = evaluate_model(model, config, episodes=args.episodes, seed=args.seed, deterministic=not args.stochastic)
    result["model"] = str(args.model.resolve())
    result["training_manifest"] = (
        str(manifest_path.resolve()) if manifest_path is not None else None
    )
    result["assumed_canonical"] = bool(args.assume_canonical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Evaluation result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
