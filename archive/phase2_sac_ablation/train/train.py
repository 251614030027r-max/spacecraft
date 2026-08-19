"""From-scratch SAC training with one declared curriculum and no resume path."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium
import stable_baselines3
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from train.callbacks import (
    CurriculumEarlyStopCallback,
    EnvironmentDiagnosticsCallback,
    FixedTaskDiagnosticCallback,
    Phase2LearningDiagnosticsCallback,
)
from train.behavior_initialization import initialize_actor_from_mpc
from train.configs import model_kwargs, serializable_hyperparameters


def fixed_simple_environment_config() -> SE3RendezvousConfig:
    base = SE3RendezvousConfig()
    return replace(
        base,
        curriculum_enabled=False,
        shell_inner_m=base.initial_shell_inner_m,
        shell_outer_m=base.initial_shell_outer_m,
        max_relative_attitude_rad=base.initial_max_relative_attitude_rad,
        inertial_velocity_component_limit_m_s=(
            base.initial_inertial_velocity_component_limit_m_s
        ),
        target_tumble_scale=base.initial_target_tumble_scale,
    )


def training_environment_config(
    *,
    fixed_simple_diagnostic: bool = False,
    phase2: bool = False,
    phase2_stage_a_only: bool = False,
) -> SE3RendezvousConfig:
    if fixed_simple_diagnostic and phase2:
        raise ValueError("fixed-simple and Phase-2 modes are mutually exclusive")
    if phase2:
        return replace(
            SE3RendezvousConfig(),
            phase2_enabled=True,
            curriculum_enabled=False,
            target_tumble_scale=0.25,
            phase2_warmup_steps=(10**12 if phase2_stage_a_only else 200_000),
            observation_attitude_scale_rad=math.radians(60.0),
            observation_distance_scale_m=5.0,
            observation_angular_velocity_scale_rad_s=0.03,
            observation_velocity_scale_m_s=0.35,
        )
    if fixed_simple_diagnostic:
        return fixed_simple_environment_config()
    return replace(SE3RendezvousConfig(), curriculum_enabled=True)


def evaluation_environment_config(
    *,
    fixed_simple_diagnostic: bool = False,
    phase2: bool = False,
    phase2_stage_a_only: bool = False,
) -> SE3RendezvousConfig:
    if fixed_simple_diagnostic and phase2:
        raise ValueError("fixed-simple and Phase-2 modes are mutually exclusive")
    if phase2:
        return replace(
            training_environment_config(
                phase2=True, phase2_stage_a_only=phase2_stage_a_only
            ),
            phase2_warmup_steps=(10**12 if phase2_stage_a_only else 0),
        )
    if fixed_simple_diagnostic:
        return fixed_simple_environment_config()
    return SE3RendezvousConfig(curriculum_enabled=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean SAC baseline")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--fixed-simple-diagnostic", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--phase2-stage-a-only", action="store_true")
    parser.add_argument("--phase2-bc-dataset", type=Path)
    parser.add_argument("--diagnostic-eval-freq", type=int, default=50_000)
    parser.add_argument("--diagnostic-eval-episodes", type=int, default=10)
    parser.add_argument("--diagnostic-eval-seed", type=int, default=20_288_000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def validate_request(args: argparse.Namespace, project_root: Path) -> tuple[Path, Path]:
    diagnostic_eval_freq = int(getattr(args, "diagnostic_eval_freq", 50_000))
    diagnostic_eval_episodes = int(
        getattr(args, "diagnostic_eval_episodes", 10)
    )
    fixed_simple_diagnostic = bool(
        getattr(args, "fixed_simple_diagnostic", False)
    )
    phase2 = bool(getattr(args, "phase2", False))
    phase2_stage_a_only = bool(getattr(args, "phase2_stage_a_only", False))
    phase2_bc_dataset = getattr(args, "phase2_bc_dataset", None)
    if phase2_stage_a_only and not phase2:
        raise ValueError("--phase2-stage-a-only requires --phase2")
    if phase2_bc_dataset is not None and not (phase2 and phase2_stage_a_only):
        raise ValueError("--phase2-bc-dataset requires fixed Phase-2 Stage-A mode")
    if phase2_bc_dataset is not None and not Path(phase2_bc_dataset).is_file():
        raise FileNotFoundError("Phase-2 behavior initialization dataset not found")
    if fixed_simple_diagnostic and phase2:
        raise ValueError("fixed-simple and Phase-2 modes are mutually exclusive")
    if min(
        args.steps,
        args.checkpoint_freq,
        diagnostic_eval_freq,
        diagnostic_eval_episodes,
    ) <= 0:
        raise ValueError("step frequencies and episode counts must be positive")
    if fixed_simple_diagnostic and args.steps > 200_000:
        raise ValueError("fixed-simple diagnostic training is capped at 200000 steps")
    if phase2_stage_a_only and args.steps > 200_000:
        raise ValueError("Phase-2 Stage-A diagnostic training is capped at 200000 steps")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_name):
        raise ValueError("run-name contains unsupported characters")
    model_dir = project_root / "models" / args.run_name
    log_dir = project_root / "logs" / args.run_name
    if model_dir.exists() or log_dir.exists():
        raise FileExistsError(
            "run output already exists; use a new run-name to preserve provenance"
        )
    return model_dir, log_dir


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    model_dir, log_dir = validate_request(args, project_root)
    device = resolve_device(args.device)
    training_config = training_environment_config(
        fixed_simple_diagnostic=args.fixed_simple_diagnostic,
        phase2=args.phase2,
        phase2_stage_a_only=args.phase2_stage_a_only,
    )
    evaluation_config = evaluation_environment_config(
        fixed_simple_diagnostic=args.fixed_simple_diagnostic,
        phase2=args.phase2,
        phase2_stage_a_only=args.phase2_stage_a_only,
    )

    if args.validate_only:
        env = SE3RendezvousEnv(training_config)
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
        print("Validation passed: configuration, environment API and device are usable.")
        return

    model_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir()
    manifest_path = log_dir / "manifest.json"
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "started_at_utc": started_at,
        "run_name": args.run_name,
        "algorithm": "sac",
        "experiment_mode": (
            "fixed_simple_diagnostic"
            if args.fixed_simple_diagnostic
            else "phase2_stage_a_v2_1"
            if args.phase2 and args.phase2_stage_a_only
            else "phase2_two_stage_v2_1"
            if args.phase2
            else "curriculum_baseline"
        ),
        "seed": args.seed,
        "device_requested": args.device,
        "device_resolved": device,
        "requested_timesteps": args.steps,
        "actual_model_timesteps": 0,
        "fresh_initialization": True,
        "initial_replay_buffer_transitions": 0,
        "behavior_initialization": (
            {
                "method": "mpc_actor_mean_behavior_initialization",
                "dataset": str(args.phase2_bc_dataset.resolve()),
            }
            if args.phase2_bc_dataset is not None
            else None
        ),
        "resume_supported": False,
        "training_environment": asdict(training_config),
        "evaluation_environment": asdict(evaluation_config),
        "hyperparameters": serializable_hyperparameters(
            phase2=args.phase2,
            behavior_initialized=args.phase2_bc_dataset is not None,
        ),
        "versions": _versions(),
        "command": [sys.executable, "-B", "-m", "train.train", *sys.argv[1:]],
        "paths": {
            "model_directory": str(model_dir.resolve()),
            "log_directory": str(log_dir.resolve()),
            "monitor": str((log_dir / "train.monitor.csv").resolve()),
            "final_model": str((model_dir / "final_model.zip").resolve()),
        },
    }
    _write_manifest(manifest_path, manifest)

    raw_env = SE3RendezvousEnv(training_config)
    raw_env.reset(seed=args.seed)
    monitor_keywords = (
        (
            "completed",
            "constraint_success",
            "distance_failure",
            "time_failure",
            "joint_success",
            "corridor_violation_steps",
            "fov_violation_steps",
            "total_speed_violation_steps",
            "closing_speed_violation_steps",
            "phase2_stage",
        )
        if args.phase2
        else (
            "completed",
            "distance_failure",
            "time_failure",
            "joint_success",
            "position_success",
            "attitude_success",
            "curriculum_progress",
            "curriculum_ceiling",
            "curriculum_rehearsal",
            "curriculum_rotation_progress",
            "curriculum_translation_progress",
            "episode_shell_outer_m",
            "episode_target_tumble_scale",
        )
    )
    env = Monitor(
        raw_env,
        filename=str(log_dir / "train"),
        info_keywords=monitor_keywords,
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=device,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        **model_kwargs(
            phase2=args.phase2,
            behavior_initialized=args.phase2_bc_dataset is not None,
        ),
    )
    if args.phase2_bc_dataset is not None:
        behavior_report = initialize_actor_from_mpc(
            model,
            args.phase2_bc_dataset,
            output_path=log_dir / "behavior_initialization.json",
            seed=args.seed,
        )
        model.save(model_dir / "behavior_initialized_model")
        manifest["behavior_initialization"] = behavior_report
        manifest["paths"]["behavior_initialized_model"] = str(
            (model_dir / "behavior_initialized_model.zip").resolve()
        )
        _write_manifest(manifest_path, manifest)
    callback_items = [
        CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoint_dir),
            name_prefix="sac",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    ]
    phase2_learning_diagnostics: Phase2LearningDiagnosticsCallback | None = None
    if args.phase2:
        phase2_learning_diagnostics = Phase2LearningDiagnosticsCallback(
            environment_config=replace(training_config, phase2_warmup_steps=10**12),
            output_directory=log_dir,
            interval_steps=25_000,
        )
        callback_items.append(phase2_learning_diagnostics)
    fixed_diagnostic: FixedTaskDiagnosticCallback | None = None
    phase2_diagnostic: FixedTaskDiagnosticCallback | None = None
    phase2_warmup_diagnostic: FixedTaskDiagnosticCallback | None = None
    curriculum_guard: CurriculumEarlyStopCallback | None = None
    if args.fixed_simple_diagnostic:
        fixed_diagnostic = FixedTaskDiagnosticCallback(
            environment_config=evaluation_config,
            output_directory=log_dir,
            eval_freq_steps=args.diagnostic_eval_freq,
            episodes=args.diagnostic_eval_episodes,
            base_seed=args.diagnostic_eval_seed,
            required_failure_windows=2,
        )
        callback_items.append(fixed_diagnostic)
    elif args.phase2 and args.phase2_stage_a_only:
        phase2_warmup_diagnostic = FixedTaskDiagnosticCallback(
            environment_config=evaluation_config,
            output_directory=log_dir,
            eval_freq_steps=args.diagnostic_eval_freq,
            episodes=args.diagnostic_eval_episodes,
            base_seed=args.diagnostic_eval_seed + 10_000,
            required_failure_windows=3,
            diagnostic_name="phase2_stage_a",
            first_eval_step=args.diagnostic_eval_freq,
        )
        callback_items.append(phase2_warmup_diagnostic)
    elif args.phase2:
        phase2_warmup_diagnostic = FixedTaskDiagnosticCallback(
            environment_config=replace(
                evaluation_config, phase2_warmup_steps=10**12
            ),
            output_directory=log_dir,
            eval_freq_steps=args.steps + 1,
            episodes=args.diagnostic_eval_episodes,
            base_seed=args.diagnostic_eval_seed + 10_000,
            required_failure_windows=1,
            diagnostic_name="phase2_warmup",
            first_eval_step=training_config.phase2_warmup_steps,
        )
        phase2_diagnostic = FixedTaskDiagnosticCallback(
            environment_config=evaluation_config,
            output_directory=log_dir,
            eval_freq_steps=args.diagnostic_eval_freq,
            episodes=args.diagnostic_eval_episodes,
            base_seed=args.diagnostic_eval_seed,
            required_failure_windows=2,
            diagnostic_name="phase2_nominal",
            first_eval_step=(
                training_config.phase2_warmup_steps + args.diagnostic_eval_freq
            ),
        )
        callback_items.append(phase2_warmup_diagnostic)
        callback_items.append(phase2_diagnostic)
    elif not args.phase2:
        curriculum_guard = CurriculumEarlyStopCallback(
            output_directory=log_dir,
            curriculum_start_step=training_config.curriculum_start_step,
            window_steps=25_000,
            required_failure_windows=3,
            required_stall_windows=8,
        )
        callback_items.append(curriculum_guard)
    callback_items.append(EnvironmentDiagnosticsCallback())
    callbacks = CallbackList(
        callback_items
    )
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        model.save(model_dir / "final_model")
        if fixed_diagnostic is not None:
            manifest["fixed_simple_diagnostic"] = {
                "stopped_early": fixed_diagnostic.stopped_early,
                "stop_reason": fixed_diagnostic.stop_reason,
                "checks": fixed_diagnostic.records,
            }
        if phase2_diagnostic is not None:
            manifest["phase2_nominal_diagnostic"] = {
                "stopped_early": phase2_diagnostic.stopped_early,
                "stop_reason": phase2_diagnostic.stop_reason,
                "checks": phase2_diagnostic.records,
            }
        if phase2_warmup_diagnostic is not None:
            manifest["phase2_warmup_diagnostic"] = {
                "stopped_early": phase2_warmup_diagnostic.stopped_early,
                "stop_reason": phase2_warmup_diagnostic.stop_reason,
                "checks": phase2_warmup_diagnostic.records,
            }
        if curriculum_guard is not None:
            manifest["curriculum_early_stop"] = {
                "stopped_early": curriculum_guard.stopped_early,
                "stop_reason": curriculum_guard.stop_reason,
                "windows": curriculum_guard.records,
            }
        manifest.update(
            status=(
                "diagnostic_stopped_early"
                if fixed_diagnostic is not None and fixed_diagnostic.stopped_early
                else "diagnostic_completed"
                if fixed_diagnostic is not None
                else "phase2_stopped_early"
                if (
                    phase2_diagnostic is not None
                    and phase2_diagnostic.stopped_early
                )
                or (
                    phase2_warmup_diagnostic is not None
                    and phase2_warmup_diagnostic.stopped_early
                )
                else "completed"
                if phase2_diagnostic is not None
                else "curriculum_stopped_early"
                if curriculum_guard is not None and curriculum_guard.stopped_early
                else "completed"
            ),
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            actual_model_timesteps=int(model.num_timesteps),
        )
    except KeyboardInterrupt:
        model.save(model_dir / "interrupted_model")
        manifest.update(
            status="interrupted",
            interrupted_at_utc=datetime.now(timezone.utc).isoformat(),
            actual_model_timesteps=int(model.num_timesteps),
            interrupted_model=str((model_dir / "interrupted_model.zip").resolve()),
        )
        raise
    except Exception as error:
        manifest.update(
            status="failed",
            failed_at_utc=datetime.now(timezone.utc).isoformat(),
            actual_model_timesteps=int(model.num_timesteps),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        _write_manifest(manifest_path, manifest)
        env.close()


if __name__ == "__main__":
    main()
