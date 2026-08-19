"""Canonical from-scratch Pure SAC entry for the Phase-2 task."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import asdict
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

from env.phase2_env import Phase2Mode, phase2_environment_config
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from train.callbacks import DeterministicPhase2EvalCallback, Phase2DiagnosticsCallback
from train.configs import model_kwargs, serializable_hyperparameters


def training_environment_config(
    mode: Phase2Mode = "full_mission",
) -> SE3RendezvousConfig:
    """Return the exact canonical distribution for every training episode."""

    return phase2_environment_config(mode)


def evaluation_environment_config(
    mode: Phase2Mode = "full_mission",
) -> SE3RendezvousConfig:
    return phase2_environment_config(mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canonical Phase-2 Pure SAC")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--mode",
        choices=("phase1_pretrain", "full_mission"),
        default="full_mission",
    )
    parser.add_argument(
        "--actor-init",
        type=Path,
        help="S1 model used for actor-only S2 initialization; critic/replay stay fresh",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        help="explicit increasing evaluation steps; overrides --eval-freq",
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=20_288_000)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


def validate_request(args: argparse.Namespace, project_root: Path) -> tuple[Path, Path]:
    positive = [args.steps, args.checkpoint_freq]
    for name in ("eval_freq", "eval_episodes"):
        if hasattr(args, name):
            positive.append(getattr(args, name))
    eval_steps = getattr(args, "eval_steps", None)
    if eval_steps is not None:
        if any(step <= 0 for step in eval_steps) or list(eval_steps) != sorted(set(eval_steps)):
            raise ValueError("eval-steps must be positive, unique, and increasing")
        if eval_steps[-1] > args.steps:
            raise ValueError("eval-steps cannot exceed total training steps")
    if min(positive) <= 0:
        raise ValueError("step frequencies and episode counts must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_name):
        raise ValueError("run-name contains unsupported characters")
    model_dir = project_root / "models" / args.run_name
    log_dir = project_root / "logs" / args.run_name
    if model_dir.exists() or log_dir.exists():
        raise FileExistsError("run output already exists; use a new run-name")
    if getattr(args, "actor_init", None) is not None:
        if args.mode != "full_mission":
            raise ValueError("actor-init is only valid for full_mission S2")
        if not args.actor_init.is_file():
            raise FileNotFoundError(args.actor_init)
    return model_dir, log_dir


def _versions() -> dict[str, str]:
    return {"python": platform.python_version(), "torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__, "numpy": version("numpy"), "scipy": version("scipy")}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    model_dir, log_dir = validate_request(args, root)
    training_config = training_environment_config(args.mode)
    evaluation_config = evaluation_environment_config(args.mode)
    device = resolve_device(args.device)
    if args.validate_only:
        env = SE3RendezvousEnv(training_config)
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()
        print("Validation passed: canonical Phase-2 environment and SAC configuration are usable.")
        return

    model_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir()
    manifest_path = log_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 5,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "algorithm": "pure_sac",
        "mission_task_version": "s1v2_acquisition",
        "translational_observation_frame": "chaser_body",
        "translational_position_frame": "chaser_body",
        "translational_velocity_observation": "tracking_error",
        "force_action_frame": "chaser_body",
        "observation_schema": training_config.phase2_observation_schema,
        "phase2_mode": args.mode,
        "seed": args.seed,
        "requested_timesteps": args.steps,
        "actual_model_timesteps": 0,
        "fresh_initialization": args.actor_init is None,
        "actor_initialization": (
            None if args.actor_init is None else str(args.actor_init.resolve())
        ),
        "critic_initialization": "fresh",
        "initial_replay_buffer_transitions": 0,
        "training_environment": asdict(training_config),
        "evaluation_environment": asdict(evaluation_config),
        "hyperparameters": serializable_hyperparameters(),
        "callbacks": ["checkpoint", "phase2_diagnostics", "deterministic_phase2_evaluation"],
        "evaluation_steps": args.eval_steps,
        "versions": _versions(),
        "command": [sys.executable, "-B", "-m", "train.train", *sys.argv[1:]],
    }
    _write_manifest(manifest_path, manifest)
    raw_env = SE3RendezvousEnv(training_config)
    raw_env.reset(seed=args.seed)
    env = Monitor(
        raw_env,
        filename=str(log_dir / "train"),
        info_keywords=(
            "completed",
            "gate_reached",
            "constraint_success",
            "phase1_speed_failure",
            "premature_entry_failure",
            "terminal_constraint_failure",
            "distance_failure",
            "time_failure",
            "mission_phase",
            "phase1_curriculum_difficulty",
            "phase1_curriculum_full_sample",
        ),
    )
    model = SAC("MlpPolicy", env, seed=args.seed, device=device, verbose=1, tensorboard_log=str(log_dir / "tensorboard"), **model_kwargs())
    if args.actor_init is not None:
        source = SAC.load(args.actor_init, device=device)
        if source.observation_space != model.observation_space or source.action_space != model.action_space:
            raise ValueError("actor-init spaces differ from the canonical mission")
        model.actor.load_state_dict(source.actor.state_dict())
    callbacks = CallbackList([
        CheckpointCallback(save_freq=args.checkpoint_freq, save_path=str(checkpoint_dir), name_prefix="sac", save_replay_buffer=False, save_vecnormalize=False),
        Phase2DiagnosticsCallback(log_dir, probe_config=training_config),
        DeterministicPhase2EvalCallback(log_dir / "evaluations", mode=args.mode, eval_freq_steps=args.eval_freq, eval_steps=(None if args.eval_steps is None else tuple(args.eval_steps)), episodes=args.eval_episodes, base_seed=args.eval_seed),
    ])
    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=True, progress_bar=False)
        model.save(model_dir / "final_model")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), actual_model_timesteps=int(model.num_timesteps))
    except KeyboardInterrupt:
        model.save(model_dir / "interrupted_model")
        manifest.update(status="interrupted", interrupted_at_utc=datetime.now(timezone.utc).isoformat(), actual_model_timesteps=int(model.num_timesteps))
        raise
    finally:
        _write_manifest(manifest_path, manifest)
        env.close()


if __name__ == "__main__":
    main()
