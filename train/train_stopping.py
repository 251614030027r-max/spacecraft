"""Train the learned stopping option from zero (``train/stopping.py``).

The task, the interface, the reward terms and Pure MPC are the V3e mainline's;
the differences are the stopping head, ``Q_H`` on the true MPC suffix return,
and the decision discount 0.999. ``--steps`` counts **outer decision points**
(each continue or handoff decision counts one) with one gradient update each,
as in V3e; the Pure MPC suffix after a handoff only labels ``Q_H`` and its
simulation cost is reported separately. The legacy ``train.train_hybrid`` is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from stable_baselines3.common.monitor import Monitor

from env.hybrid_env import PrecaptureHybridEnv, hybrid_mpc_config
from train.hybrid_configs import hybrid_model_kwargs, serializable_hybrid_hyperparameters
from train.mainline import MAINLINE_V3E, MAINLINE_V3E_EVALUATION_FLAGS
from train.stopping import (
    HANDOFF_FEATURE_BLOCKS,
    STOPPING,
    STOPPING_SAC,
    HandoffOptionEnv,
    SimulatedDecisionCheckpoint,
    StoppingSAC,
    handoff_feature_index,
    stopping_configs,
)
from train.train_hybrid import _code_provenance

METHOD = "learned_stopping_option"

INFO_KEYWORDS = (
    "completed",
    "terminal_region_active",
    "illegal_terminal_entry_count",
    "hybrid_qp_zero_fallbacks",
    "hybrid_v3_episode_qp_zero_fallbacks",
    "hybrid_v3_episode_control_steps",
    "hybrid_v3_episode_reference_jump_violations",
    "hybrid_branch_switches",
) + HandoffOptionEnv.EPISODE_KEYS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True, help="outer decision points (continue or handoff)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-freq", type=int, default=5_000, help="outer decisions")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--learning-starts", type=int, default=None, help="smoke tests only")
    parser.add_argument("--net-arch", type=str, default=None, help="smoke tests only, e.g. 16,16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    log_dir = args.log_root / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    environment_config, hybrid_config = stopping_configs()
    mpc_config = hybrid_mpc_config(hybrid_config, environment_config)
    raw_env = PrecaptureHybridEnv(environment_config, hybrid_config)
    raw_env.reset(seed=args.seed)
    feature_index = handoff_feature_index(raw_env.policy_observation_slices())

    model_kwargs = hybrid_model_kwargs(STOPPING_SAC)
    smoke_overrides = {}
    if args.learning_starts is not None:
        model_kwargs["learning_starts"] = smoke_overrides["learning_starts"] = args.learning_starts
    if args.net_arch is not None:
        arch = [int(v) for v in args.net_arch.split(",")]
        model_kwargs["policy_kwargs"]["net_arch"] = arch
        smoke_overrides["net_arch"] = arch

    manifest = {
        "run_name": args.run_name,
        "method": METHOD,
        **_code_provenance(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_outer_decisions": args.steps,
        "budget_semantics": "outer decision points (each continue or handoff decision "
        "counts one), one gradient update per outer decision after learning_starts; "
        "Pure MPC suffix decisions label Q_H only and are counted separately",
        "fresh_initialization": True,
        "smoke_overrides": smoke_overrides or None,
        "mainline": MAINLINE_V3E,
        # Kept so the shared V3 loaders accept the manifest; the environment is
        # rebuilt from ``hybrid`` / ``training_environment`` below, not from flags.
        "waypoint_parametrization": "task_state_v3",
        "monotone_commit": True,
        "baseline_anchored_residual": False,
        "evaluation_flags": MAINLINE_V3E_EVALUATION_FLAGS,
        "stopping": asdict(STOPPING),
        "handoff_feature_blocks": list(HANDOFF_FEATURE_BLOCKS),
        "handoff_feature_index": feature_index,
        "hyperparameters": serializable_hybrid_hyperparameters(STOPPING_SAC),
        "training_environment": asdict(environment_config),
        "hybrid": asdict(hybrid_config),
        "mpc": {
            "horizon_steps": mpc_config.horizon_steps,
            "reference_source": mpc_config.reference_source,
            "external_reference_hold_steps": mpc_config.external_reference_hold_steps,
            "solver": mpc_config.solver,
            "runtime_diagnostics": mpc_config.runtime_diagnostics,
        },
        "observation_dimension": int(raw_env.observation_space.shape[0]),
        "action_dimension": int(raw_env.action_space.shape[0]),
        "command": [sys.executable, "-B", "-m", "train.train_stopping", *sys.argv[1:]],
        "status": "running",
    }
    manifest_path = log_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))

    option_env = HandoffOptionEnv(raw_env, STOPPING, seed=args.seed)
    env = Monitor(option_env, filename=str(log_dir / "train"), info_keywords=INFO_KEYWORDS)
    model = StoppingSAC(
        "MlpPolicy",
        env,
        stopping=asdict(STOPPING),
        handoff_feature_index=feature_index,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        **model_kwargs,
    )
    option_env.handoff_probability = model.handoff_probability
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=SimulatedDecisionCheckpoint(args.checkpoint_freq, checkpoint_dir),
            reset_num_timesteps=True,
            progress_bar=False,
        )
        model.save(log_dir / "final_model")
        manifest.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat())
    except KeyboardInterrupt:
        model.save(log_dir / "interrupted_model")
        manifest.update(status="interrupted", interrupted_at_utc=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        manifest.update(
            actual_outer_decisions=int(model.num_timesteps),
            continue_transitions=int(model.continue_transitions),
            handoff_episodes=int(model.handoff_episodes),
            suffix_simulated_decisions=int(model.suffix_decisions_total),
            handoff_labels=int(model.handoff_labels),
            gradient_updates=int(model._n_updates),
        )
        manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
        env.close()


if __name__ == "__main__":
    main()
