"""Train the coupled SAC-MPC upper layer. Every run starts from zero.

The learned layer emits one 3D waypoint in the target body frame per 2 s
decision; the constrained MPC flies it and is the only thing that touches the
actuators. The manifest written here is the reproduction record: config plus
seed is enough to rebuild the run, which is why no checkpoint is ever an input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv, hybrid_mpc_config
from env.phase2_env import precapture_planning_environment_config
from env.se3_rendezvous_env import SE3RendezvousConfig
from train.hybrid_configs import (
    SAC_MPC_HYBRID,
    hybrid_model_kwargs,
    serializable_hybrid_hyperparameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, required=True, help="decision steps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--parametrization",
        choices=["absolute", "radial_local", "arrival_condition"],
        default="arrival_condition",
        help=(
            "How the action names the reference. 'absolute' is what the first "
            "training attempt used and is measured to be unlearnable -- 70%% of "
            "that box commands a point further out than the chaser starts. "
            "'radial_local' names a point in coordinates built from the "
            "chaser's own position. 'arrival_condition' is the default now: a "
            "2D action naming how far the commit has gone and how far out the "
            "inertially frozen hold sits, whose commit end reproduces the "
            "fixed-setpoint Pure MPC controller wrench for wrench."
        ),
    )
    parser.add_argument(
        "--execution-feedback",
        dest="execution_feedback",
        action="store_true",
        default=True,
        help="Feed the lower layer's execution summary back into the policy "
        "observation (the default; this is the coupling's reverse channel).",
    )
    parser.add_argument(
        "--no-execution-feedback",
        dest="execution_feedback",
        action="store_false",
        help="Single-factor ablation: identical run with the reverse channel "
        "removed and nothing else changed.",
    )
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def accelerated_training_configs(
    *,
    horizon_steps: int,
    waypoint_parametrization: str,
    execution_feedback: bool = True,
) -> tuple[SE3RendezvousConfig, PrecaptureHybridConfig]:
    """Return engineering-equivalent configs used only by hybrid training."""

    environment_config = replace(
        precapture_planning_environment_config(), cache_target_trajectory=False
    )
    hybrid_config = PrecaptureHybridConfig(
        horizon_steps=horizon_steps,
        waypoint_parametrization=waypoint_parametrization,
        decision_discount_factor=SAC_MPC_HYBRID.gamma,
        runtime_diagnostics=False,
        include_target_phase_and_time_observation=True,
        include_execution_feedback_observation=execution_feedback,
    )
    return environment_config, hybrid_config


def main() -> None:
    args = parse_args()
    log_dir = args.log_root / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=args.horizon,
        waypoint_parametrization=args.parametrization,
        execution_feedback=args.execution_feedback,
    )
    mpc_config = hybrid_mpc_config(hybrid_config, environment_config)

    manifest = {
        "run_name": args.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "requested_decision_steps": args.steps,
        "fresh_initialization": True,
        "critic_initialization": "fresh",
        "initial_replay_buffer_transitions": 0,
        "architecture": "one architecture for the whole mission: the MPC is the "
        "only actuator path from 17 m to contact; there is no phase switch",
        "action_space": (
            f"{hybrid_config.action_dimension}D action, "
            f"parametrisation={hybrid_config.waypoint_parametrization}, "
            "resolved to an absolute waypoint in the target body frame, "
            f"radially clipped to [{hybrid_config.minimum_waypoint_radius_m}, "
            f"{hybrid_config.maximum_waypoint_radius_m}] m. The interface to "
            "the MPC is the unchanged 3D waypoint in every parametrisation."
        ),
        "observation_space": (
            "canonical 24D full-state core"
            + (
                ", a 6D continuous absolute-target-attitude representation and "
                "normalized remaining episode time"
                if hybrid_config.include_target_phase_and_time_observation
                else ""
            )
            + (
                ", and a 3D execution-feedback summary of the previous "
                "decision (fallback fraction, peak solved-step slack, mean "
                "actuator usage)"
                if hybrid_config.include_execution_feedback_observation
                else ""
            )
        ),
        "coupling_direction": (
            "bidirectional: the policy proposes an arrival condition and the "
            "constrained MPC returns how hard that proposal was to execute"
            if hybrid_config.include_execution_feedback_observation
            else "one-way: the policy proposes and the MPC executes, with no "
            "return path"
        ),
        "decision_period_s": hybrid_config.decision_period_steps
        * environment_config.dt_s,
        "reward": "time, force, torque, safety, and event terms are summed "
        "over the decision period; potential shaping is evaluated once at "
        "the decision boundary as weight * (gamma_SAC * Phi(s_next) - "
        "Phi(s)); nothing rewards entering legally or waiting",
        "training_environment": asdict(environment_config),
        "hybrid": asdict(hybrid_config),
        "mpc": {
            "horizon_steps": mpc_config.horizon_steps,
            "reference_source": mpc_config.reference_source,
            "external_reference_frame": mpc_config.external_reference_frame,
            "external_reference_hold_steps": mpc_config.external_reference_hold_steps,
            "precapture_attitude_reference": mpc_config.precapture_attitude_reference,
            "terminal_weight": mpc_config.terminal_weight,
            "input_weight": mpc_config.input_weight,
            "linearization_source": mpc_config.linearization_source,
            "solver": mpc_config.solver,
            "runtime_diagnostics": mpc_config.runtime_diagnostics,
        },
        "performance_semantics": {
            "target_trajectory_mode": "on_demand_rk45",
            "target_trajectory_equivalence": (
                "same propagate_rk45 call, parameters, gravity, RK45Settings, "
                "dt_s and time stamps as eager cache construction"
            ),
            "target_nfev_semantics": (
                "real per-step RK45 function-evaluation count; eager-cache runs "
                "reported zero via synthetic cached IntegrationDiagnostics, so "
                "target_nfev is not directly comparable across the switch"
            ),
            "target_cache_benefit_note": (
                "avoids unused propagation and the phase-keyed LRU only for "
                "episodes shorter than the 300 s limit; speed benefit shrinks "
                "toward zero as episode length approaches 150 decisions"
            ),
        },
        "hyperparameters": serializable_hybrid_hyperparameters(),
        "command": [sys.executable, "-B", "-m", "train.train_hybrid", *sys.argv[1:]],
        "status": "running",
    }
    manifest_path = log_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))

    raw_env = PrecaptureHybridEnv(environment_config, hybrid_config)
    raw_env.reset(seed=args.seed)
    env = Monitor(
        raw_env,
        filename=str(log_dir / "train"),
        info_keywords=(
            "completed",
            "terminal_region_active",
            "illegal_terminal_entry_count",
            "hybrid_waypoint_radius_m",
            "hybrid_qp_zero_fallbacks",
        ),
    )
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
        **hybrid_model_kwargs(SAC_MPC_HYBRID),
    )
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoint_dir),
                name_prefix="sac_mpc",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        ]
    )
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=False,
        )
        model.save(log_dir / "final_model")
        manifest.update(
            status="completed",
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
            actual_decision_steps=int(model.num_timesteps),
        )
    except KeyboardInterrupt:
        model.save(log_dir / "interrupted_model")
        manifest.update(
            status="interrupted",
            interrupted_at_utc=datetime.now(timezone.utc).isoformat(),
            actual_decision_steps=int(model.num_timesteps),
        )
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=1, default=str))
        env.close()


if __name__ == "__main__":
    main()
