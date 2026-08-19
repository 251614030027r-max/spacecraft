"""Use constrained-MPC trajectories to compare Phase-2 observation schemas."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from controllers.mpc.config import phase2_mpc_config
from controllers.mpc.controller import MPCController
from controllers.mpc.prediction import (
    LocalRelativePredictionModel,
    RelativePredictionModel,
    relative_to_vector,
)
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings
from dynamics.types import GeneralizedForce
from env.action import wrench_to_normalized
from env.observation import build_phase2_observation, build_phase2_observation_v1
from env.scenarios import chaser_parameters, target_parameters
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from train.train import evaluation_environment_config, training_environment_config


class ActionRegressor(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = input_size
        for _ in range(3):
            layers.extend((nn.Linear(width, 256), nn.ReLU()))
            width = 256
        layers.extend((nn.Linear(width, 6), nn.Tanh()))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _controller(env_config: SE3RendezvousConfig) -> MPCController:
    task = (
        env_config.phase2_warmup_task
        if env_config.phase2_warmup_steps > 0
        else env_config.phase2_task
    )
    config = phase2_mpc_config(task=task)
    reference = RelativePredictionModel(
        target_parameters=target_parameters(),
        chaser_parameters=chaser_parameters(),
        dt_s=env_config.dt_s,
        gravity_options=GravityOptions(include_j2=env_config.include_j2),
        solver_settings=RK45Settings(
            rtol=env_config.solver_rtol,
            atol=env_config.solver_atol,
            max_step=env_config.dt_s,
        ),
    )
    local = LocalRelativePredictionModel(reference.chaser_parameters, env_config.dt_s)
    return MPCController(config, local, reference)


def _observations(env: SE3RendezvousEnv) -> tuple[np.ndarray, np.ndarray]:
    assert env.relative is not None and env.target_state is not None
    config = env.config
    task = env._active_phase2_task
    common = {
        "relative": env.relative,
        "task": task,
        "attitude_scale_rad": config.observation_attitude_scale_rad,
        "distance_scale_m": config.observation_distance_scale_m,
        "angular_velocity_scale_rad_s": config.observation_angular_velocity_scale_rad_s,
        "velocity_scale_m_s": config.observation_velocity_scale_m_s,
        "softsign_limit": config.observation_softsign_limit,
    }
    old = build_phase2_observation_v1(**common)
    new = build_phase2_observation(
        **common,
        target_angular_velocity_rad_s=env.target_state.omega,
        target_angular_velocity_scale_rad_s=(
            config.phase2_observation_target_angular_velocity_scale_rad_s
        ),
    )
    return old, new


def _collect_group(
    *,
    name: str,
    config: SE3RendezvousConfig,
    episodes: int,
    base_seed: int,
    trajectory_offset: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    env = SE3RendezvousEnv(config)
    controller = _controller(config)
    old_rows: list[np.ndarray] = []
    new_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    trajectory_rows: list[int] = []
    records = []
    try:
        for episode in range(episodes):
            seed = base_seed + episode
            env.reset(seed=seed)
            controller.reset()
            terminated = truncated = False
            fallbacks = 0
            command_times = []
            while not (terminated or truncated):
                assert env.relative is not None and env.target_state is not None
                old, new = _observations(env)
                started = perf_counter()
                wrench, diagnostics = controller.command(
                    relative_to_vector(env.relative),
                    target_state=env.target_state,
                    time_seconds=env.time_seconds,
                )
                command_times.append(perf_counter() - started)
                action = wrench_to_normalized(
                    GeneralizedForce.from_vector(wrench),
                    max_torque_per_axis_nm=config.max_torque_per_axis_nm,
                    max_force_per_axis_n=config.max_force_per_axis_n,
                )
                old_rows.append(old)
                new_rows.append(new)
                action_rows.append(action)
                trajectory_rows.append(trajectory_offset + episode)
                fallbacks += int(diagnostics.used_zero_fallback)
                _, _, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "group": name,
                    "trajectory": trajectory_offset + episode,
                    "seed": seed,
                    "steps": env.step_count,
                    "completed": bool(info["completed"]),
                    "constraint_success": bool(info["constraint_success"]),
                    "distance_failure": bool(info["distance_failure"]),
                    "fallbacks": fallbacks,
                    "command_time_median_s": float(np.median(command_times)),
                }
            )
            print(json.dumps(records[-1]))
    finally:
        env.close()
    return (
        {
            "observation_v1": np.asarray(old_rows, dtype=np.float32),
            "observation_v2": np.asarray(new_rows, dtype=np.float32),
            "action": np.asarray(action_rows, dtype=np.float32),
            "trajectory": np.asarray(trajectory_rows, dtype=np.int64),
        },
        records,
    )


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error = prediction - truth
    saturated = np.any(np.abs(truth) >= 0.95, axis=1)
    result = {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "axis_mae": np.mean(np.abs(error), axis=0).tolist(),
        "samples": int(len(truth)),
        "teacher_saturation_sample_fraction": float(np.mean(saturated)),
    }
    result["saturated_sample_mae"] = (
        float(np.mean(np.abs(error[saturated]))) if np.any(saturated) else None
    )
    return result


def _fit(
    observations: np.ndarray,
    actions: np.ndarray,
    train_mask: np.ndarray,
    output: Path,
    seed: int,
) -> tuple[ActionRegressor, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = observations[train_mask]
    valid_x = observations[~train_mask]
    train_y = actions[train_mask]
    valid_y = actions[~train_mask]
    mean_x = np.mean(train_x, axis=0)
    std_x = np.maximum(np.std(train_x, axis=0), 1.0e-4)
    x_train = torch.as_tensor((train_x - mean_x) / std_x, device=device)
    y_train = torch.as_tensor(train_y, device=device)
    x_valid = torch.as_tensor((valid_x - mean_x) / std_x, device=device)
    model = ActionRegressor(observations.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_loss = float("inf")
    best_state = None
    rng = np.random.default_rng(seed)
    batch_size = 256
    stale = 0
    for epoch in range(400):
        order = rng.permutation(len(train_x))
        model.train()
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start : start + batch_size], device=device)
            prediction = model(x_train[indices])
            loss = torch.mean((prediction - y_train[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(torch.mean((model(x_valid) - torch.as_tensor(valid_y, device=device)) ** 2))
        if valid_loss < best_loss - 1.0e-7:
            best_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 40:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prediction = model(x_train).cpu().numpy()
        valid_prediction = model(x_valid).cpu().numpy()
    torch.save(
        {
            "state_dict": best_state,
            "input_mean": mean_x,
            "input_std": std_x,
            "input_size": observations.shape[1],
        },
        output,
    )
    return model, mean_x, std_x, {
        "device": str(device),
        "epochs": epoch + 1,
        "train": _metrics(train_prediction, train_y),
        "holdout": _metrics(valid_prediction, valid_y),
    }


def _evaluate_clone(
    model: ActionRegressor,
    mean_x: np.ndarray,
    std_x: np.ndarray,
    *,
    use_v2: bool,
    config: SE3RendezvousConfig,
    seeds: list[int],
) -> list[dict]:
    device = next(model.parameters()).device
    env = SE3RendezvousEnv(config)
    records = []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            terminated = truncated = False
            action_norms = []
            while not (terminated or truncated):
                old, new = _observations(env)
                observation = new if use_v2 else old
                tensor = torch.as_tensor(
                    ((observation - mean_x) / std_x)[None], device=device
                )
                with torch.no_grad():
                    action = model(tensor).cpu().numpy()[0]
                action_norms.append(float(np.linalg.norm(action)))
                _, _, terminated, truncated, info = env.step(action)
            records.append(
                {
                    "seed": seed,
                    "steps": env.step_count,
                    "completed": bool(info["completed"]),
                    "constraint_success": bool(info["constraint_success"]),
                    "distance_failure": bool(info["distance_failure"]),
                    "time_failure": bool(info["time_failure"]),
                    "final_position_error_m": float(info["position_error_m"]),
                    "mean_action_l2": float(np.mean(action_norms)),
                }
            )
    finally:
        env.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=26_090_000)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("logs/phase2_v2_1_mpc_observation_diagnostic"),
    )
    args = parser.parse_args()
    if args.episodes_per_group < 2:
        raise ValueError("at least two episodes per group are needed for holdout")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    stage_a = evaluation_environment_config(
        phase2=True, phase2_stage_a_only=True
    )
    nominal = evaluation_environment_config(phase2=True)
    groups = (
        ("stage_a", stage_a),
        ("nominal_interior", replace(nominal, phase2_near_boundary_probability=0.0)),
        ("nominal_near_boundary", replace(nominal, phase2_near_boundary_probability=1.0)),
    )
    parts = []
    teacher_records = []
    offset = 0
    for group_index, (name, config) in enumerate(groups):
        data, records = _collect_group(
            name=name,
            config=config,
            episodes=args.episodes_per_group,
            base_seed=args.seed + group_index * 10_000,
            trajectory_offset=offset,
        )
        parts.append(data)
        teacher_records.extend(records)
        offset += args.episodes_per_group
    dataset = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in parts[0]
    }
    np.savez_compressed(args.output_directory / "teacher_dataset.npz", **dataset)
    holdout_trajectories = np.arange(
        args.episodes_per_group - 1, offset, args.episodes_per_group
    )
    train_mask = ~np.isin(dataset["trajectory"], holdout_trajectories)
    v1_model, v1_mean, v1_std, v1_result = _fit(
        dataset["observation_v1"],
        dataset["action"],
        train_mask,
        args.output_directory / "observation_v1_regressor.pt",
        args.seed,
    )
    v2_model, v2_mean, v2_std, v2_result = _fit(
        dataset["observation_v2"],
        dataset["action"],
        train_mask,
        args.output_directory / "observation_v2_1_regressor.pt",
        args.seed,
    )
    clone_seeds = [args.seed + 90_000 + index for index in range(3)]
    v1_closed_loop = _evaluate_clone(
        v1_model,
        v1_mean,
        v1_std,
        use_v2=False,
        config=stage_a,
        seeds=clone_seeds,
    )
    v2_closed_loop = _evaluate_clone(
        v2_model,
        v2_mean,
        v2_std,
        use_v2=True,
        config=stage_a,
        seeds=clone_seeds,
    )
    result = {
        "schema_version": 1,
        "episodes_per_group": args.episodes_per_group,
        "teacher": teacher_records,
        "teacher_all_completed": all(row["completed"] for row in teacher_records),
        "teacher_all_constraint_success": all(
            row["constraint_success"] for row in teacher_records
        ),
        "teacher_total_fallbacks": sum(row["fallbacks"] for row in teacher_records),
        "dataset_samples": int(len(dataset["action"])),
        "holdout_trajectories": holdout_trajectories.tolist(),
        "observation_v1": {**v1_result, "closed_loop_stage_a": v1_closed_loop},
        "observation_v2_1": {**v2_result, "closed_loop_stage_a": v2_closed_loop},
    }
    (args.output_directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
