"""Transparent MPC behavior initialization for the Phase-2 SAC actor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC


def load_teacher_dataset(path: Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"teacher dataset not found: {source}")
    with np.load(source) as archive:
        required = {"observation_v2", "action", "trajectory"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"teacher dataset missing arrays: {sorted(missing)}")
        observations = np.asarray(archive["observation_v2"], dtype=np.float32)
        actions = np.asarray(archive["action"], dtype=np.float32)
        trajectories = np.asarray(archive["trajectory"], dtype=np.int64)
    if observations.ndim != 2 or observations.shape[1] != 23:
        raise ValueError("teacher observations must have shape=(N, 23)")
    if actions.shape != (len(observations), 6):
        raise ValueError("teacher actions must have shape=(N, 6)")
    if trajectories.shape != (len(observations),):
        raise ValueError("teacher trajectories must have shape=(N,)")
    if len(np.unique(trajectories)) < 2:
        raise ValueError("teacher dataset needs at least two trajectories")
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
        raise ValueError("teacher dataset contains non-finite values")
    if np.max(np.abs(actions)) > 1.0 + 1.0e-6:
        raise ValueError("teacher actions exceed normalized action bounds")
    return {
        "observations": observations,
        "actions": actions,
        "trajectories": trajectories,
    }


def trajectory_holdout_mask(trajectories: np.ndarray) -> tuple[np.ndarray, list[int]]:
    unique = np.unique(np.asarray(trajectories, dtype=np.int64))
    holdout = unique[1::2]
    if not len(holdout):
        holdout = unique[-1:]
    train_mask = ~np.isin(trajectories, holdout)
    if not np.any(train_mask) or np.all(train_mask):
        raise ValueError("trajectory split must contain train and holdout samples")
    return train_mask, holdout.tolist()


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction) - np.asarray(target)
    saturated = np.any(np.abs(target) >= 0.95, axis=1)
    return {
        "samples": int(len(target)),
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "axis_mae": np.mean(np.abs(error), axis=0).tolist(),
        "teacher_saturation_sample_fraction": float(np.mean(saturated)),
        "saturated_sample_mae": (
            float(np.mean(np.abs(error[saturated]))) if np.any(saturated) else None
        ),
    }


def initialize_actor_from_mpc(
    model: SAC,
    dataset_path: Path,
    *,
    output_path: Path,
    seed: int,
    maximum_epochs: int = 400,
    patience_epochs: int = 40,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    initial_log_std: float = -2.0,
) -> dict[str, Any]:
    """Fit SAC deterministic actor actions without touching critic or replay."""

    if min(maximum_epochs, patience_epochs, batch_size) <= 0 or learning_rate <= 0:
        raise ValueError("behavior initialization settings must be positive")
    data = load_teacher_dataset(dataset_path)
    observations = data["observations"]
    actions = data["actions"]
    trajectories = data["trajectories"]
    if observations.shape[1:] != model.observation_space.shape:
        raise ValueError("teacher observation shape does not match SAC actor")
    train_mask, holdout_trajectories = trajectory_holdout_mask(trajectories)
    device = model.device
    train_x = torch.as_tensor(observations[train_mask], device=device)
    train_y = torch.as_tensor(actions[train_mask], device=device)
    holdout_x = torch.as_tensor(observations[~train_mask], device=device)
    holdout_y = torch.as_tensor(actions[~train_mask], device=device)
    parameters = [
        *model.actor.latent_pi.parameters(),
        *model.actor.mu.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(maximum_epochs):
        order = rng.permutation(len(train_x))
        model.actor.train()
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(order[start : start + batch_size], device=device)
            latent = model.actor.latent_pi(train_x[indices])
            prediction = torch.tanh(model.actor.mu(latent))
            loss = torch.mean((prediction - train_y[indices]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.actor.eval()
        with torch.no_grad():
            holdout_prediction = torch.tanh(
                model.actor.mu(model.actor.latent_pi(holdout_x))
            )
            holdout_loss = float(torch.mean((holdout_prediction - holdout_y) ** 2))
        if holdout_loss < best_loss - 1.0e-7:
            best_loss = holdout_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.actor.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience_epochs:
            break
    if best_state is None:
        raise RuntimeError("behavior initialization did not produce a finite model")
    model.actor.load_state_dict(best_state)
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(initial_log_std)
    model.actor.eval()
    with torch.no_grad():
        train_prediction = torch.tanh(
            model.actor.mu(model.actor.latent_pi(train_x))
        ).cpu().numpy()
        holdout_prediction = torch.tanh(
            model.actor.mu(model.actor.latent_pi(holdout_x))
        ).cpu().numpy()
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "mpc_actor_mean_behavior_initialization",
        "dataset": str(Path(dataset_path).resolve()),
        "seed": int(seed),
        "device": str(device),
        "epochs": epoch + 1,
        "maximum_epochs": maximum_epochs,
        "patience_epochs": patience_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "total_samples": int(len(observations)),
        "trajectories": np.unique(trajectories).tolist(),
        "holdout_trajectories": holdout_trajectories,
        "train": _metrics(train_prediction, actions[train_mask]),
        "holdout": _metrics(holdout_prediction, actions[~train_mask]),
        "critic_initialized_from_teacher": False,
        "replay_prefilled": False,
        "actor_log_std_pretrained": False,
        "actor_initial_log_std": float(initial_log_std),
        "actor_initial_std": float(np.exp(initial_log_std)),
    }
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
