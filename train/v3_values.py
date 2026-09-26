"""V3 fixed-policy values and the one-way arbiter (M3 / M4).

Two values, each a 5-head ensemble fitted by supervised regression on complete
discounted decision-level returns (the simulator is deterministic and every
trajectory is complete, so no TD is needed):

    V_L(s) ~ return of "the learned policy continues from s to the end"
    V_B(s) ~ return of "Pure MPC takes over at s and flies to the end"

Diversity comes from bootstrap resampling *by episode* (states inside one
episode are strongly correlated, so resampling states would understate the
spread) plus independent initialisation. Each head stops early on its own
out-of-bag episodes.

The baseline value never sees the learned branch's private state
(``task_state`` and ``applied_direction`` are zeroed at its input): Pure MPC's
future does not depend on them, and after a handoff they are stale.

Arbiter (docs/V3_METHOD_EXECUTION_ORDER_20260924.md, M4), z = 1:

    decision 0:  learned iff  mu_L - z*sd_L > mu_B + z*sd_B
    on learned:  hand back iff mu_B - z*sd_B > mu_L + z*sd_L
    on baseline: stay for the rest of the episode
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

DECISION_GAMMA = 0.99
HEADS = 5
HIDDEN = (256, 256, 256, 256)  # same width as the SAC critic
BASELINE_MASKED_BLOCKS = ("task_state", "applied_direction")


def discounted_returns(rewards: Sequence[float], gamma: float = DECISION_GAMMA) -> np.ndarray:
    """G_t = sum_{j>=t} gamma^(j-t) r_j, to the end of the episode (terminal at the end)."""

    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        out[index] = running
    return out


def baseline_mask(slices: dict[str, slice], dimension: int) -> np.ndarray:
    mask = np.ones(dimension, dtype=np.float32)
    for name in BASELINE_MASKED_BLOCKS:
        if name in slices:
            mask[slices[name]] = 0.0
    return mask


def _mlp(inputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = inputs
    for hidden in HIDDEN:
        layers += [nn.Linear(width, hidden), nn.ReLU()]
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


@dataclass
class ValueEnsemble:
    """Standardised-input, standardised-target MLP ensemble with an input mask."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    input_mask: np.ndarray
    target_mean: float
    target_scale: float
    heads: list[nn.Sequential]
    meta: dict[str, Any] = field(default_factory=dict)

    def _inputs(self, observations: np.ndarray) -> torch.Tensor:
        x = np.asarray(observations, dtype=np.float32).reshape(-1, self.input_mean.size)
        x = (x * self.input_mask - self.input_mean) / self.input_scale
        return torch.as_tensor(x, dtype=torch.float32)

    @torch.no_grad()
    def head_values(self, observations: np.ndarray) -> np.ndarray:
        x = self._inputs(observations)
        values = torch.stack([head(x).squeeze(-1) for head in self.heads], dim=0)
        return values.numpy().astype(np.float64) * self.target_scale + self.target_mean

    def mean_std(self, observation: np.ndarray) -> tuple[float, float]:
        values = self.head_values(observation)[:, 0]
        return float(values.mean()), float(values.std(ddof=0))

    def save(self, path: Path) -> str:
        payload = {
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "input_mask": self.input_mask,
            "target_mean": self.target_mean,
            "target_scale": self.target_scale,
            "heads": [head.state_dict() for head in self.heads],
            "meta": self.meta,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        Path(path).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "ValueEnsemble":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        heads = []
        for state in payload["heads"]:
            head = _mlp(int(payload["input_mean"].size))
            head.load_state_dict(state)
            head.eval()
            heads.append(head)
        return cls(
            input_mean=payload["input_mean"],
            input_scale=payload["input_scale"],
            input_mask=payload["input_mask"],
            target_mean=float(payload["target_mean"]),
            target_scale=float(payload["target_scale"]),
            heads=heads,
            meta=dict(payload["meta"]),
        )


def fit_ensemble(
    observations: np.ndarray,
    returns: np.ndarray,
    episode_ids: np.ndarray,
    *,
    input_mask: np.ndarray,
    seed: int,
    heads: int = HEADS,
    max_epochs: int = 400,
    patience: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
) -> ValueEnsemble:
    """Fit ``heads`` MLPs, each on an episode bootstrap, early-stopped on its out-of-bag episodes."""

    x_all = np.asarray(observations, dtype=np.float32) * input_mask
    y_all = np.asarray(returns, dtype=np.float64)
    episodes = np.unique(episode_ids)
    if episodes.size < 4:
        raise ValueError("need at least four episodes to bootstrap a value")
    input_mean = x_all.mean(axis=0)
    input_scale = np.maximum(x_all.std(axis=0), 1.0e-3).astype(np.float32)
    input_mean = (input_mean * (input_mask > 0)).astype(np.float32)
    target_mean = float(y_all.mean())
    target_scale = float(max(y_all.std(), 1.0e-6))
    x_norm = (x_all - input_mean) / input_scale
    y_norm = (y_all - target_mean) / target_scale
    rng = np.random.default_rng(seed)
    fitted: list[nn.Sequential] = []
    head_meta = []
    for head_index in range(heads):
        torch.manual_seed(seed * 1000 + head_index)
        drawn = rng.choice(episodes, size=episodes.size, replace=True)
        out_of_bag = np.setdiff1d(episodes, drawn)
        if out_of_bag.size == 0:  # astronomically unlikely; keep one episode out
            out_of_bag = episodes[:1]
            drawn = drawn[drawn != out_of_bag[0]]
        train_index = np.concatenate([np.flatnonzero(episode_ids == e) for e in drawn])
        valid_index = np.concatenate([np.flatnonzero(episode_ids == e) for e in out_of_bag])
        x_train = torch.as_tensor(x_norm[train_index])
        y_train = torch.as_tensor(y_norm[train_index], dtype=torch.float32)
        x_valid = torch.as_tensor(x_norm[valid_index])
        y_valid = torch.as_tensor(y_norm[valid_index], dtype=torch.float32)
        model = _mlp(x_all.shape[1])
        optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best = (float("inf"), None, 0)
        stale = 0
        order_rng = np.random.default_rng(seed * 1000 + head_index)
        for epoch in range(max_epochs):
            model.train()
            order = order_rng.permutation(train_index.size)
            for start in range(0, order.size, batch_size):
                batch = order[start : start + batch_size]
                optimiser.zero_grad()
                loss = torch.mean((model(x_train[batch]).squeeze(-1) - y_train[batch]) ** 2)
                loss.backward()
                optimiser.step()
            model.eval()
            with torch.no_grad():
                valid = float(torch.mean((model(x_valid).squeeze(-1) - y_valid) ** 2))
            if valid < best[0] - 1.0e-6:
                best = (valid, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        assert best[1] is not None
        model.load_state_dict(best[1])
        model.eval()
        fitted.append(model)
        head_meta.append(
            {
                "bootstrap_episodes": int(np.unique(drawn).size),
                "out_of_bag_episodes": int(out_of_bag.size),
                "best_epoch": int(best[2]),
                "out_of_bag_mse_return_units": float(best[0] * target_scale**2),
            }
        )
    return ValueEnsemble(
        input_mean=input_mean,
        input_scale=input_scale,
        input_mask=np.asarray(input_mask, dtype=np.float32),
        target_mean=target_mean,
        target_scale=target_scale,
        heads=fitted,
        meta={"heads": head_meta, "fit_seed": seed, "hidden": list(HIDDEN)},
    )


@dataclass(frozen=True)
class ArbiterDecision:
    branch: str
    mu_learned: float
    sd_learned: float
    mu_baseline: float
    sd_baseline: float


class OneWayArbiter:
    """Initial choice plus a one-way learned -> baseline handback (M4)."""

    def __init__(self, learned: ValueEnsemble, baseline: ValueEnsemble, *, z: float = 1.0) -> None:
        self.learned = learned
        self.baseline = baseline
        self.z = float(z)

    def decide(self, observation: np.ndarray, current_branch: str | None) -> ArbiterDecision:
        mu_l, sd_l = self.learned.mean_std(observation)
        mu_b, sd_b = self.baseline.mean_std(observation)
        return ArbiterDecision(
            one_way_rule(current_branch, mu_l, sd_l, mu_b, sd_b, self.z),
            mu_l,
            sd_l,
            mu_b,
            sd_b,
        )


def one_way_rule(
    current_branch: str | None,
    mu_learned: float,
    sd_learned: float,
    mu_baseline: float,
    sd_baseline: float,
    z: float = 1.0,
) -> str:
    if current_branch == "baseline":
        return "baseline"
    if current_branch is None:
        return (
            "learned"
            if mu_learned - z * sd_learned > mu_baseline + z * sd_baseline
            else "baseline"
        )
    if current_branch != "learned":
        raise ValueError("branch must be None, 'learned' or 'baseline'")
    return (
        "baseline"
        if mu_baseline - z * sd_baseline > mu_learned + z * sd_learned
        else "learned"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
