"""Independent Monte-Carlo calibration check for the Phase-2 Pure SAC critic.

This is the ground-truth probe that `Phase2DiagnosticsCallback.
_short_horizon_bootstrapped_value_probe` explicitly does NOT provide: that probe
bootstraps after 64 steps, and with gamma=0.997 the bootstrap carries
0.997**64 = 0.825 of the estimate, so it is very nearly circular and cannot
detect value bias.

Here we roll out full episodes and compare the critic's own Q(s0, a0) against:

  * hard return   sum_t gamma^t r_t                       (deterministic policy)
  * soft return   r_0 + sum_{t>=1} gamma^t (r_t - alpha*log pi(a_t|s_t))
                                                          (stochastic after a0)

The soft return is the exact quantity SAC's critic is trained to represent, so
``Q(s0,a0) - soft_return`` is the critic's calibration error. Read-only: no
training, no writes outside --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

from eval.evaluate_policy import (
    _manifest_for_model,
    environment_config_from_manifest,
)
from env.se3_rendezvous_env import SE3RendezvousEnv


def _alpha(model: SAC) -> float:
    if getattr(model, "log_ent_coef", None) is not None:
        return float(torch.exp(model.log_ent_coef).item())
    return float(model.ent_coef_tensor.item())


def calibration(
    model: SAC,
    config,
    *,
    episodes: int,
    seed: int,
    gamma: float,
) -> dict[str, Any]:
    alpha = _alpha(model)
    env = SE3RendezvousEnv(config)
    records: list[dict[str, Any]] = []
    try:
        for index in range(episodes):
            observation, info = env.reset(seed=seed + index)
            initial_gate = float(info["gate_position_error_m"])
            tensor = torch.as_tensor(observation[None], dtype=torch.float32)
            first_action, _ = model.predict(observation, deterministic=True)
            with torch.no_grad():
                q_values = model.critic(
                    tensor,
                    torch.as_tensor(
                        np.asarray(first_action)[None], dtype=torch.float32
                    ),
                )
            q0 = float(min(q_values[0].item(), q_values[1].item()))

            # deterministic hard return
            observation, reward, terminated, truncated, info = env.step(first_action)
            hard = float(reward)
            discount = gamma
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                hard += discount * float(reward)
                discount *= gamma
            hard_steps = int(info["step_count"])
            hard_gate = bool(info["gate_reached"])
            hard_min_gate = float(info["gate_position_error_m"])

            # entropy-augmented soft return, stochastic after the first action
            observation, _ = env.reset(seed=seed + index)
            observation, reward, terminated, truncated, info = env.step(first_action)
            soft = float(reward)
            entropy_term = 0.0
            discount = gamma
            while not (terminated or truncated):
                tensor = torch.as_tensor(observation[None], dtype=torch.float32)
                with torch.no_grad():
                    action_tensor, log_prob = model.actor.action_log_prob(tensor)
                log_prob_value = float(log_prob.item())
                observation, reward, terminated, truncated, info = env.step(
                    action_tensor[0].numpy()
                )
                soft += discount * (float(reward) - alpha * log_prob_value)
                entropy_term += discount * (-alpha * log_prob_value)
                discount *= gamma

            records.append(
                {
                    "seed": seed + index,
                    "initial_gate_position_error_m": initial_gate,
                    "critic_q_min_s0": q0,
                    "hard_discounted_return": hard,
                    "soft_discounted_return": soft,
                    "entropy_contribution": entropy_term,
                    "calibration_error": q0 - soft,
                    "deterministic_steps": hard_steps,
                    "deterministic_gate_reached": hard_gate,
                    "deterministic_final_gate_error_m": hard_min_gate,
                }
            )
    finally:
        env.close()

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in records]))

    return {
        "schema_version": 1,
        "probe": "independent_monte_carlo_value_calibration",
        "is_independent_monte_carlo_ground_truth": True,
        "gamma": gamma,
        "entropy_coefficient": alpha,
        "episodes": episodes,
        "base_seed": seed,
        "summary": {
            "critic_q_min_s0": mean("critic_q_min_s0"),
            "hard_discounted_return": mean("hard_discounted_return"),
            "soft_discounted_return": mean("soft_discounted_return"),
            "entropy_contribution": mean("entropy_contribution"),
            "calibration_error": mean("calibration_error"),
            "deterministic_survival_s": float(
                np.mean([r["deterministic_steps"] for r in records]) * 0.1
            ),
            "deterministic_gate_rate": float(
                np.mean([r["deterministic_gate_reached"] for r in records])
            ),
        },
        "episode_records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte-Carlo calibration of the Pure SAC critic"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_290_000)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.manifest or _manifest_for_model(args.model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Use the run's OWN manifest config, not the current canonical config, so
    # older observation schemas stay loadable.
    config = environment_config_from_manifest(manifest)
    model = SAC.load(args.model, device=args.device)
    result = calibration(
        model, config, episodes=args.episodes, seed=args.seed, gamma=args.gamma
    )
    result["model"] = str(args.model.resolve())
    result["training_manifest"] = str(manifest_path.resolve())
    result["observation_schema"] = config.phase2_observation_schema
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    s = result["summary"]
    print(
        json.dumps(
            {
                "model": args.model.name,
                "schema": result["observation_schema"],
                "alpha": result["entropy_coefficient"],
                "critic_Q_s0": round(s["critic_q_min_s0"], 3),
                "soft_MC_return": round(s["soft_discounted_return"], 3),
                "calibration_error": round(s["calibration_error"], 3),
                "hard_MC_return": round(s["hard_discounted_return"], 3),
                "survival_s": round(s["deterministic_survival_s"], 1),
                "gate_rate": s["deterministic_gate_rate"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
