"""Long-lived diagnostics for the canonical Phase-2 Pure SAC path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from env.phase2_env import Phase2Mode, phase2_environment_config
from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv


class Phase2DiagnosticsCallback(BaseCallback):
    """Record episode outcomes and compact actor/critic health metrics."""

    def __init__(self, output_directory: Path, *, probe_config: SE3RendezvousConfig, interval_steps: int = 25_000, probe_seed: int = 20_289_000) -> None:
        super().__init__(verbose=0)
        self.output_directory = Path(output_directory)
        self.probe_config = probe_config
        self.interval_steps = int(interval_steps)
        self.probe_seed = int(probe_seed)
        self._next_probe = self.interval_steps
        self._probe_observations: np.ndarray | None = None
        self._minimum_gate_distance_by_env: list[float] = []
        self.episodes: list[dict] = []
        self.probes: list[dict] = []

    def _on_training_start(self) -> None:
        env = SE3RendezvousEnv(self.probe_config)
        try:
            self._probe_observations = np.asarray([env.reset(seed=self.probe_seed + i)[0] for i in range(32)], dtype=np.float32)
        finally:
            env.close()
        self._minimum_gate_distance_by_env = [
            float("inf") for _ in range(self.training_env.num_envs)
        ]

    @staticmethod
    def _stats(value: np.ndarray) -> dict[str, float]:
        array = np.asarray(value, dtype=np.float64)
        return {"mean": float(np.mean(array)), "std": float(np.std(array))}

    def _probe(self) -> None:
        assert self._probe_observations is not None
        observations = torch.as_tensor(self._probe_observations, device=self.model.device)
        with torch.no_grad():
            means, log_stds, _ = self.model.actor.get_action_dist_params(observations)
            actions = torch.tanh(means)
            q_values = self.model.critic(observations, actions)
        alpha = float(torch.exp(self.model.log_ent_coef).item()) if self.model.log_ent_coef is not None else float(self.model.ent_coef_tensor.item())
        record = {
            "training_step": int(self.num_timesteps),
            "actor_action_std": self._stats(torch.exp(log_stds).cpu().numpy()),
            "actor_action_saturation_fraction": float(np.mean(np.abs(actions.cpu().numpy()) >= 0.95)),
            "entropy_coefficient": alpha,
            "critic_q1": self._stats(q_values[0].cpu().numpy()),
            "critic_q2": self._stats(q_values[1].cpu().numpy()),
            "short_horizon_bootstrapped_value_probe": (
                self._short_horizon_bootstrapped_value_probe(alpha)
            ),
        }
        self.probes.append(record)
        for key in ("entropy_coefficient", "actor_action_saturation_fraction"):
            self.logger.record(f"phase2/{key}", record[key])

    def _short_horizon_bootstrapped_value_probe(
        self, alpha: float, *, episodes: int = 4, horizon_steps: int = 64
    ) -> dict[str, object]:
        """Compare Q with a short rollout that bootstraps at its horizon.

        Sampling uses an isolated Torch RNG state so diagnostics do not perturb
        the training trajectory.  This is not an independent Monte Carlo return.
        """

        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        records = []
        try:
            torch.manual_seed(self.probe_seed + int(self.num_timesteps))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.probe_seed + int(self.num_timesteps))
            for episode in range(episodes):
                env = SE3RendezvousEnv(self.probe_config)
                try:
                    observation, _ = env.reset(seed=self.probe_seed + episode)
                    discount = 1.0
                    soft_return = 0.0
                    one_step_target = np.nan
                    first_reward = np.nan
                    first_entropy = np.nan
                    first_q1 = np.nan
                    first_q2 = np.nan
                    terminated = truncated = False
                    for step in range(horizon_steps):
                        obs_tensor = torch.as_tensor(
                            observation[None, :], device=self.model.device
                        )
                        with torch.no_grad():
                            action_tensor, log_prob = self.model.actor.action_log_prob(
                                obs_tensor
                            )
                            q1, q2 = self.model.critic(obs_tensor, action_tensor)
                        action = action_tensor[0].cpu().numpy()
                        next_observation, reward, terminated, truncated, _ = env.step(
                            action
                        )
                        reward = float(reward)
                        log_prob_value = float(log_prob.item())
                        soft_return += discount * (
                            reward if step == 0 else reward - alpha * log_prob_value
                        )
                        if step == 0:
                            first_reward = reward
                            first_q1 = float(q1.item())
                            first_q2 = float(q2.item())
                        discount *= float(self.model.gamma)
                        if terminated or truncated:
                            if step == 0:
                                one_step_target = reward
                            break
                        observation = next_observation
                        next_tensor = torch.as_tensor(
                            observation[None, :], device=self.model.device
                        )
                        with torch.no_grad():
                            next_action, next_log_prob = (
                                self.model.actor.action_log_prob(next_tensor)
                            )
                            target_q1, target_q2 = self.model.critic_target(
                                next_tensor, next_action
                            )
                            next_soft_value = torch.minimum(
                                target_q1, target_q2
                            ) - alpha * next_log_prob.reshape(-1, 1)
                        if step == 0:
                            first_entropy = float((-alpha * next_log_prob).item())
                            one_step_target = reward + float(
                                self.model.gamma * next_soft_value.item()
                            )
                        if step + 1 == horizon_steps:
                            soft_return += discount * float(next_soft_value.item())
                finally:
                    env.close()
                minimum_q = min(first_q1, first_q2)
                records.append(
                    {
                        "seed": self.probe_seed + episode,
                        "horizon_steps": step + 1,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "environment_reward": first_reward,
                        "next_entropy_contribution": first_entropy,
                        "one_step_soft_target": one_step_target,
                        "critic_q1": first_q1,
                        "critic_q2": first_q2,
                        "short_soft_bootstrapped_return": soft_return,
                        "minimum_q_minus_bootstrapped_return": (
                            minimum_q - soft_return
                        ),
                    }
                )
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return {
            "episodes": episodes,
            "horizon_steps": horizon_steps,
            "records": records,
            "estimate_kind": "short_horizon_soft_return_with_terminal_bootstrap",
            "is_independent_monte_carlo_ground_truth": False,
            "mean_minimum_q_minus_bootstrapped_return": float(
                np.mean(
                    [
                        record["minimum_q_minus_bootstrapped_return"]
                        for record in records
                    ]
                )
            ),
        }

    def _write(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "phase2_diagnostics.json").write_text(
            json.dumps({"schema_version": 4, "episodes": self.episodes, "probes": self.probes}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _on_step(self) -> bool:
        for index, (info, done) in enumerate(
            zip(self.locals.get("infos", []), self.locals.get("dones", []))
        ):
            gate_distance = float(info.get("gate_position_error_m", np.inf))
            self._minimum_gate_distance_by_env[index] = min(
                self._minimum_gate_distance_by_env[index], gate_distance
            )
            if done:
                self.episodes.append({
                    "training_step": int(self.num_timesteps),
                    "completed": bool(info.get("completed", False)),
                    "constraint_success": bool(info.get("constraint_success", False)),
                    "position_error_m": float(info.get("position_error_m", np.nan)),
                    "mission_phase": int(info.get("mission_phase", 0)),
                    "gate_reached": bool(info.get("gate_reached", False)),
                    "minimum_gate_position_error_m": float(
                        self._minimum_gate_distance_by_env[index]
                    ),
                    "terminal_gate_position_error_m": gate_distance,
                    "terminal_attitude_error_rad": float(
                        info.get("attitude_error_rad", np.nan)
                    ),
                    "terminal_speed_m_s": float(
                        info.get("total_speed_m_s", np.nan)
                    ),
                    "terminal_angular_velocity_rad_s": float(
                        info.get("angular_velocity_error_rad_s", np.nan)
                    ),
                    "phase1_speed_failure": bool(
                        info.get("phase1_speed_failure", False)
                    ),
                    "premature_entry_failure": bool(
                        info.get("premature_entry_failure", False)
                    ),
                    "terminal_constraint_failure": bool(
                        info.get("terminal_constraint_failure", False)
                    ),
                    "distance_failure": bool(info.get("distance_failure", False)),
                    "time_failure": bool(info.get("time_failure", False)),
                    "phase1_curriculum_difficulty": float(
                        info.get("phase1_curriculum_difficulty", 1.0)
                    ),
                    "phase1_curriculum_frontier": float(
                        info.get("phase1_curriculum_frontier", 1.0)
                    ),
                    "phase1_curriculum_full_sample": bool(
                        info.get("phase1_curriculum_full_sample", True)
                    ),
                    "violation_steps": {name: int(info.get(f"{name}_violation_steps", 0)) for name in ("corridor", "fov", "total_speed", "closing_speed")},
                })
                self._minimum_gate_distance_by_env[index] = float("inf")
        if self.num_timesteps >= self._next_probe:
            self._probe()
            self._next_probe += self.interval_steps
            self._write()
        return True

    def _on_training_end(self) -> None:
        self._write()


class DeterministicPhase2EvalCallback(BaseCallback):
    """Periodic fixed-seed evaluation without curriculum or early-stop logic."""

    def __init__(self, output_directory: Path, *, mode: Phase2Mode, eval_freq_steps: int = 50_000, eval_steps: tuple[int, ...] | None = None, episodes: int = 10, base_seed: int = 20_288_000) -> None:
        super().__init__(verbose=0)
        self.output_directory = Path(output_directory)
        self.mode = mode
        self.eval_freq_steps = int(eval_freq_steps)
        self.eval_steps = eval_steps
        self.episodes = int(episodes)
        self.base_seed = int(base_seed)
        self._eval_index = 0
        self._next_eval = (
            self.eval_steps[0] if self.eval_steps else self.eval_freq_steps
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        from eval.evaluate_policy import evaluate_model

        result = evaluate_model(
            self.model,
            phase2_environment_config(self.mode),
            episodes=self.episodes,
            seed=self.base_seed,
            deterministic=True,
        )
        result["training_step"] = int(self.num_timesteps)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.output_directory / f"evaluation_{self.num_timesteps:09d}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.record("phase2_eval/completion_rate", result["rates"]["episode_completion"])
        self.logger.record("phase2_eval/constraint_success_rate", result["rates"]["constraint_success"])
        if self.eval_steps:
            self._eval_index += 1
            self._next_eval = (
                self.eval_steps[self._eval_index]
                if self._eval_index < len(self.eval_steps)
                else float("inf")
            )
        else:
            self._next_eval += self.eval_freq_steps
        return True


__all__ = ["DeterministicPhase2EvalCallback", "Phase2DiagnosticsCallback"]
