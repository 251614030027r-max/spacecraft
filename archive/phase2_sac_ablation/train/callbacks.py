"""Low-overhead training and fixed-task diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv


class EnvironmentDiagnosticsCallback(BaseCallback):
    def __init__(self, log_interval_steps: int = 5_000) -> None:
        super().__init__(verbose=0)
        self.log_interval_steps = int(log_interval_steps)
        self._next_log = self.log_interval_steps
        self._episodes = 0
        self._completed = 0
        self._distance_failures = 0
        self._timeouts = 0

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", []), self.locals.get("dones", [])
        ):
            if not done:
                continue
            self._episodes += 1
            self._completed += int(bool(info.get("completed", False)))
            self._distance_failures += int(
                bool(info.get("distance_failure", False))
            )
            self._timeouts += int(bool(info.get("time_failure", False)))
        if self.num_timesteps >= self._next_log:
            denominator = max(1, self._episodes)
            self.logger.record("baseline/episodes", self._episodes)
            self.logger.record(
                "baseline/completion_rate", self._completed / denominator
            )
            self.logger.record(
                "baseline/distance_failure_rate",
                self._distance_failures / denominator,
            )
            self.logger.record("baseline/timeout_rate", self._timeouts / denominator)
            self._next_log += self.log_interval_steps
        return True


class Phase2LearningDiagnosticsCallback(BaseCallback):
    """Record replay coverage and actor/critic behavior on fixed Stage-A states."""

    def __init__(
        self,
        *,
        environment_config: SE3RendezvousConfig,
        output_directory: Path,
        interval_steps: int = 25_000,
        replay_sample_size: int = 4_096,
        probe_states: int = 32,
        probe_seed: int = 20_289_000,
    ) -> None:
        super().__init__(verbose=0)
        if min(interval_steps, replay_sample_size, probe_states) <= 0:
            raise ValueError("Phase-2 diagnostic sizes must be positive")
        if not environment_config.phase2_enabled:
            raise ValueError("Phase-2 diagnostics require a Phase-2 environment")
        self.environment_config = environment_config
        self.output_directory = Path(output_directory)
        self.interval_steps = int(interval_steps)
        self.replay_sample_size = int(replay_sample_size)
        self.probe_states = int(probe_states)
        self.probe_seed = int(probe_seed)
        self._next_log = self.interval_steps
        self._probe_observations: np.ndarray | None = None
        self.records: list[dict] = []

    @staticmethod
    def _summary(values: np.ndarray) -> dict[str, float]:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        return {
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "p05": float(np.percentile(flat, 5)),
            "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)),
        }

    @staticmethod
    def _inverse_softsign(values: np.ndarray, limit: float) -> np.ndarray:
        clipped = np.clip(values, -limit + 1.0e-6, limit - 1.0e-6)
        return limit * clipped / (limit - np.abs(clipped))

    def _on_training_start(self) -> None:
        env = SE3RendezvousEnv(self.environment_config)
        observations = []
        try:
            for index in range(self.probe_states):
                observation, _ = env.reset(seed=self.probe_seed + index)
                observations.append(observation)
        finally:
            env.close()
        self._probe_observations = np.asarray(observations, dtype=np.float32)

    def _probe(self) -> dict:
        assert self._probe_observations is not None
        observations = torch.as_tensor(
            self._probe_observations, device=self.model.device
        )
        with torch.no_grad():
            means, log_stds, _ = self.model.actor.get_action_dist_params(observations)
            actions = torch.tanh(means)
            q_values = self.model.critic(observations, actions)
        action_np = actions.detach().cpu().numpy()
        mean_np = means.detach().cpu().numpy()
        std_np = torch.exp(log_stds).detach().cpu().numpy()
        q1 = q_values[0].detach().cpu().numpy()
        q2 = q_values[1].detach().cpu().numpy()
        return {
            "deterministic_action": self._summary(action_np),
            "actor_pre_tanh_mean": self._summary(mean_np),
            "actor_std": self._summary(std_np),
            "action_saturation_fraction": float(np.mean(np.abs(action_np) >= 0.95)),
            "q1": self._summary(q1),
            "q2": self._summary(q2),
            "twin_q_disagreement": self._summary(np.abs(q1 - q2)),
        }

    def _replay(self) -> dict:
        replay = self.model.replay_buffer
        size = int(replay.size())
        count = min(size, self.replay_sample_size)
        rng = np.random.default_rng(self.probe_seed + self.num_timesteps)
        indices = rng.choice(size, size=count, replace=False)
        observations = np.asarray(replay.observations[indices, 0], dtype=np.float64)
        actions = np.asarray(replay.actions[indices, 0], dtype=np.float64)
        rewards = np.asarray(replay.rewards[indices, 0], dtype=np.float64)
        dones = np.asarray(replay.dones[indices, 0], dtype=np.float64)
        limit = self.environment_config.observation_softsign_limit
        position_raw = self._inverse_softsign(observations[:, 3:6], limit)
        position_error = np.linalg.norm(
            position_raw * self.environment_config.observation_distance_scale_m,
            axis=1,
        )
        margins = observations[:, 19:23]
        return {
            "buffer_size": size,
            "sample_size": count,
            "position_error_m": self._summary(position_error),
            "near_goal_fraction_lt_1m": float(np.mean(position_error < 1.0)),
            "near_goal_fraction_lt_0_5m": float(np.mean(position_error < 0.5)),
            "constraints_satisfied_fraction": float(np.mean(np.all(margins >= 0.0, axis=1))),
            "corridor_violation_fraction": float(np.mean(margins[:, 0] < 0.0)),
            "fov_violation_fraction": float(np.mean(margins[:, 1] < 0.0)),
            "total_speed_violation_fraction": float(np.mean(margins[:, 2] < 0.0)),
            "closing_speed_violation_fraction": float(np.mean(margins[:, 3] < 0.0)),
            "action_l2": self._summary(np.linalg.norm(actions, axis=1)),
            "action_saturation_fraction": float(np.mean(np.abs(actions) >= 0.95)),
            "terminal_transition_fraction": float(np.mean(dones > 0.5)),
            "reward": self._summary(rewards),
        }

    def _write(self) -> None:
        path = self.output_directory / "phase2_learning_diagnostics.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "observation_schema": self.environment_config.phase2_observation_schema,
                    "probe_seed": self.probe_seed,
                    "probe_states": self.probe_states,
                    "records": self.records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_log:
            return True
        replay_size = int(self.model.replay_buffer.size())
        if replay_size > 0:
            record = {
                "training_step": int(self.num_timesteps),
                "probe": self._probe(),
                "replay": self._replay(),
            }
            self.records.append(record)
            self._write()
            self.logger.record(
                "phase2_probe/q_disagreement_mean",
                record["probe"]["twin_q_disagreement"]["mean"],
            )
            self.logger.record(
                "phase2_replay/constraints_satisfied_fraction",
                record["replay"]["constraints_satisfied_fraction"],
            )
            self.logger.record(
                "phase2_replay/near_goal_fraction_lt_1m",
                record["replay"]["near_goal_fraction_lt_1m"],
            )
        self._next_log += self.interval_steps
        return True


class FixedTaskDiagnosticCallback(BaseCallback):
    """Evaluate one fixed easy task every 50k steps and stop on sustained collapse."""

    def __init__(
        self,
        *,
        environment_config: SE3RendezvousConfig,
        output_directory: Path,
        eval_freq_steps: int = 50_000,
        episodes: int = 10,
        base_seed: int = 20_288_000,
        required_failure_windows: int = 2,
        diagnostic_name: str = "fixed_simple",
        first_eval_step: int | None = None,
    ) -> None:
        super().__init__(verbose=0)
        if min(eval_freq_steps, episodes, required_failure_windows) <= 0:
            raise ValueError("diagnostic intervals and counts must be positive")
        if environment_config.curriculum_enabled:
            raise ValueError("fixed-task diagnostic evaluation must disable curriculum")
        self.environment_config = environment_config
        self.output_directory = Path(output_directory)
        self.eval_freq_steps = int(eval_freq_steps)
        self.episodes = int(episodes)
        self.base_seed = int(base_seed)
        self.required_failure_windows = int(required_failure_windows)
        if not diagnostic_name.replace("_", "").isalnum():
            raise ValueError("diagnostic_name must be alphanumeric with underscores")
        self.diagnostic_name = diagnostic_name
        self._next_eval_step = (
            self.eval_freq_steps
            if first_eval_step is None
            else int(first_eval_step)
        )
        if self._next_eval_step <= 0:
            raise ValueError("first_eval_step must be positive")
        self._consecutive_failure_windows = 0
        self.records: list[dict] = []
        self.stopped_early = False
        self.stop_reason: str | None = None

    def _write_status(self) -> None:
        value = {
            "schema_version": 1,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "eval_freq_steps": self.eval_freq_steps,
            "first_eval_step": self._next_eval_step if not self.records else self.records[0]["training_step"],
            "episodes_per_check": self.episodes,
            "base_seed": self.base_seed,
            "required_failure_windows": self.required_failure_windows,
            "consecutive_failure_windows": self._consecutive_failure_windows,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "checks": self.records,
        }
        (self.output_directory / f"{self.diagnostic_name}_diagnostic_status.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _evaluate(self, training_step: int) -> dict:
        env = SE3RendezvousEnv(self.environment_config)
        episode_records: list[dict] = []
        try:
            for episode in range(self.episodes):
                observation, reset_info = env.reset(seed=self.base_seed + episode)
                episode_return = 0.0
                initial_position_error = float(reset_info["position_error_m"])
                minimum_position_error = initial_position_error
                minimum_position_time_s = 0.0
                maximum_position_error = initial_position_error
                minimum_margins = {
                    key: float(reset_info[key])
                    for key in (
                        "corridor_lateral_margin_m",
                        "fov_margin_rad",
                        "total_speed_margin_m_s",
                        "closing_speed_margin_m_s",
                    )
                    if key in reset_info
                }
                first_violation_time_s: float | None = None
                action_l1_sum = 0.0
                action_l2_sum = 0.0
                saturated_components = 0
                action_components = 0
                terminated = truncated = False
                info: dict = {}
                while not (terminated or truncated):
                    action, _ = self.model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += float(reward)
                    action_array = np.asarray(action, dtype=np.float64)
                    action_l1_sum += float(np.linalg.norm(action_array, ord=1))
                    action_l2_sum += float(np.linalg.norm(action_array))
                    saturated_components += int(np.sum(np.abs(action_array) >= 0.95))
                    action_components += int(action_array.size)
                    position_error = float(info["position_error_m"])
                    if position_error < minimum_position_error:
                        minimum_position_error = position_error
                        minimum_position_time_s = float(info["time_seconds"])
                    maximum_position_error = max(maximum_position_error, position_error)
                    for key in minimum_margins:
                        minimum_margins[key] = min(
                            minimum_margins[key], float(info[key])
                        )
                    if (
                        first_violation_time_s is None
                        and minimum_margins
                        and any(value < 0.0 for value in minimum_margins.values())
                    ):
                        first_violation_time_s = float(info["time_seconds"])
                if bool(info["completed"]):
                    behavior_class = "completed"
                elif minimum_position_error < 0.5:
                    behavior_class = "near_goal_then_failed"
                elif minimum_position_error < 0.5 * initial_position_error:
                    behavior_class = "approached_then_failed"
                elif first_violation_time_s is not None:
                    behavior_class = "constraint_loss_without_close_approach"
                else:
                    behavior_class = "no_material_approach"
                episode_records.append(
                    {
                        "episode": episode,
                        "seed": self.base_seed + episode,
                        "return": episode_return,
                        "steps": env.step_count,
                        "completed": bool(info["completed"]),
                        "distance_failure": bool(info["distance_failure"]),
                        "time_failure": bool(info["time_failure"]),
                        "joint_success": bool(info["joint_success"]),
                        "initial_position_error_m": initial_position_error,
                        "minimum_position_error_m": minimum_position_error,
                        "minimum_position_time_s": minimum_position_time_s,
                        "maximum_position_error_m": maximum_position_error,
                        "final_position_error_m": float(info["position_error_m"]),
                        "first_violation_time_s": first_violation_time_s,
                        "minimum_margins": minimum_margins,
                        "mean_action_l1": action_l1_sum / max(1, env.step_count),
                        "mean_action_l2": action_l2_sum / max(1, env.step_count),
                        "action_saturation_fraction": saturated_components
                        / max(1, action_components),
                        "behavior_class": behavior_class,
                        **(
                            {"constraint_success": bool(info["constraint_success"])}
                            if "constraint_success" in info
                            else {}
                        ),
                    }
                )
        finally:
            env.close()
        rates = {
            key: mean(float(record[key]) for record in episode_records)
            for key in (
                "completed",
                "joint_success",
                "distance_failure",
                "time_failure",
            )
        }
        if episode_records and "constraint_success" in episode_records[0]:
            rates["constraint_success"] = mean(
                float(record["constraint_success"])
                for record in episode_records
            )
        result = {
            "schema_version": 1,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_step": training_step,
            "episodes": self.episodes,
            "base_seed": self.base_seed,
            "environment": asdict(self.environment_config),
            "rates": rates,
            "mean_return": mean(record["return"] for record in episode_records),
            "episode_records": episode_records,
        }
        path = self.output_directory / (
            f"{self.diagnostic_name}_candidate_{training_step:07d}.json"
        )
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "training_step": training_step,
            "result": str(path.resolve()),
            "rates": rates,
            "mean_return": result["mean_return"],
        }

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_step:
            return True
        check = self._evaluate(self.num_timesteps)
        rates = check["rates"]
        sustained_failure = (
            rates["completed"] == 0.0 and rates["distance_failure"] > 0.5
        )
        self._consecutive_failure_windows = (
            self._consecutive_failure_windows + 1 if sustained_failure else 0
        )
        check["failure_window"] = sustained_failure
        check["consecutive_failure_windows"] = self._consecutive_failure_windows
        self.records.append(check)
        print(
            f"{self.diagnostic_name} diagnostic "
            f"step={self.num_timesteps} completed={rates['completed']:.3f} "
            f"distance_failure={rates['distance_failure']:.3f} "
            f"timeout={rates['time_failure']:.3f} "
            f"failure_windows={self._consecutive_failure_windows}"
        )
        self.logger.record("diagnostic/completed_rate", rates["completed"])
        self.logger.record(
            "diagnostic/distance_failure_rate", rates["distance_failure"]
        )
        self.logger.record("diagnostic/timeout_rate", rates["time_failure"])
        self.logger.record("diagnostic/mean_return", check["mean_return"])
        self._next_eval_step += self.eval_freq_steps
        if self._consecutive_failure_windows >= self.required_failure_windows:
            self.stopped_early = True
            self.stop_reason = (
                f"{self.required_failure_windows} consecutive {self.eval_freq_steps}-step "
                "checks had zero completion and distance-failure rate above 0.5"
            )
            self._write_status()
            return False
        self._write_status()
        return True


class CurriculumEarlyStopCallback(BaseCallback):
    """Advance curriculum only after mastery and stop sustained collapse."""

    def __init__(
        self,
        *,
        output_directory: Path,
        curriculum_start_step: int,
        window_steps: int = 25_000,
        required_failure_windows: int = 3,
        required_stall_windows: int = 8,
        ceiling_step: float = 0.025,
        mastery_completion_rate: float = 0.70,
        mastery_distance_failure_rate: float = 0.20,
        regression_completion_rate: float = 0.50,
        regression_distance_failure_rate: float = 0.30,
    ) -> None:
        super().__init__(verbose=0)
        if min(
            curriculum_start_step,
            window_steps,
            required_failure_windows,
            required_stall_windows,
        ) <= 0:
            raise ValueError("curriculum guard settings must be positive")
        rates = (
            ceiling_step,
            mastery_completion_rate,
            mastery_distance_failure_rate,
            regression_completion_rate,
            regression_distance_failure_rate,
        )
        if any(value <= 0.0 or value > 1.0 for value in rates):
            raise ValueError("curriculum control rates must lie in (0, 1]")
        if regression_completion_rate >= mastery_completion_rate:
            raise ValueError("regression completion rate must be below mastery rate")
        if regression_distance_failure_rate <= mastery_distance_failure_rate:
            raise ValueError("regression distance rate must exceed mastery rate")
        self.output_directory = Path(output_directory)
        self.curriculum_start_step = int(curriculum_start_step)
        self.window_steps = int(window_steps)
        self.required_failure_windows = int(required_failure_windows)
        self.required_stall_windows = int(required_stall_windows)
        self.ceiling_step = float(ceiling_step)
        self.mastery_completion_rate = float(mastery_completion_rate)
        self.mastery_distance_failure_rate = float(mastery_distance_failure_rate)
        self.regression_completion_rate = float(regression_completion_rate)
        self.regression_distance_failure_rate = float(
            regression_distance_failure_rate
        )
        self.curriculum_ceiling = 0.0
        self._highest_ceiling = 0.0
        self._windows_without_new_high = 0
        self._next_window_end = self.curriculum_start_step + self.window_steps
        self._episodes = 0
        self._completed = 0
        self._distance_failures = 0
        self._timeouts = 0
        self._consecutive_failure_windows = 0
        self.records: list[dict] = []
        self.stopped_early = False
        self.stop_reason: str | None = None

    def _on_training_start(self) -> None:
        self.training_env.env_method(
            "set_curriculum_ceiling", self.curriculum_ceiling
        )

    def _write_status(self) -> None:
        value = {
            "schema_version": 2,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "curriculum_start_step": self.curriculum_start_step,
            "window_steps": self.window_steps,
            "required_failure_windows": self.required_failure_windows,
            "required_stall_windows": self.required_stall_windows,
            "ceiling_step": self.ceiling_step,
            "mastery_completion_rate": self.mastery_completion_rate,
            "mastery_distance_failure_rate": self.mastery_distance_failure_rate,
            "regression_completion_rate": self.regression_completion_rate,
            "regression_distance_failure_rate": self.regression_distance_failure_rate,
            "curriculum_ceiling": self.curriculum_ceiling,
            "highest_curriculum_ceiling": self._highest_ceiling,
            "windows_without_new_high": self._windows_without_new_high,
            "consecutive_failure_windows": self._consecutive_failure_windows,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "windows": self.records,
        }
        (self.output_directory / "curriculum_early_stop_status.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _on_step(self) -> bool:
        if self.num_timesteps > self.curriculum_start_step:
            for info, done in zip(
                self.locals.get("infos", []), self.locals.get("dones", [])
            ):
                if not done:
                    continue
                self._episodes += 1
                self._completed += int(bool(info.get("completed", False)))
                self._distance_failures += int(
                    bool(info.get("distance_failure", False))
                )
                self._timeouts += int(bool(info.get("time_failure", False)))
        if self.num_timesteps < self._next_window_end:
            return True
        denominator = max(1, self._episodes)
        completed_rate = self._completed / denominator
        distance_failure_rate = self._distance_failures / denominator
        timeout_rate = self._timeouts / denominator
        failure_window = (
            self._episodes > 0
            and completed_rate == 0.0
            and distance_failure_rate > 0.5
        )
        self._consecutive_failure_windows = (
            self._consecutive_failure_windows + 1 if failure_window else 0
        )
        ceiling_before = self.curriculum_ceiling
        mastered = (
            self._episodes > 0
            and completed_rate >= self.mastery_completion_rate
            and distance_failure_rate <= self.mastery_distance_failure_rate
        )
        regressed = (
            self._episodes > 0
            and (
                completed_rate < self.regression_completion_rate
                or distance_failure_rate > self.regression_distance_failure_rate
            )
        )
        if mastered:
            curriculum_action = "advance"
            self.curriculum_ceiling = min(
                1.0, self.curriculum_ceiling + self.ceiling_step
            )
        elif regressed:
            curriculum_action = "rollback"
            self.curriculum_ceiling = max(
                0.0, self.curriculum_ceiling - self.ceiling_step
            )
        else:
            curriculum_action = "hold"
        self.training_env.env_method(
            "set_curriculum_ceiling", self.curriculum_ceiling
        )
        if self.curriculum_ceiling > self._highest_ceiling:
            self._highest_ceiling = self.curriculum_ceiling
            self._windows_without_new_high = 0
        elif self._highest_ceiling < 1.0:
            self._windows_without_new_high += 1
        record = {
            "window_end_step": self.num_timesteps,
            "episodes": self._episodes,
            "completed_rate": completed_rate,
            "distance_failure_rate": distance_failure_rate,
            "timeout_rate": timeout_rate,
            "failure_window": failure_window,
            "consecutive_failure_windows": self._consecutive_failure_windows,
            "curriculum_action": curriculum_action,
            "ceiling_before": ceiling_before,
            "ceiling_after": self.curriculum_ceiling,
            "highest_ceiling": self._highest_ceiling,
            "windows_without_new_high": self._windows_without_new_high,
        }
        self.records.append(record)
        print(
            "Curriculum guard "
            f"step={self.num_timesteps} episodes={self._episodes} "
            f"completed={completed_rate:.3f} "
            f"distance_failure={distance_failure_rate:.3f} "
            f"timeout={timeout_rate:.3f} "
            f"action={curriculum_action} "
            f"ceiling={ceiling_before:.3f}->{self.curriculum_ceiling:.3f} "
            f"stall_windows={self._windows_without_new_high} "
            f"failure_windows={self._consecutive_failure_windows}"
        )
        self.logger.record("curriculum_guard/completed_rate", completed_rate)
        self.logger.record(
            "curriculum_guard/distance_failure_rate", distance_failure_rate
        )
        self.logger.record("curriculum_guard/timeout_rate", timeout_rate)
        self.logger.record(
            "curriculum_guard/ceiling", self.curriculum_ceiling
        )
        self._episodes = 0
        self._completed = 0
        self._distance_failures = 0
        self._timeouts = 0
        self._next_window_end += self.window_steps
        if self._consecutive_failure_windows >= self.required_failure_windows:
            self.stopped_early = True
            self.stop_reason = (
                f"{self.required_failure_windows} consecutive {self.window_steps}-step "
                "curriculum windows had zero completion and distance-failure rate above 0.5"
            )
            self._write_status()
            return False
        if self._windows_without_new_high >= self.required_stall_windows:
            self.stopped_early = True
            self.stop_reason = (
                f"curriculum made no new ceiling progress for "
                f"{self.required_stall_windows} consecutive {self.window_steps}-step windows"
            )
            self._write_status()
            return False
        self._write_status()
        return True
