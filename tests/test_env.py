from dataclasses import replace

import numpy as np
from gymnasium.utils.env_checker import check_env

from env.se3_rendezvous_env import SE3RendezvousConfig, SE3RendezvousEnv
from dynamics.lie import make_transform, so3_exp
from dynamics.relative import RelativeState, reconstruct_chaser_state
from env.task import Phase2TaskConfig
from env.phase2_env import phase2_environment_config


def _fast_config(**changes) -> SE3RendezvousConfig:
    base = SE3RendezvousConfig(
        max_time_s=0.3,
        cache_target_trajectory=False,
        curriculum_enabled=False,
    )
    return replace(base, **changes)


def test_gymnasium_contract() -> None:
    env = SE3RendezvousEnv(_fast_config())
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_reset_is_seed_reproducible() -> None:
    env = SE3RendezvousEnv(_fast_config())
    try:
        first, first_info = env.reset(seed=42)
        second, second_info = env.reset(seed=42)
    finally:
        env.close()
    assert np.allclose(first, second)
    assert first_info["relative_distance_m"] == second_info["relative_distance_m"]


def test_step_returns_finite_reward_and_declared_flags() -> None:
    env = SE3RendezvousEnv(_fast_config())
    try:
        env.reset(seed=7)
        observation, reward, terminated, truncated, info = env.step(np.zeros(6))
    finally:
        env.close()
    assert observation.shape == (12,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert {"completed", "distance_failure", "time_failure"} <= info.keys()


def test_curriculum_reaches_exact_final_envelope() -> None:
    config = _fast_config(
        curriculum_enabled=True,
        curriculum_start_step=10,
        curriculum_end_step=20,
        curriculum_rehearsal_end_step=30,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.total_transition_count = 30
        _, info = env.reset(seed=3)
    finally:
        env.close()
    assert info["curriculum_progress"] == 1.0
    assert info["episode_shell_inner_m"] == config.shell_inner_m
    assert info["episode_shell_outer_m"] == config.shell_outer_m
    assert info["episode_target_tumble_scale"] == config.target_tumble_scale


def test_curriculum_separates_rotation_then_translation() -> None:
    config = _fast_config(
        curriculum_enabled=True,
        curriculum_start_step=0,
        curriculum_end_step=100,
        curriculum_rehearsal_end_step=100,
        curriculum_rotation_fraction=0.5,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.total_transition_count = 25
        _, rotation_stage = env.reset(seed=9)
        env.total_transition_count = 75
        _, translation_stage = env.reset(seed=9)
    finally:
        env.close()
    assert rotation_stage["episode_shell_outer_m"] == config.initial_shell_outer_m
    assert rotation_stage["curriculum_rotation_progress"] > 0.0
    assert rotation_stage["curriculum_translation_progress"] == 0.0
    assert translation_stage["curriculum_rotation_progress"] == 1.0
    assert translation_stage["episode_shell_outer_m"] > config.initial_shell_outer_m


def test_curriculum_rehearsal_is_seeded_and_never_exceeds_frontier() -> None:
    config = _fast_config(
        curriculum_enabled=True,
        curriculum_start_step=10,
        curriculum_end_step=110,
        curriculum_rehearsal_end_step=130,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.total_transition_count = 60
        _, first = env.reset(seed=17)
        env.total_transition_count = 60
        _, second = env.reset(seed=17)
    finally:
        env.close()
    assert first["curriculum_progress"] == second["curriculum_progress"]
    assert first["curriculum_progress"] <= first["curriculum_ceiling"] == 0.5
    assert config.initial_target_tumble_scale <= first[
        "episode_target_tumble_scale"
    ] <= config.target_tumble_scale


def test_phase2_environment_contract_and_feasible_reset() -> None:
    env = SE3RendezvousEnv(_fast_config(phase2_enabled=True))
    try:
        check_env(env, skip_render_check=True)
        observation, info = env.reset(seed=31)
    finally:
        env.close()
    assert observation.shape == (23,)
    assert info["constraint_feasible"]
    assert info["constraint_success"]
    assert 3.0 <= info["axial_remaining_m"] <= 7.0


def test_canonical_phase2_reset_step_zero_and_random_actions() -> None:
    config = replace(
        phase2_environment_config("full_mission"),
        max_time_s=0.5,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        observation, info = env.reset(seed=2030)
        assert observation.shape == (24,)
        assert info["constraint_feasible"]
        assert info["mission_phase"] == 0
        assert 12.0 <= info["target_center_distance_m"] <= 18.0
        for action in (np.zeros(6), env.action_space.sample()):
            observation, reward, _, _, info = env.step(action)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert all(np.isfinite(info[name]) for name in (
                "corridor_lateral_margin_m", "fov_margin_rad",
                "total_speed_margin_m_s", "closing_speed_margin_m_s",
            ))
    finally:
        env.close()


def test_phase2_random_and_zero_action_smoke_remain_finite() -> None:
    env = SE3RendezvousEnv(
        _fast_config(phase2_enabled=True, max_time_s=2.0)
    )
    try:
        observation, _ = env.reset(seed=32)
        for step in range(20):
            action = np.zeros(6) if step < 10 else env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            assert np.isfinite(info["corridor_lateral_margin_m"])
            if terminated or truncated:
                break
    finally:
        env.close()


def test_phase2_constraint_violation_is_recorded_but_not_immediate_failure() -> None:
    config = _fast_config(phase2_enabled=True, max_time_s=1.0)
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=33)
        assert env.target_state is not None
        bad_relative = RelativeState(
            make_transform(np.eye(3), np.array([-6.0, 5.0, 0.0])),
            np.zeros(6),
        )
        bad_chaser = reconstruct_chaser_state(env.target_state, bad_relative)
        _, reset_info = env.reset(
            seed=33,
            options={"target_state": env.target_state, "chaser_state": bad_chaser},
        )
        _, _, terminated, truncated, info = env.step(np.zeros(6))
    finally:
        env.close()
    assert not reset_info["constraint_feasible"]
    assert not terminated
    assert not truncated
    assert not info["constraint_success"]
    assert info["corridor_violation_steps"] == 1


def test_phase2_warmup_switches_once_to_nominal_at_episode_boundary() -> None:
    config = _fast_config(phase2_enabled=True, phase2_warmup_steps=10)
    env = SE3RendezvousEnv(config)
    try:
        _, warmup = env.reset(seed=34)
        env.total_transition_count = 10
        _, nominal = env.reset(seed=34)
    finally:
        env.close()
    assert warmup["phase2_stage"] == "warmup"
    assert 2.0 <= warmup["axial_remaining_m"] <= 4.0
    assert warmup["episode_target_tumble_scale"] == 0.18
    assert nominal["phase2_stage"] == "nominal"
    assert 3.0 <= nominal["axial_remaining_m"] <= 7.0
    assert nominal["episode_target_tumble_scale"] == 0.25


def test_phase2_distance_failure_pays_remaining_horizon_terminal_cost() -> None:
    config = _fast_config(
        phase2_enabled=True, max_time_s=10.0, max_distance_m=8.0
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=35)
        assert env.target_state is not None
        escaped = RelativeState(
            make_transform(np.eye(3), np.array([-9.0, 0.0, 0.0])),
            np.zeros(6),
        )
        chaser = reconstruct_chaser_state(env.target_state, escaped)
        env.reset(
            seed=35,
            options={"target_state": env.target_state, "chaser_state": chaser},
        )
        _, reward, terminated, _, info = env.step(np.zeros(6))
    finally:
        env.close()
    assert terminated
    assert info["distance_failure"]
    assert info["terminal_failure_penalty"] < 0.0
    component_total = (
        info["reward_task_dissipation"]
        + info["reward_task_penalty"]
        + info["reward_actuation_penalty"]
        + info["reward_constraint_penalty"]
    )
    assert np.isclose(
        reward, component_total + info["terminal_failure_penalty"]
    )


def _reset_mission_at_gate(env: SE3RendezvousEnv, seed: int = 260813):
    env.reset(seed=seed)
    assert env.target_state is not None
    gate = env.config.phase2_mission.gate_position
    relative = RelativeState(make_transform(np.eye(3), gate), np.zeros(6))
    chaser = reconstruct_chaser_state(env.target_state, relative)
    return env.reset(
        seed=seed,
        options={"target_state": env.target_state, "chaser_state": chaser},
    )


def test_phase1_pretrain_gate_is_single_episode_success() -> None:
    env = SE3RendezvousEnv(
        replace(
            phase2_environment_config("phase1_pretrain"),
            cache_target_trajectory=False,
            max_time_s=1.0,
        )
    )
    try:
        observation, _ = _reset_mission_at_gate(env)
        assert observation[-1] == 0.0
        _, _, terminated, truncated, info = env.step(np.zeros(6))
    finally:
        env.close()
    assert terminated and not truncated
    assert info["gate_success"] and info["completed"]
    assert info["reward_event"] == env.config.phase2_mission.gate_reward


def test_s1v2_gate_does_not_require_attitude_or_angular_rate_regulation() -> None:
    env = SE3RendezvousEnv(
        replace(
            phase2_environment_config("phase1_pretrain"),
            cache_target_trajectory=False,
            max_time_s=1.0,
        )
    )
    try:
        env.reset(seed=260815)
        assert env.target_state is not None
        relative = RelativeState(
            make_transform(
                so3_exp([0.0, np.deg2rad(35.0), 0.0]),
                env.config.phase2_mission.gate_position,
            ),
            np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        chaser = reconstruct_chaser_state(env.target_state, relative)
        env.reset(
            seed=260815,
            options={"target_state": env.target_state, "chaser_state": chaser},
        )
        _, _, terminated, _, info = env.step(np.zeros(6))
    finally:
        env.close()
    assert info["attitude_error_rad"] > env.config.phase2_mission.gate_attitude_tolerance_rad
    assert info["angular_velocity_error_rad_s"] > (
        env.config.phase2_mission.gate_angular_velocity_tolerance_rad_s
    )
    assert info["fov_angle_rad"] < env.config.phase2_mission.gate_fov_tolerance_rad
    assert terminated and info["gate_success"]


def test_full_mission_gate_switches_once_and_changes_active_reference() -> None:
    env = SE3RendezvousEnv(
        replace(
            phase2_environment_config("full_mission"),
            cache_target_trajectory=False,
            max_time_s=1.0,
        )
    )
    try:
        _reset_mission_at_gate(env)
        observation, _, terminated, truncated, first = env.step(np.zeros(6))
        _, _, _, _, second = env.step(np.zeros(6))
    finally:
        env.close()
    assert not terminated and not truncated
    assert observation[-1] == 1.0
    assert first["gate_transition"] and first["mission_phase"] == 1
    assert np.allclose(
        first["active_reference_position_m"], env.config.phase2_task.desired_position
    )
    assert not second["gate_transition"]
    assert second["reward_event"] == 0.0


def test_full_mission_terminal_truth_violation_terminates() -> None:
    env = SE3RendezvousEnv(
        replace(
            phase2_environment_config("full_mission"),
            cache_target_trajectory=False,
            max_time_s=1.0,
        )
    )
    try:
        _reset_mission_at_gate(env)
        env.step(np.zeros(6))
        assert env.target_state is not None
        bad = RelativeState(
            make_transform(np.eye(3), np.array([-7.0, 8.0, 0.0])),
            np.zeros(6),
        )
        env.chaser_state = reconstruct_chaser_state(env.target_state, bad)
        env.relative = bad
        assert isinstance(env._reward, object)
        env._reward.reset(env.relative, env.config.phase2_task.desired_position)
        _, _, terminated, _, info = env.step(np.zeros(6))
    finally:
        env.close()
    assert terminated
    assert info["terminal_constraint_failure"]
    assert not info["constraint_success"]


def test_phase1_coarse_speed_is_soft_and_only_catastrophic_speed_terminates() -> None:
    env = SE3RendezvousEnv(
        replace(
            phase2_environment_config("phase1_pretrain"),
            cache_target_trajectory=False,
            max_time_s=2.0,
        )
    )
    try:
        env.reset(seed=260813)
        assert env.target_state is not None
        mission = env.config.phase2_mission
        soft_state = RelativeState(
            make_transform(np.eye(3), np.array([-12.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0, 0.60, 0.0, 0.0]),
        )
        env.chaser_state = reconstruct_chaser_state(env.target_state, soft_state)
        env.relative = soft_state
        env._reward.reset(env.relative, mission.gate_position)
        _, _, terminated, _, soft_info = env.step(np.zeros(6))
        assert not terminated
        assert not soft_info["phase1_speed_failure"]

        assert env.target_state is not None
        catastrophic_state = RelativeState(
            make_transform(np.eye(3), np.array([-12.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0, 1.10, 0.0, 0.0]),
        )
        env.chaser_state = reconstruct_chaser_state(
            env.target_state, catastrophic_state
        )
        env.relative = catastrophic_state
        env._reward.reset(env.relative, mission.gate_position)
        _, _, terminated, _, catastrophic_info = env.step(np.zeros(6))
    finally:
        env.close()
    assert terminated
    assert catastrophic_info["phase1_speed_failure"]


def test_phase1_training_curriculum_expands_continuously_to_full_distribution() -> None:
    config = replace(
        phase2_environment_config("phase1_pretrain"),
        phase1_curriculum_enabled=True,
        phase1_curriculum_easy_full_probability=0.0,
        phase1_curriculum_full_probability=0.0,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        _, easy = env.reset(seed=263400)
        env.total_transition_count = (
            config.phase1_curriculum_start_step
            + config.phase1_curriculum_end_step
        ) // 2
        _, middle = env.reset(seed=263401)
        env.total_transition_count = config.phase1_curriculum_end_step
        _, full = env.reset(seed=263402)
    finally:
        env.close()
    assert easy["phase1_curriculum_difficulty"] == 0.0
    assert middle["phase1_curriculum_difficulty"] == 0.5
    assert full["phase1_curriculum_difficulty"] == 1.0
    assert 9.02 <= easy["target_center_distance_m"] <= 9.05
    assert 12.0 <= full["target_center_distance_m"] <= 18.0


def test_adaptive_phase1_curriculum_promotes_and_demotes_persistent_frontier() -> None:
    config = replace(
        phase2_environment_config("phase1_pretrain"),
        phase1_curriculum_enabled=True,
        phase1_curriculum_adaptive=True,
        phase1_curriculum_start_step=0,
        phase1_curriculum_window_episodes=2,
        phase1_curriculum_difficulty_step=0.1,
        phase1_curriculum_easy_full_probability=0.0,
        phase1_curriculum_full_probability=0.0,
        cache_target_trajectory=False,
    )
    env = SE3RendezvousEnv(config)
    try:
        env.reset(seed=263500)
        env._update_phase1_curriculum(True)
        env._update_phase1_curriculum(True)
        assert env._phase1_curriculum_frontier == 0.1
        _, promoted = env.reset(seed=263501)
        env._update_phase1_curriculum(False)
        env._update_phase1_curriculum(False)
        assert env._phase1_curriculum_frontier == 0.0
    finally:
        env.close()
    assert promoted["phase1_curriculum_difficulty"] == 0.1
