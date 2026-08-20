import argparse
import sys
from dataclasses import asdict

import numpy as np
import pytest

from eval.evaluate_policy import environment_config_from_manifest
from env.reward import Phase2MissionReward
from env.task import Phase2TaskConfig
from train.configs import PURE_SAC, model_kwargs
from train.train import evaluation_environment_config, training_environment_config, validate_request
from env.observation import (
    PHASE2_MISSION_BODY_TRANSLATION_OBSERVATION_SCHEMA,
    PHASE2_MISSION_OBSERVATION_SCHEMA,
)
from env.se3_rendezvous_env import SE3RendezvousEnv


def test_only_one_pure_sac_configuration_is_exposed() -> None:
    values = model_kwargs()
    assert PURE_SAC.learning_starts == 5_000
    assert PURE_SAC.learning_rate == 1.0e-4
    assert PURE_SAC.tau == 0.001
    assert PURE_SAC.gradient_steps == 4
    assert values["ent_coef"] == "auto_0.005"
    assert values["target_entropy"] == -6.0
    assert values["gamma"] == 0.997
    assert Phase2MissionReward(task=Phase2TaskConfig()).discount_factor == PURE_SAC.gamma
    assert values["policy_kwargs"]["net_arch"] == [256, 256, 256, 256]


def test_phase1_pretrain_and_full_mission_are_explicit() -> None:
    phase1 = training_environment_config("phase1_pretrain")
    full = training_environment_config("full_mission")
    assert phase1.phase2_enabled and phase1.phase2_mission_enabled
    assert not phase1.curriculum_enabled
    assert not phase1.phase1_curriculum_enabled
    assert not phase1.phase1_curriculum_adaptive
    assert not evaluation_environment_config("phase1_pretrain").phase1_curriculum_enabled
    assert phase1 == evaluation_environment_config("phase1_pretrain")
    assert np.isclose(
        phase1.phase2_mission.initial_direction_half_angle_rad,
        np.deg2rad(15.0),
    )
    mission = phase1.phase2_mission
    assert mission.gate_semantics == "acquisition_v2"
    assert mission.initial_distance_min_m == 10.0
    assert mission.initial_distance_max_m == 14.0
    assert np.isclose(mission.initial_attitude_limit_rad, np.deg2rad(20.0))
    assert mission.initial_speed_limit_m_s == 0.05
    assert mission.initial_angular_velocity_component_limit_rad_s == 0.01
    assert mission.gate_position_tolerance_m == 1.5
    assert mission.gate_speed_tolerance_m_s == 0.30
    assert np.isclose(mission.gate_fov_tolerance_rad, np.deg2rad(45.0))
    assert mission.phase1_cruise_speed_m_s == 0.15
    assert phase1.phase2_training_mode == "phase1_pretrain"
    assert phase1.phase2_observation_schema == PHASE2_MISSION_OBSERVATION_SCHEMA
    assert phase1.phase2_observation_schema == (
        "phase2_mission_v3_body_velocity_error_24d"
    )
    assert full.phase2_training_mode == "full_mission"
    assert phase1.phase2_task == full.phase2_task
    assert phase1.phase2_mission == full.phase2_mission


def test_v2_body_velocity_model_schema_remains_loadable_without_reinterpretation() -> None:
    serialized = asdict(training_environment_config("phase1_pretrain"))
    serialized["phase2_observation_schema"] = (
        PHASE2_MISSION_BODY_TRANSLATION_OBSERVATION_SCHEMA
    )
    legacy = environment_config_from_manifest(
        {"evaluation_environment": serialized}
    )
    env = SE3RendezvousEnv(legacy)
    try:
        assert env.config.phase2_observation_schema == (
            "phase2_mission_v2_body_translation_24d"
        )
    finally:
        env.close()


def test_training_evaluation_and_manifest_share_task_semantics() -> None:
    training = training_environment_config("full_mission")
    evaluation = evaluation_environment_config("full_mission")
    restored = environment_config_from_manifest({"evaluation_environment": asdict(evaluation)})
    assert training == evaluation == restored


def test_pre_semantic_fix_manifest_remains_loadable() -> None:
    evaluation = evaluation_environment_config("phase1_pretrain")
    serialized = asdict(evaluation)
    serialized["phase2_mission"].pop("phase1_soft_speed_m_s")
    serialized["phase2_mission"].pop("phase1_catastrophic_speed_limit_m_s")
    restored = environment_config_from_manifest(
        {"evaluation_environment": serialized}
    )
    assert restored.phase2_mission.phase1_speed_limit_m_s == 0.50
    assert restored.phase2_mission.phase1_soft_speed_m_s == 0.25
    assert restored.phase2_mission.phase1_catastrophic_speed_limit_m_s == 1.0


def test_legacy_mission_manifest_defaults_to_legacy_gate_semantics() -> None:
    evaluation = evaluation_environment_config("phase1_pretrain")
    serialized = asdict(evaluation)
    serialized["phase2_mission"].pop("gate_semantics")
    restored = environment_config_from_manifest(
        {"evaluation_environment": serialized}
    )
    assert restored.phase2_mission.gate_semantics == "legacy_regulation"


def test_legacy_training_modules_are_not_imported_by_main_entry() -> None:
    import train.train  # noqa: F401

    forbidden = {"train.behavior_initialization"}
    assert forbidden.isdisjoint(sys.modules)


def test_existing_run_is_rejected(tmp_path) -> None:
    (tmp_path / "models" / "duplicate").mkdir(parents=True)
    args = argparse.Namespace(steps=100, checkpoint_freq=50, run_name="duplicate")
    with pytest.raises(FileExistsError):
        validate_request(args, tmp_path)
