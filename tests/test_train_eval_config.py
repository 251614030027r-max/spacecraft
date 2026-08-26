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
    assert PURE_SAC.buffer_size == 300_000
    assert PURE_SAC.learning_rate == 1.0e-4
    assert PURE_SAC.tau == 0.001
    assert PURE_SAC.gradient_steps == 1
    assert values["ent_coef"] == 0.005
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
    assert mission.waypoint_semantics == "acquisition_v2"
    assert mission.initial_distance_min_m == 10.0
    assert mission.initial_distance_max_m == 14.0
    assert np.isclose(mission.initial_attitude_limit_rad, np.deg2rad(20.0))
    assert mission.initial_speed_limit_m_s == 0.05
    assert mission.initial_angular_velocity_component_limit_rad_s == 0.01
    assert mission.waypoint_position_tolerance_m == 1.5
    assert mission.waypoint_speed_tolerance_m_s == 0.30
    assert np.isclose(mission.waypoint_fov_tolerance_rad, np.deg2rad(45.0))
    assert mission.phase1_cruise_speed_m_s == 0.15
    assert phase1.phase2_training_mode == "phase1_pretrain"
    assert phase1.phase2_observation_schema == PHASE2_MISSION_OBSERVATION_SCHEMA
    assert phase1.phase2_observation_schema == (
        "phase2_mission_v4_phase_guidance_error_24d"
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


def test_legacy_mission_manifest_defaults_to_legacy_waypoint_semantics() -> None:
    evaluation = evaluation_environment_config("phase1_pretrain")
    serialized = asdict(evaluation)
    serialized["phase2_mission"].pop("waypoint_semantics")
    restored = environment_config_from_manifest(
        {"evaluation_environment": serialized}
    )
    assert restored.phase2_mission.waypoint_semantics == "legacy_regulation"


def test_legacy_training_modules_are_not_imported_by_main_entry() -> None:
    import train.train  # noqa: F401

    forbidden = {"train.behavior_initialization"}
    assert forbidden.isdisjoint(sys.modules)


def test_existing_run_is_rejected(tmp_path) -> None:
    (tmp_path / "models" / "duplicate").mkdir(parents=True)
    args = argparse.Namespace(steps=100, checkpoint_freq=50, run_name="duplicate")
    with pytest.raises(FileExistsError):
        validate_request(args, tmp_path)


def test_evaluate_model_runs_end_to_end_on_a_plain_predictor() -> None:
    """evaluate_model only needs .predict, and every info key it reads must exist.

    A missing key raised only once a real model was loaded, which the sandbox
    never does, so exercise the whole loop against a trivial predictor.
    """

    import numpy as np

    from dataclasses import replace
    from env.phase2_env import phase2_environment_config
    from eval.evaluate_policy import evaluate_model

    class ZeroPredictor:
        def predict(self, observation, deterministic=True):  # noqa: ARG002
            return np.zeros(6, dtype=np.float32), None

    config = replace(
        phase2_environment_config("single_phase"),
        cache_target_trajectory=False,
        max_time_s=3.0,
    )
    result = evaluate_model(ZeroPredictor(), config, episodes=2, seed=262000)
    assert result["rates"]["episode_completion"] == 0.0
    record = result["episode_records"][0]
    assert set(record["best_completion_conditions"]) == {
        "position_error_m",
        "attitude_error_rad",
        "total_speed_m_s",
        "angular_velocity_rad_s",
    }
    assert record["best_completion_streak"] == 0
    assert set(record["minimum_margins"]) == {
        "corridor_axial_margin_m",
        "corridor_lateral_margin_m",
        "fov_margin_rad",
        "total_speed_margin_m_s",
        "closing_speed_margin_m_s",
    }


def test_shared_path_reports_the_main_table_metrics() -> None:
    """The four differentiating columns must come off the shared path.

    Completion rate cannot separate Pure SAC, Pure MPC and the hybrid, so the
    table rests on completion time, force impulse, worst constraint margin and
    per-step compute. Three of those were unreadable here until the path
    recorded them; if a row ever has to be filled by a fork, the rows stop
    being comparable, so pin them.
    """

    import numpy as np

    from dataclasses import replace
    from env.phase2_env import phase2_environment_config
    from eval.evaluate_policy import evaluate_model

    class ZeroPredictor:
        def predict(self, observation, deterministic=True):  # noqa: ARG002
            return np.zeros(6, dtype=np.float32), None

    config = replace(
        phase2_environment_config("single_phase"),
        cache_target_trajectory=False,
        max_time_s=2.0,
    )
    result = evaluate_model(ZeroPredictor(), config, episodes=2, seed=262000)

    record = result["episode_records"][0]
    assert record["steps"] == 20
    assert record["survival_s"] == pytest.approx(2.0)
    assert "discounted_return" in record

    table = result["main_table"]
    assert table["episodes"] == 2
    assert table["completed_episodes"] == 0
    # No episode completed, so there is no completion time and no completed-only
    # impulse. None, never zero: a zero would read as "instant" in a table.
    assert table["completion_time_s"] is None
    assert table["force_impulse_n_s"]["completed_only"] is None
    assert table["force_impulse_n_s"]["all_episodes"] is not None
    assert set(table["worst_constraint_margin"]) == {
        "corridor_axial_margin_m",
        "corridor_lateral_margin_m",
        "fov_margin_rad",
        "total_speed_margin_m_s",
        "closing_speed_margin_m_s",
    }


def test_controller_and_environment_compute_are_timed_apart() -> None:
    """The compute column is the controller; the RK45 truth step is not it.

    Timing them together over-reports the controller by about 8.6x here, and
    ``experiments.evaluate_mpc`` has always timed ``controller.command`` alone,
    so a merged number on one row and a controller-only number on another is
    not a comparison. The retained ``full_control_cycle_runtime_s`` is their
    sum and must stay strictly larger than the controller half.
    """

    import numpy as np

    from dataclasses import replace
    from env.phase2_env import phase2_environment_config
    from eval.evaluate_policy import evaluate_model

    class ZeroPredictor:
        def predict(self, observation, deterministic=True):  # noqa: ARG002
            return np.zeros(6, dtype=np.float32), None

    config = replace(
        phase2_environment_config("single_phase"),
        cache_target_trajectory=False,
        max_time_s=2.0,
    )
    result = evaluate_model(ZeroPredictor(), config, episodes=1, seed=262000)
    record = result["episode_records"][0]

    controller = record["controller_time_s"]
    environment = record["environment_step_time_s"]
    combined = record["full_control_cycle_runtime_s"]
    assert controller["count"] == environment["count"] == combined["count"] == 20
    assert combined["mean"] > controller["mean"]
    # A no-op predictor against an RK45 truth step: the simulator has to
    # dominate, which is exactly why the two were worth separating.
    assert environment["mean"] > controller["mean"]

    compute = result["main_table"]["per_step_compute_s"]
    assert compute["control_period_s"] == pytest.approx(config.dt_s)
    assert compute["controller"]["count"] == 20
    assert compute["controller_mean_over_budget"] == pytest.approx(
        compute["controller"]["mean"] / config.dt_s
    )


def test_main_table_reads_effort_over_completed_episodes_only() -> None:
    """Force impulse is a time integral, so dying early scores a smaller one.

    Ranking rows by the raw mean would reward dying early, so the completed-only
    figure is the one the table uses and the all-episode figure sits beside it.
    """

    from eval.metrics import main_table_metrics

    margins = {
        "corridor_axial_margin_m": 1.0,
        "corridor_lateral_margin_m": 1.0,
        "fov_margin_rad": 1.0,
        "total_speed_margin_m_s": 1.0,
        "closing_speed_margin_m_s": 1.0,
    }
    records = [
        {
            "completed": True,
            "survival_s": 95.0,
            "force_impulse_n_s": 170.0,
            "torque_impulse_nm_s": 4.0,
            "minimum_margins": margins,
        },
        {   # died at 10 s, so its impulse is small for the wrong reason
            "completed": False,
            "survival_s": 10.0,
            "force_impulse_n_s": 18.0,
            "torque_impulse_nm_s": 0.4,
            "minimum_margins": {**margins, "fov_margin_rad": -0.2},
        },
    ]
    table = main_table_metrics(
        records,
        controller_times_s=[0.05, 0.15],
        environment_step_times_s=[0.008, 0.008],
        control_period_s=0.1,
    )
    assert table["force_impulse_n_s"]["completed_only"]["mean"] == pytest.approx(170.0)
    assert table["force_impulse_n_s"]["all_episodes"]["mean"] == pytest.approx(94.0)
    assert table["completion_time_s"]["mean"] == pytest.approx(95.0)
    # The worst margin spans every episode: grazing a boundary on the episodes a
    # method loses does not earn it a clean margin column.
    assert table["worst_constraint_margin"]["fov_margin_rad"] == pytest.approx(-0.2)
    assert table["per_step_compute_s"]["controller_mean_over_budget"] == pytest.approx(1.0)


def test_main_table_assembler_reads_main_table_blocks(tmp_path) -> None:
    """The three-way table is assembled from each path's main_table block.

    Every measurement path emits main_table in the shared conventions; the
    assembler must read those blocks and rank the four differentiating columns
    without re-deriving anything, so a row is never a private convention.
    """

    import json as _json

    from eval.main_table import _render, _row

    def table(compute_mean_s: float, completed: int, episodes: int) -> dict:
        return {
            "episodes": episodes,
            "completed_episodes": completed,
            "completion_rate": completed / episodes,
            "completion_time_s": {"mean": 100.0} if completed else None,
            "force_impulse_n_s": {
                "completed_only": {"mean": 170.0} if completed else None,
                "all_episodes": {"mean": 90.0},
            },
            "worst_constraint_margin": {
                "corridor_axial_margin_m": 1.0,
                "fov_margin_rad": 0.05,
            },
            "per_step_compute_s": {
                "control_period_s": 0.1,
                "controller": {"mean": compute_mean_s},
                "controller_mean_over_budget": compute_mean_s / 0.1,
            },
            "discounted_return": {"completed_only": {"mean": 4.1} if completed else None},
        }

    scripted = _row("Scripted", table(8.5e-5, 3, 3))
    mpc = _row("Pure MPC", table(0.10, 2, 2))
    assert scripted["completion"] == "3/3"
    assert scripted["compute_ms"] == "0.085"
    assert mpc["budget_x"] == "1.00x"
    # Worst margin is the single tightest across the five, not a per-column list.
    assert scripted["worst_margin"] == "+0.050"
    rendered = _render([scripted, mpc])
    assert "Scripted" in rendered and "Pure MPC" in rendered

    # A JSON missing the block is rejected, not silently skipped.
    from eval.main_table import _load_table

    empty = tmp_path / "no_block.json"
    empty.write_text(_json.dumps({"rates": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_table(empty)
