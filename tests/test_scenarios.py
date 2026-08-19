import numpy as np

from dynamics.constants import EARTH
from dynamics.disturbance import zero_disturbance
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings, propagate_rk45
from dynamics.relative import relative_state
from dynamics.types import GeneralizedForce
from env.scenarios import (
    TARGET_ORBIT_ALTITUDE_M,
    chaser_parameters,
    sample_chaser_state,
    sample_phase2_chaser_state,
    sample_phase2_mission_chaser_state,
    sample_volume_uniform_shell,
    target_initial_state,
    target_parameters,
)
from env.task import Phase2MissionConfig, Phase2TaskConfig, compute_mission_metrics, compute_task_metrics


def test_target_nominal_orbit_is_circular_500_km_leo() -> None:
    target = target_initial_state(tumble_scale=0.5)
    radius = np.linalg.norm(target.position)
    inertial_velocity = target.rotation @ target.velocity
    assert np.isclose(radius - EARTH.equatorial_radius, TARGET_ORBIT_ALTITUDE_M)
    assert abs(float(target.position @ inertial_velocity)) < 1.0e-6
    assert np.isclose(np.linalg.norm(inertial_velocity), np.sqrt(EARTH.mu / radius))


def test_target_nominal_orbit_remains_finite_over_episode() -> None:
    target = target_initial_state(tumble_scale=0.5)
    propagated, _ = propagate_rk45(
        target,
        target_parameters(),
        0.0,
        200.0,
        control=GeneralizedForce.zeros(),
        disturbance=zero_disturbance,
        gravity_options=GravityOptions(include_j2=True),
        settings=RK45Settings(rtol=1.0e-7, atol=1.0e-9, max_step=0.1),
    )
    altitude = np.linalg.norm(propagated.position) - EARTH.equatorial_radius
    assert np.all(np.isfinite(propagated.position))
    assert np.all(np.isfinite(propagated.twist))
    assert 400.0e3 <= altitude <= 600.0e3


def test_volume_uniform_shell_respects_bounds() -> None:
    rng = np.random.default_rng(10)
    radii = [
        np.linalg.norm(sample_volume_uniform_shell(rng, 2.0, 10.0))
        for _ in range(200)
    ]
    assert min(radii) >= 2.0
    assert max(radii) <= 10.0


def test_sampled_chaser_respects_declared_pose_and_rate_envelope() -> None:
    rng = np.random.default_rng(12)
    target = target_initial_state(tumble_scale=0.5)
    chaser = sample_chaser_state(
        rng,
        target,
        shell_inner_m=2.0,
        shell_outer_m=10.0,
        max_relative_attitude_rad=np.deg2rad(75.0),
        inertial_velocity_component_limit_m_s=0.125,
        relative_angular_velocity_component_limit_rad_s=0.03,
    )
    relative = relative_state(target, chaser)
    coordinates = relative.exponential_coordinates
    assert 2.0 <= np.linalg.norm(relative.position) <= 10.0
    assert np.linalg.norm(coordinates[:3]) <= np.deg2rad(75.0) + 1e-12
    assert np.max(np.abs(relative.omega)) <= 0.03 + 1e-12
    assert chaser_parameters().mass == 106.0


def test_phase2_sampler_is_reproducible_and_feasible() -> None:
    target = target_initial_state(tumble_scale=0.25)
    first = sample_phase2_chaser_state(np.random.default_rng(21), target)
    second = sample_phase2_chaser_state(np.random.default_rng(21), target)
    assert np.allclose(first.rotation, second.rotation)
    assert np.allclose(first.position, second.position)
    assert np.allclose(first.twist, second.twist)


def test_phase2_sampler_covers_nominal_task_and_near_boundary_states() -> None:
    rng = np.random.default_rng(22)
    target = target_initial_state(tumble_scale=0.25)
    task = Phase2TaskConfig()
    radial_fractions = []
    attitude_angles = []
    recovery_residuals = []
    for _ in range(500):
        relative = relative_state(target, sample_phase2_chaser_state(rng, target))
        metrics = compute_task_metrics(relative, task)
        corridor_radius = (
            metrics.port_axial_distance_m * np.tan(task.corridor_half_angle_rad)
        )
        radial_fractions.append(metrics.corridor_radial_distance_m / corridor_radius)
        attitude_angles.append(np.linalg.norm(relative.exponential_coordinates[:3]))
        lateral = relative.position - task.port_position
        axial = float(task.approach_axis @ lateral)
        lateral -= axial * task.approach_axis
        radial_direction = lateral / np.linalg.norm(lateral)
        margin_rate = (
            np.tan(task.corridor_half_angle_rad)
            * float(task.approach_axis @ metrics.position_rate_target_m_s)
            - float(radial_direction @ metrics.position_rate_target_m_s)
        )
        recovery_residuals.append(
            margin_rate
            + np.sqrt(
                2.0 * 0.03 * max(metrics.corridor_lateral_margin_m, 0.0)
            )
        )
        assert metrics.constraints_satisfied
        assert 3.0 <= metrics.axial_remaining_m <= 7.0
        assert np.max(np.abs(relative.omega)) <= 0.03 + 1e-12
    assert np.quantile(radial_fractions, 0.9) > 0.75
    assert max(radial_fractions) <= 0.95 + 1e-12
    assert max(attitude_angles) > np.deg2rad(40.0)
    assert min(recovery_residuals) >= -1e-12


def test_phase2_stage_a_sampler_enforces_kinetic_and_interior_envelopes() -> None:
    rng = np.random.default_rng(23)
    target = target_initial_state(tumble_scale=0.18)
    task = Phase2TaskConfig(
        corridor_half_angle_rad=np.deg2rad(45.0),
        fov_half_angle_rad=np.deg2rad(60.0),
    )
    for _ in range(500):
        relative = relative_state(
            target,
            sample_phase2_chaser_state(
                rng,
                target,
                task=task,
                axial_remaining_min_m=2.0,
                axial_remaining_max_m=4.0,
                max_relative_attitude_rad=np.deg2rad(35.0),
                relative_angular_velocity_component_limit_rad_s=0.018,
                near_boundary_probability=0.0,
                initial_total_speed_limit_m_s=0.15,
                maximum_interior_radial_fraction=0.65,
                minimum_fov_margin_rad=np.deg2rad(5.0),
            ),
        )
        metrics = compute_task_metrics(relative, task)
        corridor_radius = metrics.port_axial_distance_m * np.tan(
            task.corridor_half_angle_rad
        )
        assert metrics.total_speed_m_s <= 0.15 + 1.0e-12
        assert np.max(np.abs(relative.omega)) <= 0.018 + 1.0e-12
        assert metrics.corridor_radial_distance_m / corridor_radius <= 0.65 + 1.0e-12
        assert metrics.fov_margin_rad >= np.deg2rad(5.0) - 1.0e-12


def test_phase2_mission_sampler_matches_phase1_envelope() -> None:
    rng = np.random.default_rng(260813)
    mission = Phase2MissionConfig()
    target = target_initial_state(tumble_scale=0.20)
    axis = np.array([-1.0, 0.0, 0.0])
    for _ in range(200):
        relative = relative_state(
            target,
            sample_phase2_mission_chaser_state(rng, target, mission=mission),
        )
        metrics = compute_mission_metrics(relative, mission)
        cosine = float(axis @ relative.position / np.linalg.norm(relative.position))
        assert mission.initial_distance_min_m <= metrics.target_center_distance_m <= mission.initial_distance_max_m
        assert cosine >= np.cos(mission.initial_direction_half_angle_rad) - 1.0e-12
        assert metrics.attitude_error_rad <= mission.initial_attitude_limit_rad + 1.0e-12
        assert metrics.total_speed_m_s <= mission.initial_speed_limit_m_s + 1.0e-12
        assert np.max(np.abs(relative.omega)) <= mission.initial_angular_velocity_component_limit_rad_s + 1.0e-12


def test_phase1_easy_sampler_is_near_gate_but_not_initially_complete() -> None:
    rng = np.random.default_rng(263400)
    mission = Phase2MissionConfig()
    target = target_initial_state(tumble_scale=0.20)
    for _ in range(100):
        relative = relative_state(
            target,
            sample_phase2_mission_chaser_state(
                rng, target, mission=mission, difficulty=0.0
            ),
        )
        metrics = compute_mission_metrics(relative, mission)
        assert 9.02 <= metrics.target_center_distance_m <= 9.05
        assert metrics.gate_position_error_m > mission.gate_position_tolerance_m
        assert metrics.attitude_error_rad <= np.deg2rad(2.0) + 1.0e-12
        assert metrics.total_speed_m_s <= 0.03 + 1.0e-12
        assert np.max(np.abs(relative.omega)) <= 0.001 + 1.0e-12
