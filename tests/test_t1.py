"""T1 task, interface G, shared-MPC rows and reward bookkeeping.

Fast checks only; closed-loop behaviour is measured by experiments/t1_probe.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from controllers.mpc.constraints import (
    linearize_t1_constraint_margins,
    normalized_t1_constraint_margins,
    t1_cone_branch,
)
from dynamics.lie import make_transform, se3_log, so3_exp
from env.t1_env import T1Env, T1EnvConfig, T1Reward
from env.t1_task import (
    T1IntentConfig,
    T1InitialDistribution,
    T1TaskConfig,
    T1_TARGET_BOX_HALF_EXTENTS_M,
    T1_TARGET_INERTIA_KG_M2,
    T1_TARGET_MASS_KG,
    collision_clearance,
    compute_t1_metrics,
    cone_quantities,
    episode_anchor,
    intent_from_action,
    reference_position_target_frame,
    sample_t1_initial_states,
    staging_direction,
)

TASK = T1TaskConfig()
INTENT = T1IntentConfig()


def test_box_reproduces_the_declared_inertia():
    a, b, c = (2 * h for h in T1_TARGET_BOX_HALF_EXTENTS_M)
    m = T1_TARGET_MASS_KG
    box = m / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    assert np.allclose(box, np.diag(T1_TARGET_INERTIA_KG_M2), atol=1e-9)
    assert np.allclose((a, b, c), (1.233, 1.020, 0.748), atol=1e-3)


def test_desired_pose_is_inside_cone_and_clear_of_the_body():
    desired = np.asarray(TASK.desired_position)
    axial, lateral, margin = cone_quantities(desired, TASK)
    assert axial == pytest.approx(1.5) and lateral == pytest.approx(0.0)
    assert margin == pytest.approx(0.5)  # axial 1.5 - clearance 1.0
    assert collision_clearance(desired, TASK) == pytest.approx(1.5 - 0.05 - 0.95, abs=1e-9)
    # A point touching the rod tip collides; one inside the box collides.
    assert collision_clearance(np.array([-1.5 - 0.5, 0.0, 0.0]), TASK) < 0.0
    assert collision_clearance(np.zeros(3), TASK) < 0.0


def test_braking_envelope_is_an_exact_at_knots_inner_chord_of_the_sqrt():
    def sqrt_envelope(d):
        return np.sqrt(0.05**2 + 2 * (1 / 106) * d)

    for knot in TASK.envelope_knots_m:
        assert TASK.braking_speed_limit(knot) == pytest.approx(sqrt_envelope(knot))
    for d in np.linspace(0.0, 30.0, 301):
        assert TASK.braking_speed_limit(d) <= sqrt_envelope(d) + 1e-12
    assert TASK.braking_speed_limit(3.0) > 0.98 * sqrt_envelope(3.0)
    assert TASK.outer_radial_closing_speed_limit(6.0) == pytest.approx(0.05)
    assert TASK.closing_speed_limit(-1.0) == pytest.approx(0.05)
    assert TASK.braking_speed_slope(-0.1) == 0.0


def test_sampler_is_deterministic_bounded_and_initially_safe():
    d = T1InitialDistribution()
    rates, ranges = [], []
    for seed in range(40):
        t1, c1 = sample_t1_initial_states(np.random.default_rng(seed), d, TASK)
        t2, c2 = sample_t1_initial_states(np.random.default_rng(seed), d, TASK)
        assert np.array_equal(t1.rotation, t2.rotation) and np.array_equal(c1.position, c2.position)
        w = np.degrees(np.linalg.norm(t1.omega))
        offset = np.degrees(np.arccos(t1.omega[2] / np.linalg.norm(t1.omega)))
        assert 2.1 <= w <= 2.7 and 10.0 <= offset <= 20.0
        rel = np.linalg.norm(c1.position - t1.position)
        assert 9.0 <= rel <= 15.0
        dv = np.linalg.norm(c1.rotation @ c1.velocity - t1.rotation @ t1.velocity)
        assert dv <= 0.10 + 1e-12
        m = compute_t1_metrics(t1, c1, TASK)
        assert not m.violations
        assert np.degrees(m.fov_margin_rad) >= 50.0 - 15.0 - 1e-6
        rates.append(w)
        ranges.append(rel)
    assert np.ptp(rates) > 0.3 and np.ptp(ranges) > 3.0


def test_g_endpoints_and_continuity():
    rng = np.random.default_rng(3)
    target, chaser = sample_t1_initial_states(rng, T1InitialDistribution(), TASK)
    anchor = episode_anchor(target, chaser)
    R = target.rotation
    lat = np.deg2rad(20.0)
    p0 = reference_position_target_frame(R, anchor, lat, 0.0, TASK, INTENT)
    assert np.allclose(R @ p0, 7.0 * staging_direction(anchor, lat))  # inertial 7 m staging
    p_half = reference_position_target_frame(R, anchor, lat, 0.5, TASK, INTENT)
    assert np.allclose(p_half, 7.0 * TASK.approach_axis)
    p_one = reference_position_target_frame(R, anchor, lat, 1.0, TASK, INTENT)
    assert np.allclose(p_one, TASK.desired_position)
    below = reference_position_target_frame(R, anchor, lat, 0.5 - 1e-9, TASK, INTENT)
    above = reference_position_target_frame(R, anchor, lat, 0.5 + 1e-9, TASK, INTENT)
    assert np.linalg.norm(below - above) < 1e-6
    # The latitude channel acts below sigma = 0.5 and is inert at the ends of the second segment.
    other = reference_position_target_frame(R, anchor, -lat, 0.25, TASK, INTENT)
    assert np.linalg.norm(other - reference_position_target_frame(R, anchor, lat, 0.25, TASK, INTENT)) > 1.0


def test_anchor_latitude_zero_is_the_h_equator_and_initial_latitude_is_the_initial_los():
    target, chaser = sample_t1_initial_states(np.random.default_rng(5), T1InitialDistribution(), TASK)
    anchor = episode_anchor(target, chaser)
    assert abs(anchor.e1 @ anchor.h_axis) < 1e-12
    los = chaser.position - target.position
    los /= np.linalg.norm(los)
    latitude = np.arcsin(los @ anchor.h_axis)
    assert np.allclose(staging_direction(anchor, latitude), los, atol=1e-12)


def test_action_mapping():
    assert intent_from_action(np.array([1.0, -1.0])) == pytest.approx((np.pi / 2, 0.0))
    assert intent_from_action(np.array([0.0, 0.0])) == pytest.approx((0.0, 0.5))
    assert intent_from_action(np.array([-2.0, 2.0])) == pytest.approx((-np.pi / 2, 1.0))


def _relative_vector(position, rotation=np.eye(3), velocity=np.zeros(3)):
    x = np.zeros(12)
    x[:6] = se3_log(make_transform(rotation, position))
    x[9:] = rotation.T @ velocity
    return x


@pytest.mark.parametrize(
    "position",
    [np.array([-9.0, 2.0, 1.0]), np.array([2.0, 7.5, 0.0]), np.array([-4.5, 0.6, -0.3]), np.array([-6.1, 0.5, 0.2])],
)
def test_t1_rows_match_finite_differences(position):
    rng = np.random.default_rng(7)
    rotation = so3_exp(rng.normal(size=3) * 0.3)
    x = _relative_vector(position, rotation, rng.normal(size=3) * 0.05)
    x[6:9] = rng.normal(size=3) * 0.01
    omega = np.deg2rad(2.4) * np.array([0.2, 0.1, 0.97])
    jac, _ = linearize_t1_constraint_margins(x, TASK, target_angular_velocity_rad_s=omega, corridor_facets=8)
    base = normalized_t1_constraint_margins(x, TASK, target_angular_velocity_rad_s=omega, corridor_facets=8)
    active = base < 999.0
    numeric = np.zeros_like(jac)
    for i in range(12):
        step = np.zeros(12)
        step[i] = 1e-6
        up = normalized_t1_constraint_margins(x + step, TASK, target_angular_velocity_rad_s=omega, corridor_facets=8)
        down = normalized_t1_constraint_margins(x - step, TASK, target_angular_velocity_rad_s=omega, corridor_facets=8)
        numeric[:, i] = (up - down) / 2e-6
    assert np.allclose(jac[active], numeric[active], atol=1e-5)


def test_branch_selection():
    assert not t1_cone_branch(np.array([-9.0, 0.0, 0.0]), TASK)  # far on axis: sphere rows
    assert t1_cone_branch(np.array([-6.3, 0.2, 0.0]), TASK)  # mouth band, inside cone
    assert not t1_cone_branch(np.array([0.0, 6.2, 0.0]), TASK)  # beside the sphere, outside cone
    assert t1_cone_branch(np.array([-3.0, 0.0, 0.0]), TASK)  # inside the sphere, inside cone


def test_constant_sigma_one_is_bitwise_the_pure_row():
    env_a, env_b = T1Env(T1EnvConfig()), T1Env(T1EnvConfig())
    env_a.reset(290007)
    env_b.reset(290007)
    for _ in range(3):
        env_a.step(pure=True)
        env_b.step(np.array([0.37, 1.0]))
    assert np.array_equal(env_a.chaser_state.position, env_b.chaser_state.position)
    assert np.array_equal(env_a.chaser_state.velocity, env_b.chaser_state.velocity)
    assert np.array_equal(env_a.chaser_state.rotation, env_b.chaser_state.rotation)


def test_reward_bookkeeping_without_events():
    env = T1Env(T1EnvConfig())
    env.reset(290006)
    reward, terminated, truncated, info = env.step(np.array([0.0, -1.0]))
    assert not terminated and not truncated and info["event"] is None
    dv = env.summary()["delta_v_m_s"]
    expected = -(0.05 * 2.0 + 5.0 * dv)
    assert reward == pytest.approx(expected, abs=1e-9)


def test_typical_returns_rank_as_designed():
    r = T1Reward()

    def ret(duration_s, dv, event):
        n = int(round(duration_s / 2.0))
        per = -(r.time_cost_per_s * 2.0 + r.delta_v_cost_per_m_s * dv / n)
        total = sum(r.gamma**k * per for k in range(n))
        last = r.gamma ** (n - 1)
        total += last * {"success": r.capture_reward, "timeout": -r.timeout_penalty, "unsafe": -r.unsafe_penalty}[event]
        return total

    worst_success = ret(298.0, 8.0, "success")
    timeout = ret(300.0, 6.0, "timeout")
    early_unsafe = ret(40.0, 1.0, "unsafe")
    assert worst_success > 0 > timeout > early_unsafe


def test_nearest_equivalent_log_keeps_the_pose_and_moves_to_the_hint_branch():
    from controllers.mpc.prediction import nearest_equivalent_log
    from dynamics.lie import se3_exp

    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    xi = np.concatenate((np.deg2rad(175.0) * axis, [1.0, -2.0, 0.5]))
    hint = -np.deg2rad(178.0) * axis  # the other side of the pi cut
    out = nearest_equivalent_log(xi, hint)
    assert np.allclose(se3_exp(out), se3_exp(xi), atol=1e-10)
    assert np.linalg.norm(out[:3] - hint) < np.deg2rad(10.0)
    assert np.array_equal(nearest_equivalent_log(xi, xi[:3]), xi)
