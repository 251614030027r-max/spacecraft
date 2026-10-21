import numpy as np
import torch as th

from env.t2_stage import (STAGE_S, Calibration, InfoEnv, T2Mission, d_authority, spin_frame, stage_goal)


def _frame():
    # spin about +z, line of sight along +x, port axis tilted 30 deg from the spin axis towards -y
    rot = np.eye(3); omega = np.array([0.0, 0.0, np.radians(2.4)])
    axis = np.array([0.0, -np.sin(np.radians(30)), np.cos(np.radians(30))])
    return spin_frame(rot, omega, np.array([12.0, 0.0, 0.0]), axis), axis


def test_spin_frame_geometry():
    f, _ = _frame()
    b = f["basis"]
    assert np.allclose(b @ b.T, np.eye(3), atol=1e-9)
    assert np.allclose(b[1], [0, 0, 1]) and abs(f["lam_deg"] - 90) < 1e-9 and abs(f["beta_deg"] - 30) < 1e-9
    assert abs(f["lat_err_deg"] - 60) < 1e-9
    # the port's projection points along -y; rotating about +z it reaches +x after 90 deg (= 37.5 s at 2.4 deg/s)
    assert abs(np.degrees(f["phase_rad"]) - 90) < 1e-6 and abs(f["time_to_pass_s"] - 37.5) < 1e-6
    assert np.allclose(b[2], [0, -1, 0])      # towards the oncoming port


def test_stage_goal_is_a_reachable_velocity_target():
    from env.t2_stage import action_for_velocity, latitude_velocity, target_velocity
    mass = 106.0
    d = d_authority(mass)
    assert 1.3 < d < 1.4 and 2 * d / STAGE_S < 0.18          # 1.35 m, 0.17 m/s at 1.6 N
    f, _ = _frame(); b = f["basis"]
    assert 0 < latitude_velocity(f) <= 0.12                  # lam > beta: move towards the spin axis
    q = np.array([12.0, 0.0, 0.0])
    v = target_velocity(f, np.zeros(2))
    g0 = stage_goal(q, v, f, np.zeros(2), mass)
    assert not g0['rate_limited'] and np.allclose(g0['v_goal'], v) and np.allclose(g0['q_goal'], q + STAGE_S * v)
    g1 = stage_goal(q, v, f, np.array([0.5, -0.5]), mass)
    assert not g1['rate_limited'] and np.allclose(g1['v_goal'], target_velocity(f, [0.5, -0.5]))
    far = stage_goal(q, np.array([0.3, 0.0, 0.1]), f, np.array([1.0, 1.0]), mass)
    assert far['rate_limited'] and abs(np.linalg.norm(far['v_goal'] - np.array([0.3, 0.0, 0.1])) - 2 * d / STAGE_S) < 1e-9
    for a in ([0.3, -0.7], [-1.0, 1.0], [0.0, 0.0]):
        vt = target_velocity(f, a)
        assert np.allclose(action_for_velocity(f, -float(vt @ b[0]), float(vt @ b[2]) / 1.0), a)
    assert target_velocity(f, [1, 0]) @ b[0] < target_velocity(f, [-1, 0]) @ b[0]   # a_p > 0 closes in
    assert target_velocity(f, [0, 1]) @ b[2] > 0                                   # a_z > 0 meets the port earlier


def test_info_env_hides_the_true_inertia_only():
    from env.t1_env import T1Env
    e = T1Env()
    e.reset(1000000)
    view = InfoEnv(e)
    true = np.asarray(e.target_parameters.inertia)
    iso = np.asarray(view.target_parameters.inertia)
    assert np.allclose(iso, np.eye(3) * np.trace(true) / 3) and not np.allclose(iso, true)
    assert view.time_s == e.time_s and view.target_state is e.target_state


def test_calibration_fits_execution_against_plan():
    c = Calibration()
    for k in range(12):
        c.add(0.1 + 0.01 * k, 0.02 + 1.2 * (0.1 + 0.01 * k), np.array([0.3, 0.0, 0.0]), np.array([0.0, -0.02, 0.0]))
    assert c.ready
    c0, c1 = c.coefficients()
    assert abs(c0 - 0.02) < 3e-3 and abs(c1 - 1.2) < 2e-2
    dq, dv = c.residuals()
    assert np.allclose(dq, [0.3, 0, 0]) and np.allclose(dv, [0, -0.02, 0])


def test_mission_profile_keeps_urgent_and_economy_deadlines():
    m = T2Mission()
    seen = {m.sample(o)['urgent'] for o in range(1000000, 1000040)}
    assert seen == {True, False}
    for o in range(1000000, 1000040):
        x = m.sample(o)
        lo, hi = m.urgent_deadline_s if x['urgent'] else m.economy_deadline_s
        assert lo <= x['deadline_s'] <= hi and x == m.sample(o)


def test_model_term_uses_only_usable_candidates_and_is_capped():
    from env.t2_stage import OBS_DIM
    from train.t2_mve_sac import StageReplay, T2SAC
    import gymnasium as gym

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        def reset(self, seed=None, options=None): return np.zeros(OBS_DIM, np.float32), {}
        def step(self, a): return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

    from stable_baselines3.common.logger import configure
    m = T2SAC('MlpPolicy', Dummy(), model_lambda=1.0, model_start=0, learning_starts=0, batch_size=8, device='cpu', seed=0)
    m.set_logger(configure(None, []))
    b = m.replay_buffer
    for k in range(16):
        cands = [{'a': np.ones(2, np.float32) * .1, 'reward': -1.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'weight': 0.8}] if k % 2 else []
        b.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([-1.0]), np.array([False]), [{'stage_model': cands}])
    assert b.model_counts == {'transitions': 16, 'with_candidates': 8, 'candidates': 8}
    m.train(gradient_steps=5, batch_size=8)
    used = [s for s in m.model_stats if s['model_rows']]
    assert used and all(s['lambda_eff'] <= 1.0 and s['lambda_eff'] * s['g_model'] <= 0.5 * s['g_real'] + 1e-9 for s in used)
    m0 = T2SAC('MlpPolicy', Dummy(), model_lambda=0.0, learning_starts=0, batch_size=8, device='cpu', seed=0)
    m0.set_logger(configure(None, []))
    for k in range(16):
        m0.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([-1.0]), np.array([False]), [{}])
    m0.train(gradient_steps=2, batch_size=8)
    assert all(s['model_td'] is None for s in m0.model_stats)
