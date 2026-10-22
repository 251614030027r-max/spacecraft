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
    assert {k: b.model_counts[k] for k in ('transitions', 'with_candidates', 'candidates', 'entry_candidates')} == {'transitions': 16, 'with_candidates': 8, 'candidates': 8, 'entry_candidates': 0}
    m.train(gradient_steps=5, batch_size=8)
    used = [s for s in m.model_stats if s['model_rows']]
    assert used and all(s['lambda_eff'] <= 1.0 and s['lambda_eff'] * s['g_model'] <= 0.5 * s['g_real'] + 1e-9 for s in used)
    m0 = T2SAC('MlpPolicy', Dummy(), model_lambda=0.0, learning_starts=0, batch_size=8, device='cpu', seed=0)
    m0.set_logger(configure(None, []))
    for k in range(16):
        m0.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([-1.0]), np.array([False]), [{}])
    m0.train(gradient_steps=2, batch_size=8)
    assert all(s['model_td'] is None for s in m0.model_stats)


def test_stage_prediction_is_aligned_to_the_stage_start_and_uses_the_frozen_calibration():
    from dynamics.lie import so3_exp
    from env.t1_p_stage import relative_inertial
    from env.t2_stage import GAMMA_STAGE, T2StageEnv, frame_now, observe, stage_goal
    env = T2StageEnv(openings=[1000001], shaping=True, candidates=3)
    env.reset()
    e = env.env
    for k in range(12):
        env.calibration.add(0.10, 0.12, np.array([0.05 * k, 0.0, 0.0]), np.array([0.0, 0.001 * k, 0.0]))
    q, v = relative_inertial(e)
    frame = frame_now(e, q)
    goal = stage_goal(q, v, frame, np.array([0.3, 0.2]), e.chaser_parameters.mass)
    plan = env.generator.planner.solve(q, v, goal['q_goal'], goal['v_goal'])
    assert plan['feasible']
    start = {"rotation": e.target_state.rotation.copy(), "omega": e.target_state.omega.copy(), "t": e.time_s, "dv": e._delta_v, "phi": env._phi()}
    cal = env.calibration.frozen()
    p1 = env.predict_stage(plan, frame['basis'], cal, start)
    # the live calibration and the target attitude move on (as after the real stage); the prediction must not
    env.calibration.add(0.5, 2.0, np.array([3.0, 3.0, 3.0]), np.array([0.3, 0.3, 0.3]))
    e.target_state = type(e.target_state)(rotation=e.target_state.rotation @ so3_exp(16.0 * e.target_state.omega),
                                           position=e.target_state.position, omega=e.target_state.omega, velocity=e.target_state.velocity)
    p2 = env.predict_stage(plan, frame['basis'], cal, start)
    assert np.array_equal(p1['obs'], p2['obs']) and p1['reward'] == p2['reward']
    rot16 = start['rotation'] @ so3_exp(16.0 * start['omega'])
    dq, dvv = cal.residuals()
    expected = observe(e, p1['q'], p1['v'], start['t'] + 16.0, start['dv'] + p1['dv'], env.mission, rot16, start['omega'],
                       {"position_residual_m": float(np.linalg.norm(dq)), "velocity_residual_m_s": float(np.linalg.norm(dvv)), "held": False})
    assert np.allclose(p1['obs'], expected)
    w = e.config.reward
    shaped = -w.time_cost_per_s * 16.0 - w.delta_v_cost_per_m_s * p1['dv'] + GAMMA_STAGE * env.potential(p1['q'], rot16, start['omega']) - start['phi']
    assert abs(p1['reward'] - shaped) < 1e-9
    assert len(cal.rows) == 12 and len(env.calibration.rows) == 13


def test_learner_gamma_is_the_stage_discount():
    from env.t2_stage import GAMMA_STAGE
    from train.t2_mve_sac import make_model
    import inspect, train.t2_mve_sac as m
    assert 'gamma=GAMMA_STAGE' in inspect.getsource(m.make_model)
    assert abs(GAMMA_STAGE - 0.999 ** 8) < 1e-15


def test_mission_preferences_trade_time_against_fuel():
    m = T2Mission()
    for o in range(1000000, 1000020):
        x = m.sample(o)
        if x['urgent']:
            assert x['time_cost_per_s'] > 0.05 and x['fuel_cost_per_m_s'] < 20
        else:
            assert x['time_cost_per_s'] < 0.05 and x['fuel_cost_per_m_s'] > 20


def test_pure_row_is_the_mpc_alone():
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[1000000], shaping=False)
    env.reset(options={'opening': 1000000})
    for _ in range(2):
        env.step_pure()
    assert env.far_field_controls == 0 and env.generator is None


def test_entry_calibration_prediction():
    from env.t2_stage import EntryCalibration
    from env.t1_env import T1Reward
    c = EntryCalibration()
    for k in range(6):
        c.add(20.0 + k, 0.3, 60.0 + k, 0.2 + 1.5 * 0.3, k != 0, 150.0)
    assert c.ready
    w = T1Reward(time_cost_per_s=0.1, delta_v_cost_per_m_s=10.0)
    p = c.predict(30.0, 0.3, w)
    assert abs(p["duration"] - 70.0) < 1e-9 and abs(p["p_success"] - 6.0 / 8.0) < 1e-12
    assert abs(p["dv"] - 0.65) < 0.05
    assert abs(p["reward"] - (0.75 * 100 - 0.25 * 150 - 0.1 * 70 - 10 * p["dv"])) < 1e-9


def test_entry_model_predicts_the_executors_entry():
    """294000, full closing: the stage starting at 48 s opens the entry ~6 s in (tau 25 s) -- the entry
    model on the calibrated plan finds the same node and arrival, and the state gate at the entry accepts it."""
    from env.t2_stage import EntryCalibration, T2StageEnv
    env = T2StageEnv(openings=[294000], shaping=True, candidates=3, seed=1)
    env.reset(options={'opening': 294000})
    for _ in range(12):
        env.calibration.add(0.1, 0.1, np.zeros(3), np.zeros(3))
    for _ in range(6):
        env.entry_calibration.add(25.0, 0.35, 50.0, 0.6, True, 100.0)
    a = np.array([1.0, 0.0])
    for _ in range(3):
        env.step(a)
    obs, r, term, trunc, info = env.step(a)
    assert term and info['event'] == 'success'
    g = info['gate']
    assert g['reason'] == 'entry_ok' and g['predicted_entry'] and abs(g['entry_s'] - 6.0) <= 2.0
    assert g['position_error_m'] <= 0.5 and g['velocity_error_m_s'] <= 0.05
    assert info['stage_model'] and all(c['done'] for c in info['stage_model'])
    assert env.entry_calibration.rows[-1][0] == 25.0


def test_terminal_model_labels_do_not_bootstrap():
    from env.t2_stage import OBS_DIM
    from train.t2_mve_sac import T2SAC
    import gymnasium as gym
    from stable_baselines3.common.logger import configure

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        def reset(self, seed=None, options=None): return np.zeros(OBS_DIM, np.float32), {}
        def step(self, a): return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

    m = T2SAC('MlpPolicy', Dummy(), model_lambda=1.0, model_start=0, learning_starts=0, batch_size=4, device='cpu', seed=0)
    m.set_logger(configure(None, []))
    for k in range(4):
        cands = [{'a': np.zeros(2, np.float32), 'reward': 50.0, 'next_obs': np.ones(OBS_DIM, np.float32) * 9, 'weight': 1.0, 'done': True}]
        m.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([0.0]), np.array([False]), [{'stage_model': cands}])
    assert m.replay_buffer.model_counts['entry_candidates'] == 4
    data = m.replay_buffer.sample(4)
    with th.no_grad():
        # with every candidate terminal the target is the label itself, whatever the critic says at s'
        for p in m.critic_target.parameters():
            p.add_(1.0)
    loss, used = m._model_loss(data, m.ent_coef_tensor)
    q = th.cat(m.critic(data.observations, th.zeros(4, 2)), dim=1)
    expected = 0.5 * ((q - 50.0) ** 2).sum() / 4
    assert used == 4 and abs(float(loss) - float(expected)) < 1e-4


def test_basic_layering_is_rotation_unaware_closing():
    from env.t2_stage import target_velocity
    f, _ = _frame(); b = f["basis"]
    v = target_velocity(f, np.array([0.4, 0.9]), geometry=False)
    assert np.allclose(v, -(0.05 + 0.15 * 0.4) * b[0])
    assert abs(target_velocity(f, np.array([0.4, 0.9])) @ b[1]) > 0      # with the geometry: cone regulation


def test_update_ratio_scales_with_env_count():
    import inspect, train.t2_mve_sac as m
    src = inspect.getsource(m.make_model)
    assert 'gradient_steps=max(1, int(round(updates_per_transition * n_envs)))' in src and m.UPDATES_PER_TRANSITION == 2


def test_candidates_cover_the_decisive_alternatives():
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[294000], candidates=8, seed=0)
    c = env.candidate_actions(np.array([1.0, 0.7]))
    assert len(c) == 8 and all(np.all(np.abs(x) <= 1.0) for x in c)
    assert np.allclose(c[0], [1.0, 0.0]) and np.allclose(c[1], [1.0, -0.7]) and np.allclose(c[2], [0.4, 0.7])
    assert all(np.linalg.norm(x - [1.0, 0.7]) <= 0.5 + 1e-9 for x in c[4:])


def test_mispredicted_entry_event_releases_no_candidate():
    """Same stage as above, but the entry model is made to miss the executed entry: the stage releases no
    candidate (entry labels would rest on a wrong event)."""
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[294000], shaping=True, candidates=3, seed=1)
    env.reset(options={'opening': 294000})
    for _ in range(12):
        env.calibration.add(0.1, 0.1, np.zeros(3), np.zeros(3))
    for _ in range(6):
        env.entry_calibration.add(25.0, 0.35, 50.0, 0.6, True, 100.0)
    a = np.array([1.0, 0.0])
    for _ in range(3):
        env.step(a)
    env.predict_entry = lambda *args, **kwargs: {'entry': False}
    obs, r, term, trunc, info = env.step(a)
    assert term and info['gate']['reason'] == 'entry_event_mismatch' and info['stage_model'] == []


def test_paired_readout_drops_the_two_largest_gains(tmp_path):
    import json
    from experiments.t2_readout import paired
    for arm, rets in (('e_mve_s0', [10, 100, 50, 0]), ('e_sac_s0', [0, 0, 0, 5])):
        d = tmp_path / arm / 'step_100'; d.mkdir(parents=True)
        for k, ret in enumerate(rets):
            (d / f'{k}.json').write_text(json.dumps({'opening': k, 'return': ret, 'event': 'success', 'mission': {'urgent': k % 2 == 0}}))
    p = paired(tmp_path)['s0/step_100']
    assert p['n'] == 4 and abs(p['mean'] - 38.75) < 1e-9 and abs(p['without_top2'] - 2.5) < 1e-9
    assert abs(p['urgent'] - 30.0) < 1e-9 and abs(p['economy'] - 47.5) < 1e-9 and p['positive'] == 3 and p['negative'] == 1


def test_decision_applies_the_preregistered_criteria():
    from experiments.t2_decide import decide
    arm = lambda r0, r2, r3: {'0': {'mean_return': r0}, '2000': {'mean_return': r2}, '3000': {'mean_return': r3}}
    pair = lambda m, u, t2: {'mean': m, 'urgent': u, 'without_top2': t2, 'capture_mve': 21, 'capture_sac': 20, 'unsafe_mve': 0, 'unsafe_sac': 0}
    r = {'all_complete': True, 'arms': {'e_mve_s0': arm(-50, 20, 40), 'e_mve_s1': arm(-50, 20, 35), 'e_sac_s0': arm(-50, 10, 25), 'e_sac_s1': arm(-50, 10, 22)},
         'paired': {'s0/step_2000': pair(10, 1, 1), 's1/step_2000': pair(10, 1, 1), 's0/step_3000': pair(15, 3, 2), 's1/step_3000': pair(13, 1, 1)}}
    assert decide(r)['verdict'] == 'pass'
    r['paired']['s1/step_3000'] = pair(13, -9, 1)
    assert decide(r)['verdict'] == 'coupling_failure'
    r['arms'] = {k: arm(-50, -40, -30) for k in r['arms']}
    assert decide(r)['verdict'] == 'learning_failure'
    r['all_complete'] = False
    assert decide(r)['verdict'] == 'incomplete'
