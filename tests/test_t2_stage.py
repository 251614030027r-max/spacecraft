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


def _sac(beta=0.5, batch=8):
    from train.t2_mve_sac import SpacesEnv, make_model
    from stable_baselines3.common.logger import configure
    m = make_model(SpacesEnv(), 'e_mve' if beta > 0 else 'e_sac', 0, smoke=True, model_beta=beta)
    m.model_start = 0; m.learning_starts = 0; m.batch_size = batch
    m.set_logger(configure(None, [])); m._current_progress_remaining = 1.0
    return m


def test_precision_fusion_weight_bounds_the_influence_of_model_error():
    """M3: w = 1 / (1 + (e_a / sigma)^2) from the model's error on the executed intent; w |e_a| <= sigma / 2;
    rows without a factual prediction release nothing; the SAC arm has no model term."""
    from env.t2_stage import OBS_DIM
    rng = np.random.default_rng(0)
    m = _sac()
    for k in range(16):
        fact = {'reward': float(rng.normal(0, 30)), 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': False} if k % 2 else None
        cands = [{'a': np.full(2, .1, np.float32), 'reward': -1.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': False}]
        m.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([-1.0]), np.array([False]),
                            [{'model': {'fact': fact, 'cands': cands, 'diag': {'reason': 'stage'}}}])
    c = m.replay_buffer.model_counts
    assert (c['transitions'], c['with_candidates'], c['candidates']) == (16, 8, 8)
    m.train(gradient_steps=3, batch_size=8)
    data = m.replay_buffer.sample(8)
    with th.no_grad():
        target = data.rewards + (1 - data.dones) * m.gamma * m._soft_value(data.next_observations, m.ent_coef_tensor)
        th.manual_seed(1); loss, info = m._model_loss(data, target, m.ent_coef_tensor)
        b = m.replay_buffer.model_batch()
        th.manual_seed(1); y_fact = b['fact_r'].reshape(-1, 1) + m.gamma * m._soft_value(b['fact_obs'], m.ent_coef_tensor)
        e = (target - y_fact).reshape(-1)[b['fact_valid'] > 0]
        w = 1 / (1 + e ** 2 / m.td_ms)
    if loss is not None:
        assert abs(info['w_mean'] - float(w.mean())) < 1e-5
        assert info['w_err_max'] <= 0.5 * np.sqrt(m.td_ms) + 1e-6
    m0 = _sac(beta=0.0)
    for k in range(16):
        m0.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([-1.0]), np.array([False]), [{}])
    m0.train(gradient_steps=2, batch_size=8)
    assert all(s['model_td'] is None for s in m0.model_stats)


def test_intent_model_uses_the_given_attitude_and_calibration():
    """M1: the stage-end label of an intent uses the calibration it is given and the target attitude the caller
    supplies (the realized one in training), and the stage reward and potential shaping of this mission."""
    from dynamics.lie import so3_exp
    from env.t1_p_stage import plan_state, relative_inertial
    from env.t2_stage import GAMMA_STAGE, EntryCalibration, T2StageEnv, frame_now, observe, stage_goal
    env = T2StageEnv(openings=[1000001], shaping=True, candidates=3)
    env.reset()
    e = env.env
    for k in range(12):
        env.calibration.add(0.10, 0.12, np.array([0.05 * k, 0.0, 0.0]), np.array([0.0, 0.001 * k, 0.0]))
    q, v = relative_inertial(e)
    frame = frame_now(e, q)
    a = np.array([0.3, 0.2])
    goal = stage_goal(q, v, frame, a, e.chaser_parameters.mass)
    plan = env.generator.planner.solve(q, v, goal['q_goal'], goal['v_goal'])
    assert plan['feasible']
    start = {"rotation": e.target_state.rotation.copy(), "omega": e.target_state.omega.copy(), "t": e.time_s, "dv": e._delta_v, "phi": env._phi(),
             "q": q, "v": v, "basis": frame['basis'], "frame": frame}
    cal = env.calibration.frozen(); env._model_t0 = start['t']
    rot16 = start['rotation'] @ so3_exp(16.0 * start['omega']) @ so3_exp(np.array([0.0, 0.05, 0.0]))   # a "realized" attitude
    attitude = lambda s: (rot16 if abs(s - 16.0) < 1e-9 else start['rotation'] @ so3_exp(s * start['omega']), start['omega'])
    dq, dvv = cal.residuals(); b = frame['basis']
    raw = lambda s: (b.T @ dq * (s / 16.0), b.T @ dvv * (s / 16.0), 0.0)
    p = env.predict_intent(plan, a, start, cal, EntryCalibration(), raw, attitude)
    assert not p['done']
    q1, v1 = plan_state(plan, 16.0, e.chaser_parameters.mass)
    assert np.allclose(p['q'], q1 + b.T @ dq) and np.allclose(p['v'], v1 + b.T @ dvv)
    expected = observe(e, p['q'], p['v'], start['t'] + 16.0, start['dv'] + p['dv'], env.mission, rot16, start['omega'],
                       {"position_residual_m": float(np.linalg.norm(dq)), "velocity_residual_m_s": float(np.linalg.norm(dvv)), "held": False})
    assert np.allclose(p['obs'], expected)
    w = e.config.reward
    shaped = -w.time_cost_per_s * 16.0 - w.delta_v_cost_per_m_s * p['dv'] + GAMMA_STAGE * env.potential(p['q'], rot16, start['omega']) - start['phi']
    assert abs(p['reward'] - shaped) < 1e-9


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
    for k in range(19):
        c.add(20.0 + k, 0.3, 60.0 + k, 0.2 + 1.5 * 0.3, k != 0, 150.0)
    assert not c.ready
    c.add(39.0, 0.3, 79.0, 0.2 + 1.5 * 0.3, True, 150.0)
    assert c.ready
    w = T1Reward(time_cost_per_s=0.1, delta_v_cost_per_m_s=10.0)
    p = c.predict(30.0, 0.3, w)
    ps = 20.0 / 22.0
    assert abs(p["duration"] - 70.0) < 1e-9 and abs(p["p_success"] - ps) < 1e-12
    assert abs(p["dv"] - 0.65) < 0.05
    assert abs(p["reward"] - (ps * 100 - (1 - ps) * 150 - 0.1 * 70 - 10 * p["dv"])) < 1e-9


def test_anchored_replay_reproduces_the_executed_intent():
    """M1 + M2 consistency: the executed intent, predicted with its own realized residual path and the realized
    target attitude, reproduces the truth -- the stage reward at every completed stage, and at the entry stage the
    executor's entry (294000, full closing: ~6 s into the stage starting at 48 s, tau 25 s) and the realized
    return. The executor model is the executor's own rule, not an imitation of it."""
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[294000], shaping=True, candidates=3, seed=1)
    env.diagnostics = True
    env.reset(options={'opening': 294000})
    for _ in range(12):
        env.calibration.add(0.1, 0.1, np.zeros(3), np.zeros(3))
    for _ in range(20):
        env.entry_calibration.add(25.0, 0.35, 50.0, 0.6, True, 100.0)
    a = np.array([1.0, 0.0]); done = False; checked = 0; two_stage = False
    while not done:
        obs, r, term, trunc, info = env.step(a); done = term or trunc
        d = info['model']['diag']
        if d['reason'] in ('stage', 'entry'):
            f = d['anchored_fact']
            if f['done'] and f['stages'] == 2:        # the entry one stage later, holding the intent: checked at that stage
                assert not term; two_stage = True
            else:
                assert f['done'] == bool(term) and abs(f['reward'] - r) < 1e-6
            checked += 1
    assert term and info['event'] == 'success' and d['reason'] == 'entry'
    assert d['anchored_fact']['entry']['tau'] == 25.0 and abs(d['anchored_fact']['entry']['s'] - 6.0) < 1e-9
    assert checked == env.stats['stages'] and two_stage and info['model']['cands'] and env.entry_calibration.rows[-1][0] == 25.0


def test_terminal_model_labels_do_not_bootstrap():
    from env.t2_stage import OBS_DIM
    m = _sac(batch=4)
    for k in range(4):
        fact = {'reward': 0.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': True}     # exact factual: w = 1
        cands = [{'a': np.zeros(2, np.float32), 'reward': 50.0, 'next_obs': np.ones(OBS_DIM, np.float32) * 9, 'done': True}]
        m.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([0.0]), np.array([True]),
                            [{'model': {'fact': fact, 'cands': cands}}])
    assert m.replay_buffer.model_counts['entry_candidates'] == 4
    m.td_ms = 1.0
    data = m.replay_buffer.sample(4)
    with th.no_grad():
        for p in m.critic_target.parameters():     # whatever the critic says at s', a terminal label is the label
            p.add_(1.0)
        target = data.rewards + (1 - data.dones) * m.gamma * m._soft_value(data.next_observations, m.ent_coef_tensor)
    loss, info = m._model_loss(data, target, m.ent_coef_tensor)
    q = th.cat(m.critic(data.observations, th.zeros(4, 2)), dim=1)
    expected = 0.5 * ((q - 50.0) ** 2).sum() / 4
    assert info['model_rows'] == 4 and info['w_mean'] == 1.0 and abs(float(loss) - float(expected)) < 1e-3


def test_basic_layering_is_rotation_unaware_closing():
    from env.t2_stage import target_velocity
    f, _ = _frame(); b = f["basis"]
    v = target_velocity(f, np.array([0.4, 0.9]), geometry=False)
    assert np.allclose(v, -(0.05 + 0.15 * 0.4) * b[0])
    assert abs(target_velocity(f, np.array([0.4, 0.9])) @ b[1]) > 0      # with the geometry: cone regulation


def test_update_ratio_is_per_stored_transition():
    import inspect, train.t2_mve_sac as m
    assert 'gradient_steps=max(1, int(round(updates_per_transition)))' in inspect.getsource(m.make_model) and m.UPDATES_PER_TRANSITION == 2
    assert 'model.train(gradient_steps=model.gradient_steps' in inspect.getsource(m.train)


def test_candidates_cover_the_decisive_alternatives():
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[294000], candidates=8, seed=0)
    c = env.candidate_actions(np.array([1.0, 0.7]))
    assert len(c) == 8 and all(np.all(np.abs(x) <= 1.0) for x in c)
    assert np.allclose(c[0], [1.0, 0.0]) and np.allclose(c[1], [1.0, -0.7]) and np.allclose(c[2], [0.4, 0.7])
    assert all(np.linalg.norm(x - [1.0, 0.7]) <= 0.5 + 1e-9 for x in c[4:])


def test_lost_bridge_keeps_the_entry_record_open():
    """T2.2 counted a lost bridge as a failed entry. The record now runs from the first bridge to the terminal
    event; a lost bridge only ends the bridge."""
    from env.t2_stage import T2StageEnv
    env = T2StageEnv(openings=[294000], shaping=False)
    env.reset(options={'opening': 294000})
    e = env.env
    env._bridge = {"t": e.time_s, "dv": e._delta_v, "tau": 60.0, "dv_plan": 0.5}
    env._entry_open = {**env._bridge, "G": -3.0}
    env._close_bridge("lost")
    assert env._bridge is None and env._entry_open is not None and env.entry_calibration.rows == [] and env.stats['entries_lost'] == 1
    env._bridge = {"t": e.time_s, "dv": e._delta_v, "tau": 25.0, "dv_plan": 0.3}
    env._close_bridge("success")
    assert env._entry_open is None and len(env.entry_calibration.rows) == 1
    tau, dv_plan, _, _, success, _ = env.entry_calibration.rows[0]
    assert tau == 60.0 and dv_plan == 0.5 and success


def test_window_commitment_retimes_a_failed_replan_inside_the_pass():
    """Sub-mechanism: a live entry whose fixed-time re-plan fails is re-timed (2, 4, 8 s later) at the plan bound,
    then with the authority reserve; never beyond the deadline; a new request is unchanged."""
    from types import SimpleNamespace
    import env.t1_n_task as N
    from env.t2_stage import WindowGStar
    calls = []
    class Planner:
        def __init__(self, ok): self.ok = ok
        def solve(self, q0, v0, tau, history, guess, entry_speed_m_s=None):
            calls.append((self, tau)); return {"feasible": self.ok(tau), "tau_s": tau, "status": "optimal", "delta_v_m_s": 0.1}
    st = SimpleNamespace(position=np.array([8.0, 0, 0]), velocity=np.zeros(3), rotation=np.eye(3), omega=np.array([0, 0, 0.04]))
    tg = SimpleNamespace(position=np.zeros(3), velocity=np.zeros(3), rotation=np.eye(3), omega=np.array([0, 0, 0.04]))
    env = SimpleNamespace(target_state=tg, chaser_state=st, time_s=50.0, config=SimpleNamespace(max_time_s=200.0),
                          target_parameters=SimpleNamespace(inertia=np.eye(3) * 39.0))
    def failed(self, env, tau):         # the conditional G* could not re-plan the remaining arrival
        self.plan, self.last_outcome, self.last_status = None, "replan_failed", "hold:rejected:infeasible"
    orig = N.ConditionalGStar.decide
    N.ConditionalGStar.decide = failed
    try:
        g = WindowGStar(kappa=1.0, planner=Planner(lambda t: t >= 34.0), reserve_planner=Planner(lambda t: True), deadline_margin_s=45.0)
        g.plan = {"tau_s": 60.0}
        g.decide(env, 30.0)
        assert g.plan is not None and g.tau_s == 34.0 and g.last_status == "window:4:plan" and g.recoveries == 1
        calls.clear(); g.planner = Planner(lambda t: False); g.plan = {"tau_s": 60.0}
        g.decide(env, 30.0)
        assert g.last_status == "window:0:reserve" and [c[1] for c in calls] == [32.0, 34.0, 38.0, 30.0]
        calls.clear(); g.plan = {"tau_s": 60.0}; g.reserve_planner = Planner(lambda t: False)
        env.time_s = 120.0                 # deadline horizon 200 - 45 - 120 = 35 s: only slips up to 35 s
        g.decide(env, 30.0)
        assert g.plan is None and max(c[1] for c in calls) <= 35.0
        calls.clear(); g.plan = None        # a new request: no recovery
        g.decide(env, 30.0)
        assert calls == []
    finally:
        N.ConditionalGStar.decide = orig


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


def test_audit_branch_reproduces_the_continuation_bitwise():
    from env.t2_stage import T2StageEnv
    from experiments.t2_audit import branch
    env = T2StageEnv(openings=[294001], shaping=True, candidates=0)
    env.reset(options={'opening': 294001})
    a = np.array([0.5, 0.3])
    env.step(a)
    b = branch(env)
    r0, r1 = env.step(a), b.step(a)
    assert np.array_equal(r0[0], r1[0]) and r0[1] == r1[1] and r0[4]['true_reward'] == r1[4]['true_reward']


def test_t23_decision_applies_the_preregistered_criteria():
    from experiments.t2_decide import decide_t23
    arm = lambda xs: {str(k): {'mean_return': x} for k, x in zip((0, 1000, 2000, 3000, 4000), xs)}
    pair = lambda m: {'mean': m, 'urgent': m, 'without_top2': m, 'capture_mve': 22, 'capture_sac': 22, 'unsafe_mve': 0, 'unsafe_sac': 0}
    r = {'all_complete': True,
         'arms': {'e_mve_s0': arm([-60, 40, 52, 56, 58]), 'e_mve_s1': arm([-60, 38, 50, 55, 57]),
                  'e_sac_s0': arm([-60, 30, 45, 53, 57]), 'e_sac_s1': arm([-60, 31, 46, 54, 58])},
         'paired': {f's{s}/step_{k}': pair(m) for s in (0, 1) for k, m in zip((1000, 2000, 3000, 4000), (9, 6, 2, 0))}}
    assert decide_t23(r)['verdict'] == 'pass'
    r['paired']['s1/step_4000'] = pair(-8)            # end level fails (seed mean -4)
    assert decide_t23(r)['verdict'] == 'coupling_failure'
    r['paired']['s1/step_4000'] = pair(0)
    r['arms']['e_sac_s1'] = arm([-60, 31, 46, 40, 58])  # last step change 18: not converged
    assert decide_t23(r)['verdict'] == 'not_converged'
    r['all_complete'] = False
    assert decide_t23(r)['verdict'] == 'incomplete'


def test_window_readout_pairs_rigid_and_window(tmp_path):
    import json
    from experiments.t2_readout import window
    for d, rets, lost in (('rigid', [10, 20], 2), ('window', [15, 18], 0)):
        (tmp_path / d).mkdir()
        for k, ret in enumerate(rets):
            (tmp_path / d / f'rally_{k}.json').write_text(json.dumps({'opening': k, 'policy': 'rally', 'return': ret, 'event': 'success',
                                                                     'time_s': 100, 'delta_v': 1.0, 'entries_lost': lost}))
    window(tmp_path / 'rigid', tmp_path / 'window', tmp_path / 'w.json')
    w = json.loads((tmp_path / 'w.json').read_text())
    assert w['pairs'] == 2 and abs(w['paired_return']['mean'] - 1.5) < 1e-9 and w['entries_lost'] == {'rigid': 4, 'window': 0}
