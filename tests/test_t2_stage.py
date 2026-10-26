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


def test_entry_calibration_replays_executed_entries():
    """T2.4: each recorded entry's realized deviation (time beyond its plan, delta-v beyond its plan) is replayed
    on the candidate's plan and remaining time. A timed-out record is censored: it counts as a failure only where
    the candidate has no more slack than that record had; unsafe records always count."""
    from env.t2_stage import EntryCalibration
    from env.t1_env import T1Reward
    w = T1Reward(time_cost_per_s=0.1, delta_v_cost_per_m_s=10.0)
    c = EntryCalibration()
    for k in range(19):                         # 19 captures, 40 s beyond the plan, +0.1 m/s beyond the plan
        c.add(20.0, 0.3, 60.0, 0.4, True, 100.0, 150.0)
    assert not c.ready
    c.add(30.0, 0.3, 50.0, 0.4, False, 100.0, 50.0)   # timed out with 20 s of slack
    assert c.ready
    roomy = c.predict(25.0, 0.2, w, remaining=120.0)   # slack 95 s > 20 s: the censored record is left out
    assert roomy['records'] == 19 and roomy['p_success'] == 1.0
    assert abs(roomy['reward'] - (100 - 0.1 * 65.0 - 10 * 0.3)) < 1e-9
    tight = c.predict(25.0, 0.2, w, remaining=40.0)    # 65 s needed > 40 s: every record times out here
    assert tight['records'] == 20 and tight['p_success'] == 0.0 and abs(tight['reward'] - (-100 - 0.1 * 40.0 - 10 * 0.3)) < 1e-9
    c.add(20.0, 0.3, 30.0, 0.5, False, 150.0, 150.0)  # an unsafe record always counts
    assert c.predict(25.0, 0.2, w, remaining=120.0)['records'] == 20
    old = EntryCalibration(rows=[(20.0, 0.3, 60.0, 0.4, True, 100.0)] * 20)   # rows without a remaining time still load
    assert old.predict(25.0, 0.2, w, remaining=120.0)['p_success'] == 1.0


def test_entry_label_reliability_uses_the_current_outcome_model_bias():
    """T2.4: w_k = 1 / (1 + (eps_k / sigma)^2), eps_k = |e_a| + b_k. b_k is the *current* entry outcome-model bias for
    entry labels resting on the outcome model alone (kind 2; one sigma while the bias is unknown), and 0 for entry
    labels anchored to the decision's realized entry (kind 1) and for stage labels (kind 0)."""
    from env.t2_stage import OBS_DIM
    m = _sac(batch=4)
    for k in range(4):
        fact = {'reward': 0.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': True}
        cands = [{'a': np.zeros(2, np.float32), 'reward': 10.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': True, 'kind': 2},
                 {'a': np.ones(2, np.float32) * .5, 'reward': 1.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': True, 'kind': 1}]
        m.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([0.0]), np.array([True]),
                            [{'model': {'fact': fact, 'cands': cands}}])
    m.td_ms = 100.0                              # sigma 10
    data = m.replay_buffer.sample(4)
    with th.no_grad():
        target = data.rewards + (1 - data.dones) * m.gamma * m._soft_value(data.next_observations, m.ent_coef_tensor)
    _, info = m._model_loss(data, target, m.ent_coef_tensor)       # bias unknown: one sigma -> 1 / (1 + 1) = 0.5
    assert abs(info['w_mean'] - (0.5 + 1.0) / 2) < 1e-6 and abs(info['entry_unanchored_share'] - 0.5) < 1e-9
    m.entry_bias_now = -20.0                     # measured now: the stored labels are re-weighted, 1 / (1 + 4) = 0.2
    _, info = m._model_loss(data, target, m.ent_coef_tensor)
    assert abs(info['w_mean'] - (0.2 + 1.0) / 2) < 1e-6
    m.entry_bias_now = 0.0
    _, info = m._model_loss(data, target, m.ent_coef_tensor)
    assert abs(info['w_mean'] - 1.0) < 1e-6

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
    a = np.array([1.0, 0.0]); done = False; checked = 0
    while not done:
        obs, r, term, trunc, info = env.step(a); done = term or trunc
        d = info['model']['diag']
        if d['reason'] in ('stage', 'entry'):
            f = d['anchored_fact']
            assert f['done'] == bool(term) and abs(f['reward'] - r) < 1e-6
            checked += 1
    assert term and info['event'] == 'success' and d['reason'] == 'entry'
    assert d['anchored_fact']['entry']['tau'] == 25.0 and abs(d['anchored_fact']['entry']['s'] - 6.0) < 1e-9
    assert checked == env.stats['stages'] and info['model']['cands'] and env.entry_calibration.rows[-1][0] == 25.0
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
    tau, dv_plan, _, _, success, _, remaining = env.entry_calibration.rows[0]
    assert tau == 60.0 and dv_plan == 0.5 and success and remaining == env.mission["deadline_s"] - e.time_s


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


def test_t24_decision_applies_the_preregistered_criteria():
    from experiments.t2_decide import decide_t24
    arm = lambda xs: {str(k): {'mean_return': x} for k, x in zip((0, 1000, 2000, 3000, 4000), xs)}
    pair = lambda m: {'mean': m, 'urgent': m, 'without_top2': m, 'capture_mve': 23, 'capture_sac': 23, 'unsafe_mve': 0, 'unsafe_sac': 0}
    r = {'all_complete': True, 'arms': {'e_mve_s0': arm([-15, 40, 50, 55, 56]), 'e_sac_s0': arm([-15, 36, 45, 55, 54]),
                                        'e_mve_s1': arm([None, 30, 45, 50]), 'e_sac_s1': arm([None, 26, 44, 50])},
         'paired': {**{f's0/step_{k}': pair(m) for k, m in ((1000, 4), (2000, 5), (3000, 0), (4000, 2))},
                    **{f's1/step_{k}': pair(m) for k, m in ((1000, 4), (2000, 1), (3000, 0))}}}
    assert decide_t24(r)['verdict'] == 'pass'
    r['paired']['s1/step_3000'] = pair(-20)
    assert decide_t24(r)['verdict'] == 'seed0_only'
    r['arms']['e_mve_s0'] = arm([-15, 30, 35, 40, 41])
    assert decide_t24(r)['verdict'] == 'coupling_failure'
    r['all_complete'] = False
    assert decide_t24(r)['verdict'] == 'incomplete'


def test_two_stage_labels_use_the_policy_copy_and_the_executor_model():
    """T2.5: a non-terminal candidate gets a stage-2 part flown by the policy copy's action (a2, log mu) on the
    calibrated model from the candidate's predicted stage end; without a policy copy there is none (stage-1 label);
    a candidate whose own stage opens the entry stays a terminal stage-1 label."""
    from env.t2_stage import ENTRY_STAGES, T2StageEnv
    assert ENTRY_STAGES == 2
    env = T2StageEnv(openings=[294000], shaping=True, candidates=4, seed=1)
    env.reset(options={'opening': 294000})
    for _ in range(12):
        env.calibration.add(0.1, 0.1, np.zeros(3), np.zeros(3))
    for _ in range(20):
        env.entry_calibration.add(25.0, 0.35, 50.0, 0.6, True, 100.0)
    _, _, _, _, info = env.step(np.array([1.0, 0.0]))                  # no policy copy yet
    assert info['model']['cands'] and all(c['s2'] is None for c in info['model']['cands'] if not c['done'])
    calls = []
    env.label_policy = lambda o: (calls.append(o) or (np.array([1.0, 0.0]), -1.5))
    _, _, term, _, info = env.step(np.array([1.0, 0.0]))
    cands = info['model']['cands']
    two = [c for c in cands if not c['done'] and c['s2'] is not None]
    assert two and len(calls) >= len(two)
    for c in two:
        assert c['s2']['logp'] == -1.5 and np.allclose(c['s2']['a'], [1.0, 0.0]) and np.isfinite(c['s2']['reward'])
        assert c['s2']['kind'] == (2 if c['s2']['done'] else 0)
    assert all(c['s2'] is None for c in cands if c['done'])


def _retrace_fixture(c2_logp, created=0.0, d1=0.0, d2=0.0):
    """One stored candidate with a stage-2 part; value functions replaced by known constants."""
    from env.t2_stage import OBS_DIM
    m = _sac(batch=1)
    m.td_ms = 1e6
    f = lambda v: th.full((1, 1), float(v))
    batch = {'cand_obs': th.zeros(1, 1, OBS_DIM), 'cand_done': th.full((1, 1), d1), 'cand_r': f(2.0).reshape(1, 1), 'created': th.full((1,), created),
             'cand_valid': th.ones(1, 1), 'cand_kind': th.zeros(1, 1), 'c2_valid': th.ones(1, 1), 'c2_a': th.zeros(1, 1, 2),
             'c2_logp': th.full((1, 1), c2_logp), 'c2_r': th.full((1, 1), 3.0), 'c2_obs': th.ones(1, 1, OBS_DIM), 'c2_done': th.full((1, 1), d2),
             'c2_kind': th.zeros(1, 1)}
    m._soft_value = lambda obs, ec: th.where(obs[:, :1] > 0.5, th.full_like(obs[:, :1], 7.0), th.full_like(obs[:, :1], 5.0))   # V(o1)=5, V(o2)=7
    m._q_target_min = lambda obs, a: th.full((obs.shape[0], 1), 4.0)                                                           # Qbar(o1, a2) = 4
    m._log_pi = lambda obs, a: th.full((obs.shape[0], 1), -1.0)                                                                 # log pi(a2|o1) = -1
    m.replay_buffer.stored = 10
    return m, batch


def test_two_stage_soft_retrace_target():
    """T2.5 target: y = r1 + g V(o1) + g c (r2 + g (1 - d2) V(o2) - Qbar(o1, a2)), c = min(1, pi/mu); c = 0 past the age
    cap; a terminal stage 1 is the label alone."""
    from train.t2_mve_sac import AGE_CAP
    g = __import__('env.t2_stage', fromlist=['GAMMA_STAGE']).GAMMA_STAGE
    err = th.zeros(1)
    m, b = _retrace_fixture(c2_logp=-1.0)                      # mu = pi: c = 1
    y, c, delta, wk, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), err, 0.05)
    assert abs(float(c) - 1.0) < 1e-6 and abs(float(delta) - (3 + g * 7 - 4)) < 1e-5
    assert abs(float(y) - (2 + g * 5 + g * (3 + g * 7 - 4))) < 1e-5
    m, b = _retrace_fixture(c2_logp=-1.0 + np.log(4.0))        # mu = 4 pi: c = 1/4
    y, c, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), err, 0.05)
    assert abs(float(c) - 0.25) < 1e-6 and abs(float(y) - (2 + g * 5 + g * 0.25 * (3 + g * 7 - 4))) < 1e-5
    m, b = _retrace_fixture(c2_logp=-5.0)                      # mu << pi: truncated at 1
    _, c, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), err, 0.05)
    assert abs(float(c) - 1.0) < 1e-6
    m, b = _retrace_fixture(c2_logp=-1.0, d2=1.0)              # stage 2 ends in the entry: no bootstrap beyond it
    y, c, delta, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), err, 0.05)
    assert abs(float(delta) - (3 - 4)) < 1e-6
    m, b = _retrace_fixture(c2_logp=-1.0, created=-float(AGE_CAP) - 1)   # too old: stage-1 target only
    y, c, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), err, 0.05)
    assert float(c) == 0.0 and abs(float(y) - (2 + g * 5)) < 1e-5
    m, b = _retrace_fixture(c2_logp=-1.0, d1=1.0)              # terminal stage 1: the label itself
    y, c, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), err, 0.05)
    assert float(c) == 0.0 and abs(float(y) - 2.0) < 1e-6


def test_two_stage_error_estimate_counts_both_model_stages():
    """eps_k = |e_a| + b1 + g c (|e_a| + b2); w_k = 1 / (1 + (eps_k / sigma)^2)."""
    g = __import__('env.t2_stage', fromlist=['GAMMA_STAGE']).GAMMA_STAGE
    m, b = _retrace_fixture(c2_logp=-1.0)
    m.td_ms = 100.0; m.entry_bias_now = -20.0
    b['c2_kind'] = th.full((1, 1), 2.0)                        # stage 2 ends in an entry resting on the outcome model alone
    _, c, _, wk, *_ = m._candidate_targets(b, th.ones(1, 1, dtype=th.bool), th.full((1,), 3.0), 0.05)
    eps = 3.0 + g * 1.0 * (3.0 + 20.0)
    assert abs(float(wk) - 1.0 / (1.0 + eps ** 2 / 100.0)) < 1e-6


def test_learner_refreshes_the_environment_policy_copy():
    import inspect, train.t2_mve_sac as t
    src = inspect.getsource(t.train)
    assert "payload = ('actor', copy.deepcopy(model.actor).cpu())" in src and 'model._n_updates - synced[w_] >= SYNC_UPDATES' in src
    assert t.SYNC_UPDATES == 100 and t.AGE_CAP == 2000
    w = inspect.getsource(t.worker)
    assert 'env.label_policy = label_policy' in w and 'actor.load_state_dict(payload[1])' in w


def test_log_pi_matches_the_actor_sampling_density():
    from env.t2_stage import OBS_DIM
    m = _sac()
    obs = th.randn(16, OBS_DIM)
    with th.no_grad():
        a, logp = m.actor.action_log_prob(obs)
        assert th.allclose(m._log_pi(obs, a), logp.reshape(-1, 1), atol=2e-3)


def test_t25_decision_applies_the_preregistered_criteria():
    from experiments.t2_decide import decide_t25
    ks = (0, 500, 1000, 1500, 2000, 2500, 3000)
    arm = lambda xs: {str(k): {'mean_return': x} for k, x in zip(ks, xs)}
    pair = lambda m, u=None, cm=23, cs=23, um=0, us=0: {'mean': m, 'urgent': m if u is None else u, 'without_top2': m,
                                                        'capture_mve': cm, 'capture_sac': cs, 'unsafe_mve': um, 'unsafe_sac': us}
    def build(diffs, urgent=None, final=None):
        r = {'all_complete': True, 'arms': {}, 'paired': {}}
        for s in (0, 1, 2):
            r['arms'][f'e_mve_s{s}'] = arm([-15, 20, 35, 45, 50, 53, 54]); r['arms'][f'e_sac_s{s}'] = arm([-15, 15, 30, 42, 48, 53, 54])
            for k in ks[1:]:
                r['paired'][f's{s}/step_{k}'] = pair(diffs[s] if not (final is not None and k == 3000) else final, urgent)
        return r
    assert decide_t25(build([4, 3, -1]))['verdict'] == 'pass'
    assert decide_t25(build([4, 3, -1], urgent=-2))['verdict'] == 'partial'
    assert decide_t25(build([-4, 3, -1]))['verdict'] == 'no_gain'
    assert decide_t25(build([4, 3, 1], final=-8))['verdict'] == 'harmful'
    r = build([4, 3, -1]); r['all_complete'] = False
    assert decide_t25(r)['verdict'] == 'incomplete'


def test_t25_validation_mode_two_seeds_two_checkpoints():
    from experiments.t2_decide import decide_t25
    pair = lambda m: {'mean': m, 'urgent': m, 'without_top2': m, 'capture_mve': 23, 'capture_sac': 23, 'unsafe_mve': 0, 'unsafe_sac': 0}
    r = {'all_complete': True, 'arms': {}, 'paired': {}}
    for s, d in ((0, 3.0), (1, -1.0)):
        r['arms'][f'e_mve_s{s}'] = {'0': {'mean_return': -15}, '1000': {'mean_return': 40}, '2000': {'mean_return': 52}}
        r['arms'][f'e_sac_s{s}'] = {'1000': {'mean_return': 38}, '2000': {'mean_return': 50}}     # SAC 0k not required
        for k in (1000, 2000):
            r['paired'][f's{s}/step_{k}'] = pair(d)
    assert decide_t25(r, (1000, 2000), 2, False)['verdict'] == 'pass'      # 12 change > 10 would fail C if required
    assert decide_t25(r, (1000, 2000), 2, True)['verdict'] == 'partial'
    assert decide_t25(r, (1000, 2000), 3, False)['verdict'] == 'incomplete'


def test_v25_validation_decision():
    from experiments.t2_decide import decide_v25
    arm = lambda xs: {str(k): {'mean_return': x} for k, x in zip((0, 1000, 2000, 3000, 4000), xs)}
    pair = lambda m: {'mean': m, 'urgent': m, 'without_top2': m, 'capture_mve': 23, 'capture_sac': 23, 'unsafe_mve': 0, 'unsafe_sac': 0}
    r = {'all_complete': True, 'arms': {'e_mve_s0': arm([-15, 40, 50, 55, 56]), 'e_sac_s0': arm([-15, 36, 45, 55, 54]),
                                        'e_mve_s1': arm([None, 30, 45, 50, 55]), 'e_sac_s1': arm([None, 26, 44, 50])},
         'paired': {**{f's0/step_{k}': pair(m) for k, m in ((1000, 4), (2000, 5), (3000, 0), (4000, 2))},
                    **{f's1/step_{k}': pair(m) for k, m in ((1000, 4), (2000, 1), (3000, 0))}}}
    good = {'s0': {'two_stage_share': 0.8, 'decisive_share': 0.05, 'c_mean': 0.9, 'grad_cos': 0.3}}
    assert decide_v25(r, good)['verdict'] == 'pass'
    assert decide_v25(r, {'s0': dict(good['s0'], decisive_share=0.0)})['verdict'] == 'mechanism_failure'
    r['paired']['s1/step_3000'] = pair(-20)
    assert decide_v25(r, good)['verdict'] == 'mixed'


def test_stage2_rows_are_one_step_labels_without_a_ratio():
    """T2.5.1: Q(o1, a2) <- r2 + g (1 - d2) V(o2) for every non-terminal candidate with a stage 2, whatever pi/mu or the
    label age; weight 1 / (1 + (eps2 / sigma)^2), eps2 = |e_a| + b2; none after a terminal stage 1."""
    from train.t2_mve_sac import AGE_CAP
    g = __import__('env.t2_stage', fromlist=['GAMMA_STAGE']).GAMMA_STAGE
    one = th.ones(1, 1, dtype=th.bool)
    for logp, created in ((-1.0, 0.0), (-1.0 + np.log(1e4), 0.0), (-1.0, -float(AGE_CAP) - 1)):   # c = 1, c = 1e-4, too old
        m, b = _retrace_fixture(c2_logp=logp, created=created)
        m.td_ms = 100.0
        o1, a2, y2, w2, gap, d2 = m._stage2_rows(b, one, th.full((1,), 3.0), 0.05)
        assert abs(float(y2) - (3 + g * 7)) < 1e-5 and abs(float(w2) - 1.0 / (1.0 + 9.0 / 100.0)) < 1e-6
        assert float(gap) == 0.0                                 # Qbar is constant in the fixture
    m, b = _retrace_fixture(c2_logp=-1.0, d2=1.0)                # stage 2 ends in the entry: no bootstrap beyond it
    m.td_ms = 100.0; m.entry_bias_now = -20.0; b['c2_kind'] = th.full((1, 1), 2.0)
    _, _, y2, w2, _, _ = m._stage2_rows(b, one, th.full((1,), 3.0), 0.05)
    assert abs(float(y2) - 3.0) < 1e-6 and abs(float(w2) - 1.0 / (1.0 + 23.0 ** 2 / 100.0)) < 1e-6
    m, b = _retrace_fixture(c2_logp=-1.0, d1=1.0)                # terminal stage 1: no stage-2 row
    assert m._stage2_rows(b, one, th.zeros(1), 0.05) is None


def test_model_loss_averages_over_stage1_and_stage2_rows():
    """The model loss is the weighted mean over all model rows; the stage-2 rows enter at the predicted stage-1 state."""
    import inspect, train.t2_mve_sac as t
    src = inspect.getsource(t.T2SAC._model_loss)
    assert 'self.critic(o1, a2)' in src and 'n_rows += int(o1.shape[0])' in src and 'loss = 0.5 * total / max(n_rows, 1)' in src
    assert t.LEARNER_VERSION == 't2-learner-2.5.1'


def test_v251_validation_decision_and_cli(tmp_path):
    import json, subprocess, sys
    from experiments.t2_decide import decide_v251
    arm = lambda xs: {str(k): {'mean_return': x} for k, x in zip((0, 1000, 2000, 3000, 4000), xs)}
    pair = lambda m: {'mean': m, 'urgent': m, 'without_top2': m, 'capture_mve': 23, 'capture_sac': 23, 'unsafe_mve': 0, 'unsafe_sac': 0}
    r = {'all_complete': True, 'arms': {'e_mve_s0': arm([-15, 40, 50, 55, 56]), 'e_sac_s0': arm([-15, 36, 45, 55, 54]),
                                        'e_mve_s1': arm([None, 30, 45, 50, 55]), 'e_sac_s1': arm([None, 26, 44, 50])},
         'paired': {**{f's0/step_{k}': pair(m) for k, m in ((1000, 4), (2000, 5), (3000, 0), (4000, 2))},
                    **{f's1/step_{k}': pair(m) for k, m in ((1000, 4), (2000, 1), (3000, 0))}}}
    good = {'s0': {'two_stage_share': 0.8, 'decisive_share': 0.05, 'c_mean': 0.15, 'w2_mean': 0.8, 'pi_gap_sigma_median': 0.05, 'grad_cos': 0.3}}
    assert decide_v251(r, good)['verdict'] == 'pass'                                     # a low c no longer fails M
    assert decide_v251(r, {'s0': dict(good['s0'], w2_mean=0.3)})['verdict'] == 'mechanism_failure'
    assert decide_v251(r, {'s0': dict(good['s0'], pi_gap_sigma_median=0.4)})['verdict'] == 'mechanism_failure'
    assert decide_v251(r, {})['verdict'] == 'mechanism_failure'
    # the command line applies the named validation criteria with the training logs (v25 was not wired before T2.5.1)
    (tmp_path / 'TRAIN_e_mve_s0').mkdir()
    (tmp_path / 'TRAIN_e_mve_s0' / 'model_stats.json').write_text(json.dumps([dict(good['s0'], steps=1500)]))
    (tmp_path / 'R.json').write_text(json.dumps(r))
    for crit, verdict in (('v251', 'pass'), ('v25', 'mechanism_failure')):
        subprocess.run([sys.executable, '-B', '-m', 'experiments.t2_decide', '--criteria', crit, '--readout', str(tmp_path / 'R.json'),
                        '--train-root', str(tmp_path), '--output', str(tmp_path / 'D.json')], check=True, capture_output=True)
        assert json.loads((tmp_path / 'D.json').read_text())['verdict'] == verdict


def test_training_step_fits_stage2_rows_end_to_end():
    """Replay with two-stage candidates: train() runs, the stage-2 rows enter the loss and their readings are logged."""
    from env.t2_stage import OBS_DIM
    rng = np.random.default_rng(0)
    m = _sac()
    for k in range(16):
        fact = {'reward': float(rng.normal(0, 3)), 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': False}
        s2 = {'a': np.full(2, .2, np.float32), 'logp': -1.0, 'reward': 1.0, 'next_obs': np.ones(OBS_DIM, np.float32), 'done': bool(k % 4 == 0), 'kind': 0}
        cands = [{'a': np.full(2, .1, np.float32), 'reward': -1.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': False, 's2': s2},
                 {'a': np.full(2, -.1, np.float32), 'reward': -2.0, 'next_obs': np.zeros(OBS_DIM, np.float32), 'done': True, 's2': None}]
        m.replay_buffer.add(np.zeros((1, OBS_DIM)), np.zeros((1, OBS_DIM)), np.zeros((1, 2)), np.array([-1.0]), np.array([False]),
                            [{'model': {'fact': fact, 'cands': cands, 'diag': {'reason': 'stage'}}}])
    assert m.replay_buffer.model_counts['two_stage'] == 16
    m.train(gradient_steps=30, batch_size=8)
    used = [s for s in m.model_stats if s.get('model_td') is not None]
    assert used and all(np.isfinite(s['model_td']) for s in used)
    s = used[-1]
    assert s['stage2_rows'] > 0 and 0 < s['w2_mean'] <= 1 and s['pi_gap_sigma_median'] >= 0 and 0 <= s['stage2_entry_share'] <= 1
