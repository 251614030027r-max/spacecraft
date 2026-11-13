"""The ECC method's environment, execution chain, certification and learner (``docs/paper/METHOD.md``)."""
import numpy as np
import pytest
import torch as th

from env.t2_stage import (EXEC_STATE_FEATURES, GAMMA_STAGE, OBS_DIM, OBS_DIM_PRIV, PLANNER_FEATURES, STAGE_S, InfoEnv,
                          T2Mission, T2StageEnv, d_authority, mission_from_name, observe, spin_frame, stage_goal)


def _frame():
    # spin about +z, line of sight along +x, port axis tilted 30 deg from the spin axis towards -y
    rot = np.eye(3); omega = np.array([0.0, 0.0, np.radians(2.4)])
    axis = np.array([0.0, -np.sin(np.radians(30)), np.cos(np.radians(30))])
    return spin_frame(rot, omega, np.array([12.0, 0.0, 0.0]), axis), axis


# ---------------------------------------------------------------------- RWG geometry and the stage layer
def test_spin_frame_geometry():
    f, _ = _frame()
    b = f["basis"]
    assert np.allclose(b @ b.T, np.eye(3), atol=1e-9)
    assert np.allclose(b[1], [0, 0, 1]) and abs(f["lam_deg"] - 90) < 1e-9 and abs(f["beta_deg"] - 30) < 1e-9
    assert abs(f["lat_err_deg"] - 60) < 1e-9
    # the port's projection points along -y; rotating about +z it reaches +x after 90 deg (= 37.5 s at 2.4 deg/s)
    assert abs(np.degrees(f["phase_rad"]) - 90) < 1e-6 and abs(f["time_to_pass_s"] - 37.5) < 1e-6
    assert np.allclose(b[2], [0, -1, 0])      # towards the oncoming port


@pytest.mark.parametrize("lam_deg", [60.0, 90.0, 120.0])
@pytest.mark.parametrize("v_z", [-0.15, 0.15])
def test_phase_and_latitude_rates_match_the_kinematics(lam_deg, v_z):
    """METHOD.md 2.2: d phi / dt = -|omega| - v_z / (r sin lam), d lam / dt = -v_l / r (finite differences)."""
    w = np.radians(2.4); omega = np.array([0.0, 0.0, w]); axis = np.array([np.sin(np.radians(40)), 0.0, np.cos(np.radians(40))])
    lam = np.radians(lam_deg); r = 10.0
    q = r * np.array([np.sin(lam) * np.cos(1.0), np.sin(lam) * np.sin(1.0), np.cos(lam)])
    f = spin_frame(np.eye(3), omega, q, axis)
    v_l = 0.05
    v = v_z * f["basis"][2] + v_l * f["basis"][1]
    h = 1e-3
    rot_h = np.array([[np.cos(w * h), -np.sin(w * h), 0], [np.sin(w * h), np.cos(w * h), 0], [0, 0, 1]])
    g = spin_frame(rot_h, omega, q + h * v, axis)
    dphi = ((g["phase_rad"] - f["phase_rad"] + np.pi) % (2 * np.pi) - np.pi) / h
    dlam = np.radians(g["lam_deg"] - f["lam_deg"]) / h
    assert abs(dphi - (-w - v_z / (r * np.sin(lam)))) < 1e-4
    assert abs(dlam - (-v_l / r)) < 1e-4


def test_window_half_width_extremes():
    from math import pi
    from env.t2_stage import window_half_width
    assert window_half_width(0, pi / 6, pi / 4) == pi
    assert window_half_width(0, pi / 2, pi / 4) is None
    assert window_half_width(pi / 2, pi / 2, pi / 6) == pytest.approx(pi / 6)
    assert window_half_width(pi / 6, pi / 2, pi / 12) is None


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
    far = stage_goal(q, np.array([0.3, 0.0, 0.1]), f, np.array([1.0, 1.0]), mass)
    assert far['rate_limited'] and abs(np.linalg.norm(far['v_goal'] - np.array([0.3, 0.0, 0.1])) - 2 * d / STAGE_S) < 1e-9
    for a in ([0.3, -0.7], [-1.0, 1.0], [0.0, 0.0]):
        vt = target_velocity(f, a)
        assert np.allclose(action_for_velocity(f, -float(vt @ b[0]), float(vt @ b[2])), a)
    assert target_velocity(f, [1, 0]) @ b[0] < target_velocity(f, [-1, 0]) @ b[0]   # a_p > 0 closes in
    assert target_velocity(f, [0, 1]) @ b[2] > 0                                   # a_z > 0 meets the port earlier


def test_without_rotation_guidance_the_stage_closes_along_the_line_of_sight():
    from env.t2_stage import target_velocity
    f, _ = _frame(); b = f["basis"]
    v = target_velocity(f, np.array([0.4, 0.9]), geometry=False)
    assert np.allclose(v, -(0.05 + 0.15 * 0.4) * b[0])
    assert abs(target_velocity(f, np.array([0.4, 0.9])) @ b[1]) > 0      # with rotation guidance: cone regulation


def test_stage_goal_is_projected_onto_the_truth_safety_set():
    from env.t2_stage import STAGE_ENVELOPE_MARGIN_M_S, STAGE_RANGE_FLOOR_M
    env = T2StageEnv(); env.reset(); e = env.env; task = e.config.task; mass = e.chaser_parameters.mass
    rot, om = e.target_state.rotation, e.target_state.omega
    for r0, v_in in ((8.0, 0.18), (7.4, 0.15), (15.0, 0.10)):
        q = np.array([r0, 0.0, 0.0]); v = np.array([-v_in, 0.0, 0.0])
        f = spin_frame(rot, om, q, task.approach_axis)
        old = stage_goal(q, v, f, np.array([1.0, 0.0]), mass)
        g = stage_goal(q, v, f, np.array([1.0, 0.0]), mass, task=task)
        r_end = float(np.linalg.norm(g['q_goal'])); u = g['q_goal'] / r_end
        assert r_end >= STAGE_RANGE_FLOOR_M - 1e-9
        assert -float(g['v_goal'] @ u) <= task.outer_radial_closing_speed_limit(r_end) - STAGE_ENVELOPE_MARGIN_M_S + 1e-9
        if r0 >= 15.0:
            assert not g['safety_projected'] and np.allclose(g['v_goal'], old['v_goal'])
        else:
            assert g['safety_projected'] and np.linalg.norm(g['q_goal']) > np.linalg.norm(old['q_goal'])


def test_window_commitment_retimes_a_failed_replan_inside_the_pass():
    """RWG re-timing: a live entry whose fixed-time re-plan fails is re-timed (2, 4, 8 s later) at the plan bound,
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


def test_lost_entry_is_counted_and_the_episode_goes_on():
    env = T2StageEnv(openings=[294000], shaping=False)
    env.reset(options={'opening': 294000})
    env._bridge = {"t": env.env.time_s, "tau": 60.0}
    env._close_bridge("lost")
    assert env._bridge is None and env.stats['entries_lost'] == 1
    env._bridge = {"t": env.env.time_s, "tau": 25.0}
    env._close_bridge("success")
    assert env._bridge is None and env.stats['entries_lost'] == 1


# ---------------------------------------------------------------------- information boundary and interface
def test_info_env_hides_the_true_inertia_only():
    from env.t1_env import T1Env
    e = T1Env()
    e.reset(1000000)
    view = InfoEnv(e)
    true = np.asarray(e.target_parameters.inertia)
    iso = np.asarray(view.target_parameters.inertia)
    assert np.allclose(iso, np.eye(3) * np.trace(true) / 3) and not np.allclose(iso, true)
    assert view.time_s == e.time_s and view.target_state is e.target_state


def test_observation_is_the_policy_view_followed_by_the_executor_feedback():
    """cut_priv: the policy view zeroes the planner answers and the executor's internal state; z_E is the full vector."""
    from env.t1_p_stage import relative_inertial
    env = T2StageEnv(openings=[1000004]); o, _ = env.reset(options={'opening': 1000004})
    o, *_ = env.step(np.array([0.5, 0.2]))
    q, v = relative_inertial(env.env); t = env.env.target_state
    full = observe(env.env, q, v, env.env.time_s, env.env._delta_v, env.mission, t.rotation, t.omega, env.last)
    cut = list(PLANNER_FEATURES) + list(EXEC_STATE_FEATURES)
    keep = [i for i in range(OBS_DIM) if i not in cut]
    assert o.shape == (OBS_DIM_PRIV,) and not o[cut].any()
    assert np.array_equal(o[keep], full[keep]) and np.array_equal(o[OBS_DIM:], full)
    assert o[OBS_DIM + 16] in (0.0, 1.0) and env.observation_space.shape == (OBS_DIM_PRIV,)


# ---------------------------------------------------------------------- missions
def test_missions_keep_the_natural_deadlines_and_trade_time_against_fuel():
    nat = T2Mission(urgent_deadline_s=(140.0, 200.0))
    for op in range(294100, 294130):
        n = nat.sample(op)
        assert mission_from_name('natural').sample(op) == n
        assert mission_from_name('urgent' if n['urgent'] else 'economy').sample(op) == n
        u, e = mission_from_name('urgent').sample(op), mission_from_name('economy').sample(op)
        assert u['urgent'] and not e['urgent'] and 140.0 <= u['deadline_s'] <= 200.0 and 280.0 <= e['deadline_s'] <= 340.0
        assert u['time_cost_per_s'] > 0.05 and u['fuel_cost_per_m_s'] < 20 and e['time_cost_per_s'] < 0.05 and e['fuel_cost_per_m_s'] > 20
    share = np.mean([mission_from_name('urgent_mix:0.75').sample(o)['urgent'] for o in range(1_000_000, 1_002_000)])
    assert abs(share - 0.75) < 0.03
    with pytest.raises(ValueError):
        mission_from_name('scarcity')


def test_pure_row_is_the_mpc_alone():
    env = T2StageEnv(openings=[1000000], shaping=False)
    env.reset(options={'opening': 1000000})
    for _ in range(2):
        env.step_pure()
    assert env.far_field_controls == 0 and env.generator is None


# ---------------------------------------------------------------------- ECC: mode prediction and certification
def test_mode_prediction_gate_is_the_executor_rule_and_evaluation_leaves_execution_unchanged():
    """The prediction's gate reproduces the executor's metrics and descent region on real states; the opportunity
    evaluation uses its own planner instances, so an episode is the same whether or not it is evaluated."""
    from env.t1_p_stage import relative_inertial
    from env.t2_probe import executor_gate
    op = 1000006
    a = T2StageEnv(openings=[op]); b = T2StageEnv(openings=[op])
    a.reset(options={'opening': op}); b.reset(options={'opening': op})
    for k in range(4):
        a._opportunity_eval()                                  # evaluated, never applied
        for env in (a, b):
            _, _, term, trunc, info = env.step(np.array([1.0, 0.3 * (k % 2)]))
        ea, eb = a.env, b.env
        assert ea.time_s == eb.time_s and ea._delta_v == eb._delta_v and np.array_equal(ea.chaser_state.position, eb.chaser_state.position)
        q, v = relative_inertial(ea); m = ea.metrics()
        g = executor_gate(q, v, ea.target_state.rotation, ea.config.task, a.generator.bridge.descent_radius_m)
        assert abs(g['range_m'] - m.range_m) < 1e-6 and g['descending'] == a.generator.descending(a.info_env, m)
        assert abs(g['radial_margin_m_s'] - m.radial_envelope_margin_m_s) < 1e-6
        if term or trunc:
            break


def test_certification_map_on_fabricated_executor_answers():
    """C(x, a): the intent is kept when its own prediction has an event; otherwise the nearest certified probe
    (ties: the earlier event) is flown; nothing changes when no probe is certified; off by default."""
    from env.t2_probe import PROBE_GRID
    env = T2StageEnv(openings=[1000004]); env.reset(options={'opening': 1000004})
    assert env.certify is False
    rows = [{'event': 0, 't': 48.0, 'dv': 0.1, 'kind': None, 'certified': False} for _ in PROBE_GRID]
    i_near, i_far, i_tie = PROBE_GRID.index((0.0, 0.0)), PROBE_GRID.index((1.0, 1.0)), PROBE_GRID.index((0.0, 1.0))

    class Own:
        def __init__(self, ev): self.ev = ev
        def probe_intent(self, x, *args): return {'event': self.ev, 't': 20.0 if self.ev else 48.0, 'kind': 'entry' if self.ev else None, 'dv': 0.1}

    def set_rows(certified):
        rs = [dict(r) for r in rows]
        for i, t in certified.items():
            rs[i] = {'event': 1, 't': t, 'dv': 0.2, 'kind': 'entry', 'certified': True}
        env._opp = {'t': float(env.env.time_s), 'episode': env.episode, 'args': (), 'rows': rs, 'period': 150.0, 'latest': 100.0}
    env.cert_prober = Own(0)
    set_rows({i_near: 30.0, i_far: 10.0})
    x, rec = env._certify(np.array([0.2, 0.1])); assert rec['intervened'] and tuple(x) == (0.0, 0.0)
    set_rows({i_far: 20.0, i_tie: 12.0})
    x, rec = env._certify(np.array([0.5, 1.0])); assert tuple(x) == (0.0, 1.0) and rec['alt_t'] == 12.0
    set_rows({})
    x, rec = env._certify(np.array([0.2, 0.1])); assert not rec['intervened'] and tuple(x) == (0.2, 0.1)
    env.cert_prober = Own(1); set_rows({i_near: 30.0})
    x, rec = env._certify(np.array([0.2, 0.1])); assert not rec['intervened'] and tuple(x) == (0.2, 0.1)
    assert env.stats['cert_interventions'] == 2


def test_certified_set_needs_a_binding_pass_and_the_resource():
    """A real evaluation: with the budget already spent nothing is certified; with a deadline far away no pass binds."""
    env = T2StageEnv(openings=[1000004]); env.reset(options={'opening': 1000004})
    env.mission = dict(env.mission, budget_m_s=float(env.env._delta_v))
    assert not any(d['certified'] for d in env._opportunity_eval()['rows'])
    env._opp = None; env.cert_prober = None
    env.mission = dict(env.mission, budget_m_s=3.0, deadline_s=env.env.time_s + 2000.0)
    assert not any(d['certified'] for d in env._opportunity_eval()['rows'])


def test_certification_and_audit_records_reach_the_step_info():
    env = T2StageEnv(openings=[1000004], certify=True); env.audit_logging = True
    env.reset(options={'opening': 1000004})
    _, _, _, _, info = env.step(np.array([0.2, 0.1]))
    cert, audit = info['cert'], info['decision_audit']
    assert set(cert) >= {'a', 'own_event', 'own_t', 'intervened', 'cands', 'own_kind', 'own_dv', 'wall_s'} and len(cert['cands']) == 9
    assert set(audit) == {'t_start', 'remaining_s', 'remaining_dv', 'phase_rad', 'time_to_pass_s', 'lat_err_deg', 'macro_end', 'true_mode_event_t'}


# ---------------------------------------------------------------------- learner
def test_learner_settings():
    import inspect
    import train.t2_mve_sac as m
    src = inspect.getsource(m.make_model)
    assert 'gamma=GAMMA_STAGE' in src and abs(GAMMA_STAGE - 0.999 ** 8) < 1e-15
    assert 'gradient_steps=max(1, int(round(updates_per_transition)))' in src and m.UPDATES_PER_TRANSITION == 2
    assert 'model.train(gradient_steps=model.gradient_steps' in inspect.getsource(m.train)
    model = m.make_model(m.SpacesEnv(OBS_DIM_PRIV), 'e_priv', 0)
    assert model.ent_coef == m.ENT_COEF == 0.05 and model.batch_size == 256 and model.learning_starts == 200 and model.tau == 0.005
    with pytest.raises(ValueError):
        m.make_model(m.SpacesEnv(OBS_DIM_PRIV), 'e_acc', 0)


def test_critic_reads_the_executor_feedback_only_in_ecc_arms(tmp_path):
    """Every actor reads only the policy view; the e_priv critic reads z_E too, the e_sac / TD3 critics do not; the
    split survives save / load."""
    import train.t2_mve_sac as t
    env = T2StageEnv(openings=[294003]); o = env.reset(options={'opening': 294003})[0]
    x = th.as_tensor(np.stack([o, o]))
    x[1, OBS_DIM:] += 5.0                                            # change only the executor feedback
    a = th.zeros((2, 2))
    for arm, critic_sees in (('e_priv', True), ('e_sac', False)):
        m = t.make_model(t.SpacesEnv(OBS_DIM_PRIV), arm, 0, smoke=True)
        with th.no_grad():
            mu = m.actor.get_action_dist_params(x)[0]; q = m.critic(x, a)[0]
        assert th.allclose(mu[0], mu[1])                             # the deployed actor never reads it
        assert (not th.allclose(q[0], q[1])) == critic_sees
        m.save(tmp_path / f'{arm}.zip'); m2 = t.T2SAC.load(tmp_path / f'{arm}.zip', device='cpu')
        assert m2.policy.actor_dim == OBS_DIM and m2.policy.critic_dim == (OBS_DIM_PRIV if critic_sees else OBS_DIM)
        with th.no_grad():
            assert th.allclose(m2.actor.get_action_dist_params(x)[0], m.actor.get_action_dist_params(x)[0])
    td3 = t.make_model(t.SpacesEnv(OBS_DIM_PRIV), 'e_td3', 0, smoke=True)
    with th.no_grad():
        assert th.allclose(td3.actor(x)[0], td3.actor(x)[1]) and th.allclose(*td3.critic(x, a)[0])


def test_sac_update_runs_on_stored_transitions():
    import train.t2_mve_sac as t
    from stable_baselines3.common.logger import configure
    m = t.make_model(t.SpacesEnv(OBS_DIM_PRIV), 'e_priv', 0, smoke=True)
    m.set_logger(configure(None, [])); m._current_progress_remaining = 1.0
    rng = np.random.default_rng(0)
    for i in range(32):
        o = rng.normal(size=(1, OBS_DIM_PRIV)).astype(np.float32)
        m.replay_buffer.add(o, o, rng.uniform(-1, 1, size=(1, 2)).astype(np.float32), np.array([-1.0]), np.array([i % 7 == 0]), [{}])
    before = [p.clone() for p in m.critic.parameters()]
    m.train(gradient_steps=2, batch_size=8)
    assert m._n_updates == 2 and m.replay_buffer.model_counts['transitions'] == 32 and len(m.model_stats) == 1
    assert any(not th.equal(p, q) for p, q in zip(before, m.critic.parameters()))


# ---------------------------------------------------------------------- bitwise-neutral speed-ups
def test_fast_cross_and_norm_are_bitwise_numpy():
    from dynamics.fastmath import cross3, norm3
    rng = np.random.default_rng(7)
    a = rng.normal(size=(5000, 3)) * rng.choice([1e-12, 1e-3, 1.0, 1e5, 1e9], size=(5000, 1))
    b = rng.normal(size=(5000, 3)) * rng.choice([1e-9, 1.0, 1e6], size=(5000, 1))
    for x, y in zip(a, b):
        assert cross3(x, y).tobytes() == np.cross(x, y).tobytes()
        assert np.float64(norm3(x)).tobytes() == np.float64(np.linalg.norm(x)).tobytes()
    assert np.array_equal(cross3(np.eye(3), np.ones(3)), np.cross(np.eye(3), np.ones(3)))      # other shapes: numpy itself
    assert np.array_equal(cross3([1, 0, 0], [0, 1, 0]), np.cross([1, 0, 0], [0, 1, 0]))


def test_cvxpy_fast_path_stores_the_same_values_and_keeps_attribute_checks():
    import cvxpy as cp
    import controllers.mpc.cvxpy_fastpath as fast
    rng = np.random.default_rng(3)
    p, q = cp.Parameter((4, 3)), cp.Parameter((4, 3))
    for _ in range(50):
        v = rng.normal(size=(4, 3)) * 10.0 ** rng.integers(-8, 8)
        p.value = v
        q._value = fast._ORIGINAL(q, v)
        assert p.value.tobytes() == q.value.tobytes()
    with pytest.raises(ValueError):
        p.value = np.zeros((3, 4))
    nonneg = cp.Parameter(3, nonneg=True)
    with pytest.raises(ValueError):
        nonneg.value = np.array([1.0, -1.0, 0.0])                 # attributed leaves still go through CVXPY's check
