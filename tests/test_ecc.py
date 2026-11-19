"""The ECC method's environment, execution chain, coordination, potential and learner (``docs/final_method/ROUND2.md``)."""
import numpy as np
import pytest
import torch as th

from env.t2_stage import (GAMMA_STAGE, OBS_DIM, OBS_FEATURES, STAGE_S, InfoEnv, T2Mission, T2StageEnv, d_authority, observe,
                          spin_frame, stage_goal)


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


def test_observation_is_the_17_value_policy_view():
    """One observation for actor and critics: the first draft's policy view without its zeroed entries and the mission
    class; values unchanged."""
    from env.t1_p_stage import relative_inertial
    env = T2StageEnv(openings=[1000004]); o, _ = env.reset(options={'opening': 1000004})
    o, *_ = env.step(np.array([0.5, 0.2]))
    q, v = relative_inertial(env.env); t = env.env.target_state
    assert OBS_DIM == 17 == len(OBS_FEATURES) and o.shape == (17,) and o.dtype == np.float32
    assert env.observation_space.shape == (17,)
    assert np.array_equal(o, observe(env.env, q, v, env.env.time_s, env.env._delta_v, env.mission, t.rotation, t.omega))
    f = spin_frame(t.rotation, t.omega, q, env.env.config.task.approach_axis)
    assert o[0] == np.float32(f['r'] / 15.0) and o[9] == np.float32((env.mission['deadline_s'] - env.env.time_s) / 300.0)
    assert o[11] == np.float32((env.mission['budget_m_s'] - env.env._delta_v) / 3.0)


# ---------------------------------------------------------------------- the task
def test_the_task_is_time_limited_capture_with_the_first_draft_deadlines():
    m = T2Mission()
    for op in range(294100, 294160):
        x = m.sample(op)
        rng = np.random.default_rng(np.random.SeedSequence([op, 290021])); rng.integers(0, 2)
        assert x == {'deadline_s': float(round(rng.uniform(140.0, 200.0), 1)), 'budget_m_s': 3.0,
                     'time_cost_per_s': 0.10, 'fuel_cost_per_m_s': 10.0}
    env = T2StageEnv(openings=[294100], shaping=False); env.reset(options={'opening': 294100})
    assert env.env.config.max_time_s == env.mission['deadline_s'] and env.env.config.reward.time_cost_per_s == 0.10
    assert env.env.config.reward.delta_v_cost_per_m_s == 10.0


def test_pure_row_is_the_mpc_alone():
    env = T2StageEnv(openings=[1000000], shaping=False)
    env.reset(options={'opening': 1000000})
    for _ in range(2):
        env.step_pure()
    assert env.far_field_controls == 0 and env.generator is None


# ---------------------------------------------------------------------- opportunity-reachability potential
def test_nominal_entry_time_is_the_approach_to_the_entry_range_plus_the_shortest_arrival():
    from env.t2_stage import BRIDGE_TAUS_S, nominal_entry_time
    task = T2StageEnv(openings=[294000]).env.config.task
    t16, t14, t12 = (nominal_entry_time(r, 0.0, task) for r in (16.0, 14.0, 12.0))
    assert t12 == min(BRIDGE_TAUS_S) == nominal_entry_time(8.0, 0.0, task)        # inside the entry range: the transfer only
    assert t16 - t14 == pytest.approx(2.0 / 0.20, abs=0.6) and t14 - t12 == pytest.approx(2.0 / 0.20, abs=0.6)
    assert nominal_entry_time(12.0, 60.0, task) > t12                            # the latitude law has to act first
    assert nominal_entry_time(10.0, 0.0, task) < nominal_entry_time(10.0, 80.0, task)


def test_time_to_entry_follows_the_pass_windows_the_sweep_lever_and_the_deadline():
    """Phase arithmetic of section 3 on a constructed geometry: spin about z, port tilted 90 deg from the spin axis."""
    from env.t2_stage import AZIMUTH_M_S, nominal_entry_time, time_to_entry, window_half_width
    from dynamics.lie import so3_exp
    env = T2StageEnv(openings=[294000]); task = env.env.config.task
    w = np.radians(2.4); omega = np.array([0.0, 0.0, w])
    port = lambda rot: rot @ task.approach_axis
    for phase_deg in (0.0, 60.0, 200.0, 350.0):
        rot = so3_exp(np.array([np.radians(90.0), 0.0, 0.0]))         # port axis in the x-y plane
        rot = so3_exp(np.array([0.0, 0.0, np.radians(phase_deg)])) @ rot
        q = 12.0 * np.array([1.0, 0.0, 0.0])
        f = spin_frame(rot, omega, q, task.approach_axis)
        beta = np.radians(f['beta_deg']); half = window_half_width(beta, beta, np.radians(35.0))
        kappa = AZIMUTH_M_S / (w * 12.0 * np.sin(beta))
        t_m = nominal_entry_time(12.0, f['lat_err_deg'], task)
        phi = f['phase_rad']
        expect = None
        for k in range(-1, 4):
            c = -2 * np.pi * k
            lo, hi = max(t_m, (phi - c - half) / (w * (1 + kappa))), (phi - c + half) / (w * (1 - kappa))
            if lo <= hi:
                expect = lo; break
        assert time_to_entry(q, rot, omega, task, 1e4) == pytest.approx(expect)
        assert time_to_entry(q, rot, omega, task, expect - 0.5) is None or time_to_entry(q, rot, omega, task, expect - 0.5) < expect
        assert time_to_entry(q, rot, omega, task, t_m - 1.0) is None                # the deadline leaves no arrival
    assert 0.0 < kappa < 1.0
    far = 16.0 * np.array([1.0, 0.0, 0.0])
    assert time_to_entry(far, rot, omega, task, 1e4) >= time_to_entry(q, rot, omega, task, 1e4)


def test_the_sweep_lever_brings_a_late_window_within_reach():
    """Without the along-sweep lever the next window would open after the latest arrival; meeting the port advances it."""
    import env.t2_stage as T
    from dynamics.lie import so3_exp
    task = T2StageEnv(openings=[294000]).env.config.task
    w = np.radians(2.4); omega = np.array([0.0, 0.0, w])
    rot = so3_exp(np.array([0.0, 0.0, np.radians(-100.0)])) @ so3_exp(np.array([np.radians(90.0), 0.0, 0.0]))
    q = 12.0 * np.array([1.0, 0.0, 0.0])
    f = spin_frame(rot, omega, q, task.approach_axis)
    beta = np.radians(f['beta_deg']); half_s = T.window_half_width(beta, beta, np.radians(35.0)) / w
    opens = f['time_to_pass_s'] - half_s
    t_go = T.time_to_entry(q, rot, omega, task, opens - 10.0)
    assert t_go is not None and t_go < opens - 10.0
    saved = T.AZIMUTH_M_S
    try:
        T.AZIMUTH_M_S = 0.0
        assert T.time_to_entry(q, rot, omega, task, opens - 10.0) is None
    finally:
        T.AZIMUTH_M_S = saved


def test_potential_is_bounded_and_the_first_draft_potential_is_kept():
    from env.t1_p_stage import relative_inertial
    from dynamics.lie import so3_exp
    from env.t2_stage import ENTRY_RADIUS_M, POTENTIAL_CAP
    for kind in ('opportunity', 'lead24'):
        env = T2StageEnv(openings=[1000004], potential=kind); env.reset(options={'opening': 1000004})
        phi = env._phi()
        assert -POTENTIAL_CAP <= phi <= 0.0
    q, _ = relative_inertial(env.env); t = env.env.target_state
    mouth = t.rotation @ so3_exp(24.0 * t.omega) @ (ENTRY_RADIUS_M * env.env.config.task.approach_axis)
    assert phi == -min(25.0, float(np.linalg.norm(q - mouth)))
    env.mission = dict(env.mission, deadline_s=env.env.time_s + 50.0)            # no admissible arrival left
    env.potential_kind = 'opportunity'
    assert env._phi() == -POTENTIAL_CAP
    with pytest.raises(ValueError):
        T2StageEnv(openings=[1000004], potential='lead30')


def test_shaped_reward_is_potential_based_with_terminal_zero():
    env = T2StageEnv(openings=[1000004]); env.reset(options={'opening': 1000004})
    phi0 = env._phi()
    _, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
    phi1 = 0.0 if term else env._phi()
    assert r == pytest.approx(info['true_reward'] + GAMMA_STAGE * phi1 - phi0, abs=1e-12)


# ---------------------------------------------------------------------- ECC: mode prediction and coordination
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


class _Own:
    """A fabricated mode prediction of the intent itself."""

    def __init__(self, ev):
        self.ev, self.calls = ev, 0

    def probe_intent(self, x, *args):
        self.calls += 1
        return {'event': self.ev, 't': 20.0 if self.ev else 48.0, 'kind': 'entry' if self.ev else None, 'dv': 0.1}


def _coordination_env(**kw):
    from env.t2_probe import PROBE_GRID
    env = T2StageEnv(openings=[1000004], coordination=True, **kw); env.reset(options={'opening': 1000004})
    rows = [{'event': 0, 't': 48.0, 'dv': 0.1, 'kind': None, 'critical': False} for _ in PROBE_GRID]

    def set_rows(critical):
        rs = [dict(r) for r in rows]
        for i, t in critical.items():
            rs[i] = {'event': 1, 't': t, 'dv': 0.2, 'kind': 'entry', 'critical': True}
        env._opp = {'t': float(env.env.time_s), 'episode': env.episode, 'args': (), 'rows': rs, 'period': 150.0, 'latest': 100.0}
    return env, set_rows, PROBE_GRID


def test_coordination_map_on_fabricated_executor_answers(monkeypatch):
    """Keep when the intent's own prediction has an event; otherwise the nearest critical grid intent (ties: the earlier
    event); otherwise the direct entry at the last usable pass; otherwise keep. Off by default."""
    assert T2StageEnv(openings=[1000004]).coordination is False
    env, set_rows, grid = _coordination_env()
    i_near, i_far, i_tie = grid.index((0.0, 0.0)), grid.index((1.0, 1.0)), grid.index((0.0, 1.0))
    launched = []
    monkeypatch.setattr(env, 'start_direct_gstar', lambda: launched.append(1) or False)
    monkeypatch.setattr(env, '_pass_clock', lambda: (150.0, 1e9))                # another pass always remains
    env.prober = _Own(0)
    set_rows({i_near: 30.0, i_far: 10.0})
    x, rec = env._coordinate(np.array([0.2, 0.1])); assert rec['kind'] == 'correct' and tuple(x) == (0.0, 0.0)
    set_rows({i_far: 20.0, i_tie: 12.0})
    x, rec = env._coordinate(np.array([0.5, 1.0])); assert tuple(x) == (0.0, 1.0) and rec['alt_t'] == 12.0
    set_rows({})
    x, rec = env._coordinate(np.array([0.2, 0.1])); assert rec['kind'] == 'none' and tuple(x) == (0.2, 0.1)
    env.prober = _Own(1); set_rows({i_near: 30.0})
    x, rec = env._coordinate(np.array([0.2, 0.1])); assert rec['kind'] == 'keep' and tuple(x) == (0.2, 0.1)
    assert env.stats['corrections'] == 2 and not launched


def test_direct_entry_only_at_the_last_usable_pass_and_after_the_grid(monkeypatch):
    import env.t2_stage as T
    env, set_rows, grid = _coordination_env()
    env.prober = _Own(0); set_rows({})
    launched = []

    def launch():
        launched.append(env.env.time_s)
        env.generator.bridge.plan = {'tau_s': 40.0}
        return True
    monkeypatch.setattr(env, 'start_direct_gstar', launch)
    t = float(env.env.time_s)
    clock = {'ttp': 10.0, 'latest': t + 10.0 + 150.0 - 1.0}
    monkeypatch.setattr(env, '_pass_clock', lambda: (150.0, clock['latest']))
    monkeypatch.setattr(T, 'frame_now', lambda e, q: {'time_to_pass_s': clock['ttp']})
    x, rec = env._coordinate(np.array([0.2, 0.1]))
    assert rec['kind'] == 'launch' and len(launched) == 1 and env.stats['long_entries'] == 1
    clock['ttp'] = STAGE_S + 5.0                                                 # the pass is not within this stage
    assert env._coordinate(np.array([0.2, 0.1]))[1]['kind'] == 'none'
    clock['ttp'], clock['latest'] = 10.0, t + 10.0 + 150.0 + 1.0                 # another pass remains before L
    assert env._coordinate(np.array([0.2, 0.1]))[1]['kind'] == 'none'
    clock['latest'] = t + 10.0 + 150.0 - 1.0
    set_rows({grid.index((0.0, 0.0)): 30.0})                                     # a critical grid intent comes first
    assert env._coordinate(np.array([0.2, 0.1]))[1]['kind'] == 'correct'
    env.long_entry = False; set_rows({})
    assert env._coordinate(np.array([0.2, 0.1]))[1]['kind'] == 'none' and len(launched) == 1


def test_without_correction_only_the_intent_is_predicted(monkeypatch):
    """'w/o core': keep and direct entry only; the grid is not evaluated."""
    env, set_rows, grid = _coordination_env(correction=False)
    monkeypatch.setattr(env, '_opportunity_eval', lambda: (_ for _ in ()).throw(AssertionError('grid evaluated')))
    monkeypatch.setattr(env, '_pass_clock', lambda: (150.0, 1e9))
    env.prober = _Own(0)
    x, rec = env._coordinate(np.array([0.2, 0.1]))
    assert rec['kind'] == 'none' and tuple(x) == (0.2, 0.1) and env.prober.calls == 1 and 'cands' not in rec


def test_critical_set_needs_a_last_chance_and_the_resource():
    """A real evaluation: with the budget already spent nothing is critical; with a deadline far away nothing is."""
    env = T2StageEnv(openings=[1000004]); env.reset(options={'opening': 1000004})
    env.mission = dict(env.mission, budget_m_s=float(env.env._delta_v))
    assert not any(d['critical'] for d in env._opportunity_eval()['rows'])
    env._opp = None; env.prober = None
    env.mission = dict(env.mission, budget_m_s=3.0, deadline_s=env.env.time_s + 2000.0)
    assert not any(d['critical'] for d in env._opportunity_eval()['rows'])


def test_coordination_and_audit_records_reach_the_step_info():
    env = T2StageEnv(openings=[1000004], coordination=True); env.audit_logging = True
    env.reset(options={'opening': 1000004})
    _, _, _, _, info = env.step(np.array([0.2, 0.1]))
    coord, audit = info['coord'], info['decision_audit']
    assert set(coord) >= {'a', 'own_event', 'own_t', 'kind', 'cands', 'own_kind', 'own_dv', 'wall_s'} and len(coord['cands']) == 9
    assert set(audit) == {'t_start', 'remaining_s', 'remaining_dv', 'phase_rad', 'time_to_pass_s', 'lat_err_deg', 'macro_end', 'true_mode_event_t'}


# ---------------------------------------------------------------------- learner
def test_learner_settings():
    import inspect
    import train.t2_mve_sac as m
    src = inspect.getsource(m.make_model)
    assert 'gamma=GAMMA_STAGE' in src and abs(GAMMA_STAGE - 0.999 ** 8) < 1e-15
    assert 'gradient_steps=max(1, int(round(updates_per_transition)))' in src and m.UPDATES_PER_TRANSITION == 2
    assert 'model.train(gradient_steps=model.gradient_steps' in inspect.getsource(m.train)
    model = m.make_model(m.SpacesEnv(OBS_DIM), 'sac', 0)
    assert model.ent_coef == m.ENT_COEF == 0.05 and model.batch_size == 256 and model.learning_starts == 200 and model.tau == 0.005
    assert model.policy.net_arch == [256, 256] or model.policy.net_arch == {'pi': [256, 256], 'qf': [256, 256]}
    with pytest.raises(ValueError):
        m.make_model(m.SpacesEnv(OBS_DIM), 'e_priv', 0)


def test_actor_and_critics_read_the_same_observation(tmp_path):
    import train.t2_mve_sac as t
    env = T2StageEnv(openings=[294003]); o = env.reset(options={'opening': 294003})[0]
    x = th.as_tensor(np.stack([o, o])); x[1, -1] += 1.0
    a = th.zeros((2, 2))
    m = t.make_model(t.SpacesEnv(OBS_DIM), 'sac', 0, smoke=True)
    with th.no_grad():
        assert not th.allclose(m.actor.get_action_dist_params(x)[0][0], m.actor.get_action_dist_params(x)[0][1])
        assert not th.allclose(*m.critic(x, a)[0])
    assert m.actor.features_extractor.features_dim == OBS_DIM == m.critic.features_extractor.features_dim
    m.save(tmp_path / 'sac.zip'); m2 = t.T2SAC.load(tmp_path / 'sac.zip', device='cpu')
    with th.no_grad():
        assert th.allclose(m2.actor.get_action_dist_params(x)[0], m.actor.get_action_dist_params(x)[0])


def test_sac_update_runs_on_stored_transitions():
    import train.t2_mve_sac as t
    from stable_baselines3.common.logger import configure
    m = t.make_model(t.SpacesEnv(OBS_DIM), 'sac', 0, smoke=True)
    m.set_logger(configure(None, [])); m._current_progress_remaining = 1.0
    rng = np.random.default_rng(0)
    for i in range(32):
        o = rng.normal(size=(1, OBS_DIM)).astype(np.float32)
        m.replay_buffer.add(o, o, rng.uniform(-1, 1, size=(1, 2)).astype(np.float32), np.array([-1.0]), np.array([i % 7 == 0]), [{}])
    before = [p.clone() for p in m.critic.parameters()]
    m.train(gradient_steps=2, batch_size=8)
    assert m._n_updates == 2 and m.replay_buffer.model_counts['transitions'] == 32 and len(m.model_stats) == 1
    assert any(not th.equal(p, q) for p, q in zip(before, m.critic.parameters()))


def test_tensorboard_log_records_episodes_and_updates(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import train.t2_mve_sac as t
    log = t.TrainingLog(tmp_path / 'tb')
    log.episode(10, {'event': 'success', 'return_true': 50.0, 'return_shaped': 60.0, 'time_s': 120.0, 'delta_v': 1.2,
                     'exec_stats': {'corrections': 2, 'long_entries': 1}})
    log.episode(20, {'event': 'timeout', 'return_true': -120.0, 'return_shaped': -110.0, 'time_s': 150.0, 'delta_v': 1.0})
    log.updates(20, [{'real_td': 1.0, 'actor_loss': -2.0, 'q_pi_mean': 3.0}])
    log.close()
    acc = EventAccumulator(str(tmp_path / 'tb')); acc.Reload()
    tags = set(acc.Tags()['scalars'])
    assert {'episode/capture', 'episode/return_true', 'episode/corrections', 'rolling/capture_rate_100', 'train/real_td'} <= tags
    assert [e.value for e in acc.Scalars('rolling/capture_rate_100')] == [1.0, 0.5]
    assert [e.step for e in acc.Scalars('episode/outcome')] == [10, 20]


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
