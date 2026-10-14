"""Beta diagnostics: planner wiring only (no closed-loop simulation)."""

from __future__ import annotations

from diagnostics.t1_beta.beta_probe import RecordingPlanner, SmoothRendezvousPlanner, install
from env.t1_gym import T1GStarEnv


def test_beta_zero_wraps_the_frozen_planner_and_beta_positive_adds_one_term():
    env = T1GStarEnv(openings=[295009])
    env.reset()
    frozen = env.generator.planner
    install(env, 0.0)
    assert isinstance(env.generator.planner, RecordingPlanner) and env.generator.planner.inner is frozen
    assert env.generator.planner.nodes == frozen.nodes and env.generator.planner.mass == frozen.mass
    env.reset()
    install(env, 1.0)
    smooth = env.generator.planner.inner
    assert isinstance(smooth, SmoothRendezvousPlanner) and smooth.problem.is_dpp()
    assert smooth.max_force_n == env.generator.recovery_planner.max_force_n * 0.8


def _run(beta, opening, ret, x0, event="success", dv=1.0, shift=0.0):
    grid = [[shift * t, 0.0, 0.0] for t in range(5)]
    return {"beta": beta, "opening": opening, "discounted_return": ret, "event": event, "time_s": 100.0,
            "delta_v_m_s": dv, "initial_observation": [x0] + [0.0] * 21, "numerical_failure": False,
            "planner_failures": 0, "min_margins": {"fov_margin_deg": 5.0},
            "new_plan_triggers": {"first": 1, "deviation": 0, "expired": 0, "other": 0},
            "far_field_delta_v_m_s": dv, "accepted_plans": [{"tau_s": 4.0, "position_on_grid_m": grid}]}


def test_rule_is_frozen_on_b1_and_only_applied_on_b2():
    from diagnostics.t1_beta.analyze_beta import BETAS, b1_checks, b2_validate, fit_rule

    runs = {}
    for o in range(8):  # low range prefers beta 3, high range prefers beta 0
        x0 = 1.0 if o < 4 else 2.0
        for b in BETAS:
            good = (b == 3.0) if o < 4 else (b == 0.0)
            runs[(b, o)] = _run(b, o, 80.0 + (3.0 if good else 0.0), x0, dv=1.0 + 0.01 * b, shift=0.1 * b)
    checks = b1_checks(runs, list(range(8)))
    assert checks[3.0]["acts"] and checks[3.0]["safe"] and not checks[0.3]["acts"]
    frozen = fit_rule(runs, list(range(8)), list(BETAS))
    rule = frozen["rule"]
    assert rule["kind"] == "rule" and rule["feature"] == "range_10m" and rule["threshold"] == 1.5
    assert (rule["beta_low"], rule["beta_high"]) == (3.0, 0.0)
    b2 = {(b, o + 100): dict(r, opening=o + 100) for (b, o), r in runs.items()}
    out = b2_validate(b2, [o + 100 for o in range(8)], frozen)
    assert out["rule_minus_best_fixed"] == 1.5 and out["recommend_sac_beta_design"]


def test_no_state_rule_when_nothing_beats_the_best_fixed_beta():
    from diagnostics.t1_beta.analyze_beta import BETAS, fit_rule

    runs = {(b, o): _run(b, o, 80.0 + (1.0 if b == 1.0 else 0.0), float(o)) for b in BETAS for o in range(8)}
    frozen = fit_rule(runs, list(range(8)), list(BETAS))
    assert frozen["best_fixed_beta"] == 1.0 and frozen["rule"]["kind"] == "fixed"
