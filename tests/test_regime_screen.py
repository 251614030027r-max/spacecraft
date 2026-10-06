"""Regime screening: configs change only the named fields; rows; selection rule."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from experiments.regime_screen import regime_env, run_row, select, stage1_pass, summarize
from experiments.v3_handoff_scan import ImpulseMeter
from train.regimes import REGIMES, STAGE1, regime_environment
from train.stopping import stopping_configs


def test_regimes_change_only_tumble_and_initial_distribution() -> None:
    base, _ = stopping_configs()
    changed = {"phase2_target_tumble_scale", "precapture_initial_range_min_m",
               "precapture_initial_range_max_m", "precapture_initial_inertial_speed_max_m_s"}
    for regime in REGIMES.values():
        a, b = asdict(base), asdict(regime_environment(base, regime))
        assert {k for k in a if a[k] != b[k]} <= changed
    nominal = STAGE1[0]
    assert asdict(regime_environment(base, nominal)) == asdict(base)


def test_tumble_rate_is_realised() -> None:
    env = regime_env("w3.50_r18")
    env.reset(seed=269000)
    rate = np.degrees(np.linalg.norm(np.asarray(env.env.target_state.omega)))
    assert abs(rate - 3.5) < 1e-3
    assert np.linalg.norm(env._current_position()) >= 18.0 - 1e-9
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    pure = run_row(env, 269000, "pure", meter, max_decisions=2)
    nominal = run_row(env, 269000, "nominal", meter, max_decisions=2)
    env.close()
    assert pure["decisions"] == nominal["decisions"] == 2 and pure["row"] == "pure"


def _rows(clean: int, kinds: list[list[str]]) -> dict[int, dict]:
    rows = {s: {"clean_completion": True, "zero_violation": True, "failure": [], "survival_s": 90.0,
                "qp_zero_fallbacks": 0} for s in range(clean)}
    for i, kind in enumerate(kinds):
        rows[clean + i] = {"clean_completion": False, "zero_violation": True, "failure": kind, "survival_s": 300.0,
                           "qp_zero_fallbacks": 0}
    return rows


def test_stage1_pass_and_selection() -> None:
    ok = summarize(_rows(33, [["time_failure"]] * 12 + [["fov_failure"]] * 3))
    assert stage1_pass(ok)
    assert not stage1_pass(summarize(_rows(40, [["time_failure"]] * 8)))
    assert not stage1_pass(summarize(_rows(33, [["fov_failure"]] * 15)))
    cells = {"w3.50_r18": {"passes": True, "pure_fail_nominal_clean": 8},
             "w3.00_r20": {"passes": True, "pure_fail_nominal_clean": 3},
             "w3.00_r18": {"passes": True, "pure_fail_nominal_clean": 2},
             "w2.36_r15": {"passes": False, "pure_fail_nominal_clean": 9}}
    assert select(cells) == "w3.00_r20"  # closest to the mainline with rescue space >= 3
    assert select({"w3.00_r18": {"passes": True, "pure_fail_nominal_clean": 2}}) is None
