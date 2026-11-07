"""Stage-B eta diagnostics: configuration plumbing only (no simulation)."""

from __future__ import annotations

from diagnostics.t1_eta.eta_probe import BASE_INPUT_WEIGHT, env_config, select_decisions
from env.t1_env import T1EnvConfig, t1_mpc_config


def test_eta_one_is_the_frozen_configuration():
    assert env_config(1.0) == T1EnvConfig()
    assert t1_mpc_config(env_config(10.0)).input_weight == 10.0 * BASE_INPUT_WEIGHT
    assert t1_mpc_config(env_config(10.0)).terminal_weight == t1_mpc_config(T1EnvConfig()).terminal_weight


def test_pre_registered_decision_selection():
    log = [{"k": k, "status": "kept", "force_saturated_fraction": 0.0} for k in range(30)]
    log[7]["force_saturated_fraction"] = 0.4
    for k in range(20, 30):
        log[k]["status"] = "descent"
    assert select_decisions(log) == {"plan_init": 1, "mid_approach": 10, "max_saturation": 7, "first_descent": 20}
    assert select_decisions(log[:3]) == {}
