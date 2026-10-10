"""Replay fidelity, continuation-value audit, counterfactual rollouts, paper tables."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch as th

from env.hybrid_env import PrecaptureHybridEnv
from experiments.final_tables import build, mcnemar_exact
from experiments.stopping_replay import FIDELITY_KEYS, continuation_values, replay_episode, summarize
from experiments.v3_handoff_scan import ImpulseMeter
from experiments.v3_stopping import run_stopping_episode
from train.hybrid_configs import hybrid_model_kwargs
from train.stopping import STOPPING, STOPPING_SAC, StoppingSAC, handoff_feature_index, stopping_configs


def _env() -> PrecaptureHybridEnv:
    environment, hybrid = stopping_configs()
    return PrecaptureHybridEnv(environment, hybrid)


def _model(env) -> StoppingSAC:
    kwargs = hybrid_model_kwargs(STOPPING_SAC)
    kwargs["policy_kwargs"]["net_arch"] = [16, 16]
    kwargs.update(learning_starts=0, buffer_size=1000)
    return StoppingSAC("MlpPolicy", env, stopping=asdict(STOPPING), seed=0, device="cpu",
                       handoff_feature_index=handoff_feature_index(env.policy_observation_slices()), **kwargs)


def test_replay_matches_the_evaluator_and_counterfactuals_run() -> None:
    env, env_b = _env(), _env()
    model = _model(env)
    with th.no_grad():  # never hand off on its own, so both boundary states are learned decisions
        for net in (model.stop.q_handoff_1, model.stop.q_handoff_2):
            net[-1].bias.fill_(-1.0e4)
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    formal = run_stopping_episode(env, model, 263004, "stopping", meter, max_decisions=3)
    replay = replay_episode(env, model, 263004, meter, env_b.controller, {0, 1}, max_decisions=3,
                            rollout_max_decisions=2)
    for key in FIDELITY_KEYS:
        assert replay[key] == formal[key], key
    assert [r["k"] for r in replay["records"]] == [0, 1, 2]
    assert np.allclose([r["beta"] for r in replay["records"]], formal["trace"]["beta"])
    rec = replay["records"][0]
    assert rec["distance_m"] > 10.0 and 0.0 <= rec["entry_axis_misalignment_rad"] <= np.pi
    assert rec["remaining_time_s"] == pytest.approx(env.environment_config.max_time_s)
    cf = replay["counterfactual"]
    assert [c["k"] for c in cf] == [0, 1]
    for c in cf:
        assert c["handoff"]["decisions"] == 2 and c["continue"]["decisions"] == 2
        assert c["eps_H"] == pytest.approx(c["q_handoff"] - c["handoff"]["return"])
    summary = summarize({"m": {263004: dict(replay, fidelity_ok=True)}})["models"]["m"]
    assert summary["handoff"]["never"] == 1 and summary["counterfactual"]["states"] == 2
    assert summary["continuation_value_semantics"]["learned_decisions"] == 3
    env.close()
    env_b.close()


def test_continuation_values_are_deterministic_and_leave_the_rng_alone() -> None:
    env = _env()
    model = _model(env)
    obs = np.zeros(42, dtype=np.float32)
    th.manual_seed(5)
    before = th.rand(1)
    th.manual_seed(5)
    a = continuation_values(model, obs, 11)
    after = th.rand(1)
    assert th.equal(before, after)
    assert a == continuation_values(model, obs, 11)
    assert a["v_det_online"] == pytest.approx(model.stopping_values(obs)["q_continue"])
    env.close()


def _row(row, clean, k=None, sha=None, t=100.0):
    return {"row": row, "clean_completion": clean, "zero_violation": True, "survival_s": t,
            "equivalent_delta_v_m_s": t / 100, "handoff_k": k, "model_sha256": sha,
            "code_commit": "c", "code_dirty": False, "max_decisions": None}


def test_tables_pair_rows_and_check_fidelity() -> None:
    seeds = list(range(6))
    pure = {s: _row("pure", s < 3, t=90.0) for s in seeds}
    nominal = {s: _row("nominal", s < 5, t=180.0) for s in seeds}
    learned = {"m": {s: _row("learned", s < 2, sha="x", t=200.0) for s in seeds}}
    stopping = {"m": {s: _row("stopping", s < 4, k=3, sha="x", t=120.0) for s in seeds}}
    out = build(pure, nominal, learned, stopping, seeds)
    assert out["problems"] == []
    m = out["models"]["m"]
    assert m["stopping_vs_pure"]["a_only"] == [3] and m["stopping_vs_nominal"]["b_only"] == [4]
    assert m["stopping_vs_learned"]["a_only"] == [2, 3] and m["stopping"]["handoff"]["mid"] == 6
    assert m["stopping_vs_nominal"]["median_time_diff_s"] == -60.0
    stopping["m"][0]["code_commit"] = "other"
    assert build(pure, nominal, learned, stopping, seeds)["problems"]
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(0, 6) == pytest.approx(2 / 64)
