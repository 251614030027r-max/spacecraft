"""Learned stopping option: value algebra, configs, wrapper, updates, rows, readout."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.logger import configure

from experiments.v3_handoff_scan import ImpulseMeter
from experiments.v3_stage_c import FEATURE_BLOCKS
from experiments.v3_stopping import dev_checks, final_verdict, run_stopping_episode
from train.hybrid_configs import SAC_MPC_HYBRID, hybrid_model_kwargs
from train.mainline import mainline_v3e_configs
from train.stopping import (
    HANDOFF_FEATURE_BLOCKS,
    STOPPING,
    STOPPING_HEAD_20261002,
    STOPPING_SAC,
    HandoffOptionEnv,
    StoppingConfig,
    StoppingSAC,
    StopNetworks,
    discounted_returns_to_go,
    handoff_feature_index,
    soft_stopping_value,
    stopping_configs,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_returns_to_go_and_the_soft_maximum() -> None:
    np.testing.assert_allclose(discounted_returns_to_go([1.0, 2.0, 3.0], 0.5), [2.75, 3.5, 3.0])
    q_h, v_c, alpha = th.tensor([2.0, -1.0]), th.tensor([1.0, 3.0]), 0.3
    # At the stopping head's optimum the value is the soft maximum of Q_H and V_C.
    optimum = (q_h - v_c) / alpha
    value = soft_stopping_value(optimum, q_h, v_c, alpha)
    expected = alpha * th.logsumexp(th.stack([q_h / alpha, v_c / alpha]), dim=0)
    th.testing.assert_close(value, expected)
    assert value[0] >= max(2.0, 1.0) and value[1] >= 3.0
    big = th.tensor([60.0, -60.0])
    th.testing.assert_close(soft_stopping_value(big, q_h, v_c, 0.0), th.tensor([2.0, 3.0]))


def test_configs_change_only_the_discount() -> None:
    environment, hybrid = stopping_configs()
    environment_v3e, hybrid_v3e = mainline_v3e_configs()
    assert asdict(environment) == asdict(environment_v3e)
    assert replace(hybrid, decision_discount_factor=hybrid_v3e.decision_discount_factor) == hybrid_v3e
    assert hybrid.decision_discount_factor == STOPPING.gamma == STOPPING_SAC.gamma == 0.999
    assert replace(STOPPING_SAC, gamma=SAC_MPC_HYBRID.gamma) == SAC_MPC_HYBRID
    assert HANDOFF_FEATURE_BLOCKS == FEATURE_BLOCKS
    with pytest.raises(ValueError):
        StoppingConfig(behaviour_handoff_min=0.5, behaviour_handoff_max=0.1)


def test_initial_stopping_head_is_the_same_small_probability_everywhere() -> None:
    nets = StopNetworks(42, list(range(31)), [32, 32], STOPPING.initial_handoff_logit)
    beta = th.sigmoid(nets.logit(th.randn(64, 42) * 5.0))
    th.testing.assert_close(beta, th.full_like(beta, 1.0 / (1.0 + np.exp(5.0))))
    q1, q2 = nets.q_handoff(th.randn(4, 42))
    assert q1.shape == q2.shape == (4, 1)


class _FakeHybrid(gym.Env):
    """Learned decisions reward 1, baseline decisions reward 10; 3 baseline decisions end it."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.calls: list[str] = []

    def reset(self, *, seed=None, options=None):
        self.k, self.base, self.calls = 0, 0, []
        return np.zeros(3, dtype=np.float32), {}

    def step_with_branch(self, action, *, branch):
        self.calls.append(branch)
        self.k += 1
        if branch == "baseline":
            self.base += 1
        done = self.base == 3 or self.k == 50
        return np.full(3, self.k, dtype=np.float32), (10.0 if branch == "baseline" else 1.0), done, False, {}


def test_option_env_hands_off_for_good_and_labels_the_suffix() -> None:
    config = StoppingConfig(gamma=0.5, behaviour_handoff_min=1.0, behaviour_handoff_max=1.0)
    env = HandoffOptionEnv(_FakeHybrid(), config, seed=0)
    env.handoff_probability = lambda obs: 0.0  # the clip floor of 1 forces a handoff anyway
    env.reset()
    obs, reward, terminated, _, info = env.step(np.zeros(2))
    assert env.env.calls == ["baseline"] * 3 and terminated and reward == 30.0
    assert info["handoff"] and info["handoff_decision"] == 0 and info["suffix_decisions"] == 3
    np.testing.assert_allclose(info["handoff_returns"], [10 + 5 + 2.5, 10 + 5, 10])
    assert info["handoff_states"].shape == (3, 3) and info["handoff_states"][0].sum() == 0.0

    never = HandoffOptionEnv(_FakeHybrid(), replace(config, behaviour_handoff_min=0.0, behaviour_handoff_max=0.0), seed=0)
    never.handoff_probability = lambda obs: 1.0  # the cap of 0 forbids it
    never.reset()
    for _ in range(4):
        _, reward, _, _, info = never.step(np.zeros(2))
        assert reward == 1.0 and info["handoff"] is False
    assert never.env.calls == ["learned"] * 4


def test_option_env_samples_the_clipped_probability() -> None:
    config = StoppingConfig(behaviour_handoff_min=0.0, behaviour_handoff_max=0.3)
    env = HandoffOptionEnv(_FakeHybrid(), config, seed=1)
    env.handoff_probability = lambda obs: 0.9
    first = []
    for _ in range(2000):
        env.reset()
        k = 0
        while True:
            _, _, done, _, info = env.step(np.zeros(2))
            if info["handoff"]:
                first.append(k)
                break
            k += 1
    # geometric with p = 0.3: P(k = 0) = 0.3
    assert abs(np.mean(np.asarray(first) == 0) - 0.3) < 0.04


def _model(env, stopping: StoppingConfig = STOPPING, **overrides) -> StoppingSAC:
    kwargs = hybrid_model_kwargs(STOPPING_SAC)
    kwargs["policy_kwargs"]["net_arch"] = [16, 16]
    kwargs.update(learning_starts=0, buffer_size=1000, **overrides)
    return StoppingSAC("MlpPolicy", env, stopping=asdict(stopping), seed=0, device="cpu",
                       handoff_feature_index=handoff_feature_index(env.policy_observation_slices()), **kwargs)


def _env():
    environment, hybrid = stopping_configs()
    from env.hybrid_env import PrecaptureHybridEnv

    return PrecaptureHybridEnv(environment, hybrid)


def test_handoff_step_is_not_a_continue_transition_and_earns_its_updates(tmp_path: Path) -> None:
    env = _env()
    model = _model(env, learning_rate=1e-3)
    obs = np.zeros((1, 42), dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(8):
        model.replay_buffer.add(rng.normal(size=(1, 42)).astype(np.float32), rng.normal(size=(1, 42)).astype(np.float32),
                                rng.uniform(-1, 1, size=(1, 2)).astype(np.float32), np.zeros(1), np.zeros(1), [{}])
    stored = model.replay_buffer.size()
    model.num_timesteps = 10
    info = {"handoff": True, "suffix_decisions": 3, "handoff_states": rng.normal(size=(3, 42)).astype(np.float32),
            "handoff_returns": np.array([40.0, 40.0, 40.0])}
    model._store_transition(model.replay_buffer, np.zeros((1, 2)), obs, np.zeros(1), np.array([True]), [info])
    assert model.replay_buffer.size() == stored
    # The suffix labels Q_H but adds no outer decision and no update.
    assert model.handoff_buffer_size == 3 and model.num_timesteps == 10
    assert model.suffix_decisions_total == 3 and model.continue_transitions == 0
    model.set_logger(configure(None, [""]))
    states = th.as_tensor(model.replay_buffer.observations[:8, 0])
    with th.no_grad():
        before = float(th.sigmoid(model.stop_logit(states)).mean())
    model.train(1, batch_size=8)
    assert model._n_updates == 1
    # Handoff is worth +40 everywhere and the untrained continuation is ~0:
    # the stopping head must move toward handing off.
    for _ in range(40):
        model.train(1, batch_size=8)
    with th.no_grad():
        after = float(th.sigmoid(model.stop_logit(states)).mean())
    assert after > before

    model.save(tmp_path / "m")
    loaded = StoppingSAC.load(tmp_path / "m.zip", device="cpu")
    probe = rng.normal(size=42).astype(np.float32)
    assert loaded.handoff_probability(probe) == pytest.approx(model.handoff_probability(probe))
    assert loaded.stopping_values(probe) == pytest.approx(model.stopping_values(probe))
    env.close()


def test_rows_reduce_to_pure_and_learned() -> None:
    env = _env()
    model = _model(env, STOPPING_HEAD_20261002)
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    run = lambda row: run_stopping_episode(env, model, 263004, row, meter, max_decisions=2)
    pure, learned = run("pure"), run("learned")
    with th.no_grad():
        model.stop.stop_head[-1].bias.fill_(50.0)
    always = run("stopping")
    with th.no_grad():
        model.stop.stop_head[-1].bias.fill_(-50.0)
    never = run("stopping")
    env.close()
    for key in ("survival_s", "equivalent_delta_v_m_s", "completed", "failure"):
        assert always[key] == pure[key]
        assert never[key] == learned[key]
    assert always["handoff_k"] == 0 and never["handoff_k"] is None
    assert never["learned_decisions"] == 2 and len(never["trace"]["beta"]) == 2


def _episode(k, learned, gap):
    return {"handoff_k": k, "learned_decisions": learned, "clean_completion": True,
            "trace": {"q_handoff": list(gap), "q_continue": [0.0] * len(gap),
                      "beta": [1.0 if g >= 0 else 0.0 for g in gap]}}


def test_dev_checks_catch_the_degenerate_corners() -> None:
    monitor = [{"learned_decisions": "40", "handoff": "True"}] * 8
    healthy = {s: _episode([None, 3, 7, 12, 40][s % 5], 40, [0.0, -2.0, 1.0]) for s in range(48)}
    result = dev_checks(monitor, healthy)
    assert result["all_pass"]
    collapsed = {s: _episode(0, 0, []) for s in range(48)}
    result = dev_checks(monitor, collapsed)
    assert not result["D1_not_collapsed"]["pass"] and not result["D2_learned_reaches_mid_late"]["pass"]
    flat = {s: _episode(5, 40, [0.1, 0.1]) for s in range(48)}
    assert not dev_checks(monitor, flat)["D3_state_dependent_stopping_value"]["pass"]


def test_final_verdict_rules() -> None:
    ok, fail = {"passes": True}, {"passes": False}
    assert final_verdict({"a": ok, "b": ok, "c": fail}, True) == "METHOD_HOLDS"
    assert final_verdict({"a": ok, "b": fail, "c": fail}, True) == "METHOD_DOES_NOT_HOLD"
    assert final_verdict({"a": ok, "b": ok}, False) == "FIDELITY_FAIL"


def test_train_and_evaluate_cli(tmp_path: Path) -> None:
    subprocess.run([sys.executable, "-B", "-m", "train.train_stopping", "--steps", "6", "--seed", "3",
                    "--run-name", "r", "--log-root", str(tmp_path), "--learning-starts", "2", "--net-arch", "16,16",
                    "--regime", "w3.00_r18"],
                   cwd=REPOSITORY, check=True, capture_output=True)
    run_dir = tmp_path / "r"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["actual_outer_decisions"] >= 6
    assert manifest["hybrid"]["decision_discount_factor"] == 0.999
    assert manifest["regime"]["name"] == "w3.00_r18" and manifest["stopping"]["stop_rule"] == "value"
    assert (run_dir / "train.monitor.csv").exists()
    out = tmp_path / "eval"
    subprocess.run([sys.executable, "-B", "-m", "experiments.v3_stopping", "evaluate", "--run-dir", str(run_dir),
                    "--row", "stopping", "--seeds", "263004", "--max-decisions", "2", "--output-dir", str(out)],
                   cwd=REPOSITORY, check=True, capture_output=True)
    episode = json.loads((out / "seed_263004.json").read_text())
    assert episode["row"] == "stopping" and episode["decisions"] == 2 and episode["model_sha256"]


def test_coordination_gain_compares_against_the_same_policy_alone() -> None:
    from experiments.v3_stopping import coordination_gain

    row = lambda clean, t, k=None: {"clean_completion": clean, "survival_s": t, "equivalent_delta_v_m_s": t / 100, "handoff_k": k}
    learned = {0: row(True, 200), 1: row(False, 300), 2: row(True, 180)}
    stopping = {0: row(True, 150, 20), 1: row(True, 140, 5), 2: row(False, 90, 3)}
    gain = coordination_gain(learned, stopping)
    assert gain["clean_completions"] == {"learned_only": 2, "stopping": 2}
    assert gain["rescued_from_learned_only"] == [1] and gain["destroyed_from_learned_only"] == [2]
    assert gain["shared_clean"] == 1 and gain["median_time_change_s"] == -50


def test_value_row_follows_the_value_comparison() -> None:
    env = _env()
    model = _model(env)
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    run = lambda row: run_stopping_episode(env, model, 263004, row, meter, max_decisions=2)
    pure, learned = run("pure"), run("learned")
    with th.no_grad():
        model.stop.stop_head[-1].bias.fill_(-50.0)  # the head says continue; the value row ignores it
        for net in (model.stop.q_handoff_1, model.stop.q_handoff_2):
            net[-1].bias.fill_(1.0e4)
    high = run("value")
    with th.no_grad():
        for net in (model.stop.q_handoff_1, model.stop.q_handoff_2):
            net[-1].bias.fill_(-1.0e4)
    low = run("value")
    env.close()
    for key in ("survival_s", "equivalent_delta_v_m_s", "completed"):
        assert high[key] == pure[key] and low[key] == learned[key]
    assert high["handoff_k"] == 0 and low["handoff_k"] is None


def test_value_gate_verdict() -> None:
    from experiments.v3_stopping import gate_verdict

    row = lambda clean, k=None: {"clean_completion": clean, "zero_violation": True, "survival_s": 100.0,
                                 "equivalent_delta_v_m_s": 1.0, "handoff_k": k}
    pure = {s: row(s < 6) for s in range(8)}
    better = {s: row(s != 0, k=3) for s in range(8)}
    v = gate_verdict(pure, better)
    assert v["passes"] and v["rescued"] == [6, 7] and v["destroyed"] == [0]
    worse = {s: row(s not in (0, 1, 2), k=0) for s in range(8)}
    assert not gate_verdict(pure, worse)["passes"]


def test_value_rule_stopping_is_the_value_comparison() -> None:
    assert STOPPING.stop_rule == "value" and StoppingConfig().stop_rule == "head"
    env = _env()
    model = _model(env)  # the final method: value-based stopping
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    run = lambda row: run_stopping_episode(env, model, 263004, row, meter, max_decisions=2)
    pure, learned = run("pure"), run("learned")
    rng = np.random.default_rng(3)
    for _ in range(20):
        values = model.stopping_values(rng.normal(size=42).astype(np.float32))
        assert (values["beta"] >= 0.5) == (values["q_handoff"] >= values["q_continue"])
    for bias, expected, k in ((1.0e4, pure, 0), (-1.0e4, learned, None)):
        with th.no_grad():
            for net in (model.stop.q_handoff_1, model.stop.q_handoff_2):
                net[-1].bias.fill_(bias)
        for row in ("stopping", "value"):
            got = run(row)
            assert got["handoff_k"] == k
            for key in ("survival_s", "equivalent_delta_v_m_s", "completed"):
                assert got[key] == expected[key]
    env.close()


def test_dev_checks_require_stopping_to_match_the_value_rule() -> None:
    monitor = [{"learned_decisions": "40", "handoff": "True"}] * 8
    good = {s: _episode([None, 3, 7, 12, 40][s % 5], 40, [0.0, -2.0, 1.0]) for s in range(48)}
    for e in good.values():
        e["trace"]["beta"] = [1.0 if g >= 0 else 0.0 for g in e["trace"]["q_handoff"]]
    assert dev_checks(monitor, good)["D4_stopping_matches_value_rule"]["pass"]
    bad = {s: dict(e, trace=dict(e["trace"], beta=[1.0] * len(e["trace"]["beta"]))) for s, e in good.items()}
    assert not dev_checks(monitor, bad)["D4_stopping_matches_value_rule"]["pass"]
