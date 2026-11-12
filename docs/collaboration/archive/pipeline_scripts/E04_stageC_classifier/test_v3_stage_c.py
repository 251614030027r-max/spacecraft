"""Stage C: labels, folds, tau rule, first trigger, closed-loop rows, verdict."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import SAC

from experiments.v3_handoff_scan import ImpulseMeter
from experiments.v3_stage_c import (
    FEATURE_BLOCKS,
    HandoffClassifier,
    build_dataset,
    choose_tau,
    feature_index,
    first_trigger,
    fold_of,
    labels_from_scans,
    model_verdict,
    overall_verdict,
    run_closed_loop,
    trigger_summary,
)
from train.mainline import MAINLINE_V3E_EVALUATION_FLAGS, mainline_v3e_env

REPOSITORY = Path(__file__).resolve().parents[1]


def test_features_are_the_policy_independent_blocks() -> None:
    env = mainline_v3e_env()
    slices = env.policy_observation_slices()
    env.close()
    index = feature_index(slices)
    assert len(index) == 31
    used = {name for name, block in slices.items() if set(range(block.start, block.stop)) & set(index)}
    assert used == set(FEATURE_BLOCKS)
    assert not used & {"task_state", "applied_direction", "execution_feedback", "staging_direction"}


def test_labels_merge_b1_and_b2_and_folds_group_by_seed() -> None:
    b1 = {"scanned": True, "handoffs": [{"k": 0, "clean_completion": False}, {"k": 1, "clean_completion": True}]}
    b2 = {"scanned": True, "handoffs": [{"k": 0, "clean_completion": True}, {"k": 10, "clean_completion": False}]}
    assert labels_from_scans(b1, None) == {0: False, 1: True}
    assert labels_from_scans({"scanned": False, "handoffs": []}, b2) == {0: True, 10: False}
    assert fold_of(262000) == fold_of(262006) != fold_of(262001)


def test_dataset_weights_each_trajectory_to_one() -> None:
    obs = np.arange(4 * 42, dtype=np.float32).reshape(4, 42)
    records = {"m": {"obs": obs, "seed": np.array([1, 1, 1, 2]), "k": np.array([0, 1, 2, 0])}}
    b1 = {"m": {1: {"scanned": True, "handoffs": [{"k": k, "clean_completion": k == 2} for k in range(3)]},
                2: {"scanned": False, "handoffs": []}}}
    b2 = {"m": {2: {"scanned": True, "handoffs": [{"k": 0, "clean_completion": True}]}}}
    data = build_dataset(records, b1, b2, np.arange(31))
    assert data["x"].shape == (4, 31)
    assert data["w"][data["seed"] == 1].sum() == pytest.approx(1.0)
    assert data["w"][data["seed"] == 2].sum() == pytest.approx(1.0)
    assert list(data["source"]) == ["B1", "B1", "B1", "B2"]


def test_tau_is_the_smallest_grid_value_reaching_the_precision() -> None:
    p = np.array([0.99, 0.98, 0.96, 0.9, 0.6, 0.4])
    y = np.array([1, 1, 1, 0, 1, 0], dtype=float)
    tau, curve = choose_tau(p, y, np.ones(6))
    assert tau == 0.95  # >= 0.90 still admits the 0.9 negative
    assert choose_tau(np.zeros(3), np.zeros(3), np.ones(3))[0] is None


def test_first_trigger_and_summary() -> None:
    hit = first_trigger(np.array([0.1, 0.9, 0.95]), np.array([0, 1, 2]), {1: True}, 0.8)
    assert hit == {"k": 1, "label": True}
    assert first_trigger(np.array([0.1]), np.array([0]), {}, 0.8) == {"k": None, "label": None}
    off = first_trigger(np.array([0.1, 0.9]), np.array([0, 3]), {0: True, 10: False}, 0.8)
    assert off == {"k": 3, "label": None}
    summary = trigger_summary([
        {"source": "B1", "k": 4, "label": True}, {"source": "B1", "k": None, "label": None},
        {"source": "B2", "k": 3, "label": None}, {"source": "B2", "k": 10, "label": False},
    ])
    assert summary["B1"]["triggered_clean"] == 1 and summary["B1"]["never_triggered"] == 1
    assert summary["B2"]["triggered_off_grid"] == 1 and summary["B2"]["triggered_fails"] == 1


def test_classifier_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 31)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    w = np.ones(60, dtype=np.float32)
    a = HandoffClassifier.fit(x, y, w, np.arange(31), epochs=30)
    b = HandoffClassifier.fit(x, y, w, np.arange(31), epochs=30)
    np.testing.assert_array_equal(a.predict_features(x), b.predict_features(x))
    a.tau = 0.8
    a.save(tmp_path / "c.pt", {})
    c = HandoffClassifier.load(tmp_path / "c.pt")
    assert c.tau == 0.8
    np.testing.assert_array_equal(a.predict_features(x), c.predict_features(x))
    observation = np.zeros(42)
    observation[:31] = x[0]
    assert c.p_handoff(observation) == pytest.approx(float(c.predict_features(x[:1])[0]))


class _Fixed:
    def __init__(self, tau: float) -> None:
        self.tau = tau

    def p_handoff(self, observation) -> float:
        return 0.5


def test_hybrid_rows_reduce_to_pure_and_learned() -> None:
    env = mainline_v3e_env()
    policy = SAC("MlpPolicy", env, seed=0, device="cpu", policy_kwargs={"net_arch": [16]})
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    run = lambda row, clf=None: run_closed_loop(env, policy, 263004, row, clf, meter, max_decisions=2)
    pure, learned = run("pure"), run("learned")
    always, never = run("hybrid", _Fixed(0.0)), run("hybrid", _Fixed(0.9))
    env.close()
    for key in ("survival_s", "equivalent_delta_v_m_s", "completed", "failure"):
        assert always[key] == pure[key]
        assert never[key] == learned[key]
    assert always["handoff_k"] == 0 and never["handoff_k"] is None
    assert never["learned_decisions"] == 2 and always["learned_decisions"] == 0


def _row(clean, violated=False, t=100.0, dv=2.0, k=None):
    return {"clean_completion": clean, "zero_violation": not violated, "survival_s": t,
            "equivalent_delta_v_m_s": dv, "handoff_k": k, "learned_decisions": 1, "decisions": 2}


def test_model_verdict_counts_and_overall_rules() -> None:
    seeds = range(8)
    pure = {s: _row(s < 5) for s in seeds}
    learned = {s: _row(s != 0, t=190.0, dv=3.0) for s in seeds}
    hybrid = {s: _row(s != 0, t=120.0, dv=2.2, k=3) for s in seeds}
    v = model_verdict(pure, learned, hybrid)
    assert v["clean_completions"] == {"pure": 5, "learned": 7, "hybrid": 7}
    assert v["destroyed"] == [0] and v["rescued"] == [5, 6, 7]
    assert v["passes"] and v["shared_clean_seeds"] == 4
    assert v["efficiency_acceptable"] is None  # fewer than 5 shared seeds
    hybrid_violating = dict(hybrid)
    hybrid_violating[7] = _row(True, violated=True)
    assert not model_verdict(pure, learned, hybrid_violating)["passes"]

    ok = {"passes": True, "efficiency_acceptable": True}
    slow = {"passes": True, "efficiency_acceptable": False}
    fail = {"passes": False, "efficiency_acceptable": None}
    assert overall_verdict({"a": ok, "b": ok, "c": fail}, True) == "A_GO_TO_E"
    assert overall_verdict({"a": ok, "b": slow, "c": fail}, True) == "B_GO_TO_D"
    assert overall_verdict({"a": ok, "b": fail, "c": fail}, True) == "C_STOP"
    assert overall_verdict({"a": ok, "b": ok}, False) == "FIDELITY_FAIL"


def test_efficiency_rule_halves_the_learned_excess() -> None:
    seeds = range(6)
    pure = {s: _row(True, t=100.0, dv=2.0) for s in seeds}
    learned = {s: _row(True, t=180.0, dv=3.0) for s in seeds}
    good = {s: _row(True, t=130.0, dv=2.4, k=2) for s in seeds}
    bad = {s: _row(True, t=150.0, dv=2.4, k=2) for s in seeds}
    assert model_verdict(pure, learned, good)["efficiency_acceptable"] is True
    assert model_verdict(pure, learned, bad)["efficiency_acceptable"] is False


def test_record_cli_checks_against_b1(tmp_path: Path) -> None:
    env = mainline_v3e_env()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    SAC("MlpPolicy", env, seed=0, device="cpu", policy_kwargs={"net_arch": [16]}).save(run_dir / "final_model.zip")
    manifest = {"run_name": "t", "code_commit": None, "status": "completed",
                "waypoint_parametrization": "task_state_v3", "monotone_commit": True,
                "baseline_anchored_residual": False, "evaluation_flags": MAINLINE_V3E_EVALUATION_FLAGS,
                "hybrid": asdict(env.hybrid_config), "observation_dimension": int(env.observation_space.shape[0])}
    env.close()
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    b1_dir = tmp_path / "b1"
    subprocess.run([sys.executable, "-B", "-m", "experiments.v3_handoff_scan", "--run-dir", str(run_dir),
                    "--seeds", "263005", "--scan", "failures", "--verify-prefix", "", "--max-decisions", "2",
                    "--output-dir", str(b1_dir)], cwd=REPOSITORY, check=True, capture_output=True)
    record = tmp_path / "c1.npz"
    base = [sys.executable, "-B", "-m", "experiments.v3_stage_c", "record", "--run-dir", str(run_dir),
            "--seeds", "263005", "--b1-dir", str(b1_dir), "--max-decisions", "2"]
    subprocess.run(base + ["--output", str(record)], cwd=REPOSITORY, check=True, capture_output=True)
    data = np.load(record)
    assert data["obs"].shape == (2, 42) and list(data["k"]) == [0, 1]
    meta = json.loads(record.with_suffix(".json").read_text())
    assert meta["rows"] == 2 and "core" in meta["observation_slices"]

    scan_path = b1_dir / "seed_263005.json"
    scan = json.loads(scan_path.read_text())
    scan["learned_full"]["task_rewards"][0] += 1e-12
    scan_path.write_text(json.dumps(scan))
    broken = subprocess.run(base + ["--output", str(tmp_path / "c1b.npz")], cwd=REPOSITORY,
                            capture_output=True, text=True)
    assert broken.returncode != 0 and "differs from the B1" in broken.stderr
