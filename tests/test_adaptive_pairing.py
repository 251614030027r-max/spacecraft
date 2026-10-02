from __future__ import annotations

import pytest

from eval.adaptive_pairing import compare_payloads


def _record(seed: int, completed: bool, action: list[float]) -> dict:
    return {
        "seed": seed,
        "completed": completed,
        "truth_geometry_zero_violation_completed": completed,
        "survival_s": 10.0 + seed,
        "force_impulse_n_s": 20.0 + seed,
        "equivalent_delta_v_m_s": 0.2 + seed,
        "minimum_truth_normalized_margin": 0.5,
        "gate_decisions": 2,
        "gate_deviations": int(any(action)),
        "policy_actions_raw": [action, action],
        "applied_actions": [action, action],
        "qp_infeasible_steps_total": 0,
        "zero_fallback_steps_total": 0,
        "max_consecutive_zero_wrench_steps": 1,
        "maximum_successful_slack": 0.01,
    }


def test_pairing_separates_retained_rescued_destroyed_and_failed() -> None:
    baseline = {
        "records": [
            _record(1, True, [0.0, 0.0]),
            _record(2, False, [0.0, 0.0]),
            _record(3, True, [0.0, 0.0]),
            _record(4, False, [0.0, 0.0]),
        ]
    }
    candidate = {
        "records": [
            _record(1, True, [0.3, 0.4]),
            _record(2, True, [0.0, 0.0]),
            _record(3, False, [0.1, 0.0]),
            _record(4, False, [0.0, 0.0]),
        ]
    }

    result = compare_payloads(baseline, candidate)

    assert result["quadrants"]["retained"]["seeds"] == [1]
    assert result["quadrants"]["rescued"]["seeds"] == [2]
    assert result["quadrants"]["destroyed"]["seeds"] == [3]
    assert result["quadrants"]["both_failed"]["seeds"] == [4]
    assert result["gate"]["deviations"] == 2
    assert result["quadrants"]["retained"]["raw_action_l2"]["mean"] == pytest.approx(0.5)


def test_pairing_rejects_different_seed_sets() -> None:
    baseline = {"records": [_record(1, True, [0.0, 0.0])]}
    candidate = {"records": [_record(2, True, [0.0, 0.0])]}

    with pytest.raises(ValueError, match="seed sets differ"):
        compare_payloads(baseline, candidate)
