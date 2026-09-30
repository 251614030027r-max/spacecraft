"""Stage-B readout: window definitions, the gate, and the fidelity checks."""

from __future__ import annotations

import pytest

from experiments.v3_handoff_readout import (
    b2_summary,
    failure_trajectory,
    fidelity_checks,
    model_gate,
    runs_of,
    success_trajectory,
    verdict,
)


def _scan(seed: int, clean: list[bool], *, learned_completed=False, stride=1, survival=300.0) -> dict:
    ks = list(range(0, len(clean) * stride, stride))
    return {
        "seed": seed,
        "stride": stride,
        "scan": "failures",
        "max_decisions": None,
        "scanned": True,
        "verified_prefix_ks": [0],
        "code_commit": "c",
        "code_dirty": False,
        "model_sha256": "m",
        "learned_full": {
            "decisions": len(clean) * stride,
            "failure": [] if learned_completed else ["time_failure"],
            "completed": learned_completed,
            "zero_violation": True,
            "survival_s": survival,
            "equivalent_delta_v_m_s": 4.0,
        },
        "handoffs": [
            {"k": k, "clean_completion": c, "completed": c, "decisions": 50, "failure": [],
             "survival_s": 100.0 + k, "equivalent_delta_v_m_s": 3.0}
            for k, c in zip(ks, clean)
        ],
    }


def test_runs_are_maximal_consecutive_blocks() -> None:
    assert runs_of([]) == []
    assert runs_of([3, 1, 2, 7, 9, 10]) == [(1, 3), (7, 7), (9, 10)]
    assert runs_of([0, 5, 10, 20], stride=5) == [(0, 10), (20, 20)]


def test_window_width_uses_the_longest_run_not_earliest_to_latest() -> None:
    # K_rescue = {1, 4, 5, 6}: earliest 1, latest 6, but the widest run is 3.
    t = failure_trajectory(_scan(1, [False, True, False, False, True, True, True, False]))
    assert (t["earliest_k"], t["latest_k"]) == (1, 6)
    assert t["w_max_decisions"] == 3 and t["w_max_s"] == 6.0
    assert [r["width_decisions"] for r in t["runs"]] == [1, 3]
    assert t["non_degenerate"] and t["interior_window"]
    assert t["new_capability"]  # Pure from reset (k=0) fails, a later handoff completes


def test_needles_and_k0_only_windows() -> None:
    needle = failure_trajectory(_scan(2, [False, False, True, False]))
    assert needle["rescuable"] and not needle["non_degenerate"]
    destroy = failure_trajectory(_scan(3, [True, False, False]))  # Pure completes, only at k=0
    assert destroy["pure_from_reset_clean"] and not destroy["new_capability"]
    assert not destroy["non_degenerate"] and not destroy["interior_window"]


def test_gate_needs_two_nondegenerate_and_not_mainly_needles() -> None:
    wide = failure_trajectory(_scan(1, [False, True, True]))
    needle = failure_trajectory(_scan(2, [False, True, False]))
    none = failure_trajectory(_scan(3, [False, False, False]))
    assert not model_gate([wide, none])["passes"]  # only one non-degenerate
    assert model_gate([wide, wide, needle, none])["passes"]  # 2 wide >= 1 needle
    assert not model_gate([wide, wide, needle, needle, needle])["passes"]  # mainly needles
    gate = model_gate([wide, wide, needle])
    assert gate["critical_states_raw"] == 5 and gate["independent_windows"] == 3


def test_verdict() -> None:
    passing, failing = {"passes": True}, {"passes": False}
    assert verdict(False, {"a": passing, "b": passing}) == "FIDELITY_FAIL"
    assert verdict(True, {"a": passing, "b": passing, "c": failing}) == "PROCEED"
    assert verdict(True, {"a": passing, "b": failing, "c": failing}) == "STOP"


def test_gate_scan_must_be_complete_and_at_every_decision() -> None:
    with pytest.raises(ValueError):
        failure_trajectory(_scan(1, [True, True], stride=2))
    broken = _scan(1, [True, True, True])
    broken["handoffs"].pop(1)
    with pytest.raises(ValueError):
        failure_trajectory(broken)


def test_success_accounting() -> None:
    scan = _scan(4, [True, False, True, True], learned_completed=True, stride=5, survival=190.0)
    t = success_trajectory(scan)
    assert t["destroy_count"] == 1 and t["destroy_runs"] == 1
    assert t["earlier_clean_handoff_is_faster"] and t["best_time_saving_s"] == 190.0 - 110.0
    assert t["delta_v_change_at_best_m_s"] == -1.0
    summary = b2_summary([t])
    assert summary["with_destroying_handoff"] == 1 and summary["with_faster_clean_handoff"] == 1


def test_fidelity_catches_mismatches_and_missing_seeds() -> None:
    scan = _scan(262001, [False, True])
    scans = {"m": {262001: scan}}
    learned_ref = {"m": {262001: {"completed": False, "decisions": 2, "survival_s": 300.0}}}
    pure_ref = {262001: {"completed": False, "decisions": 50, "survival_s": 100.0}}
    assert fidelity_checks(scans, learned_ref, pure_ref, {}, [262001]) == []
    assert fidelity_checks(scans, learned_ref, pure_ref, {}, [262001, 262002]) == [
        "m: missing seeds [262002]"
    ]
    pure_ref[262001]["survival_s"] = 99.9
    assert any("Pure MPC" in p for p in fidelity_checks(scans, learned_ref, pure_ref, {}, [262001]))
    scan["code_dirty"] = True
    assert any("dirty" in p for p in fidelity_checks(scans, learned_ref, {}, {}, [262001]))


def test_b2_fidelity_checks_completeness_commit_stride_and_b1_agreement() -> None:
    from experiments.v3_handoff_readout import B2_STRIDE, b2_fidelity_checks

    def learned(completed):
        return {"completed": completed, "decisions": 40, "survival_s": 80.0,
                "task_rewards": [1.0, 2.0], "zero_violation": True, "failure": []}

    b1 = {"m": {
        1: {"scanned": False, "model_sha256": "x", "learned_full": learned(True)},
        2: {"scanned": True, "model_sha256": "x", "learned_full": learned(False)},
    }}

    def b2_scan(scanned, **over):
        scan = {"scan": "successes", "stride": B2_STRIDE, "max_decisions": None, "code_commit": "c",
                "code_dirty": False, "model_sha256": "x", "scanned": scanned,
                "verified_prefix_ks": [20] if scanned else [], "learned_full": learned(scanned)}
        scan.update(over)
        return scan

    good = {"m": {1: b2_scan(True), 2: b2_scan(False)}}
    assert b2_fidelity_checks(good, b1, [1, 2]) == []
    assert any("missing seeds [3]" in p for p in b2_fidelity_checks(good, b1, [1, 2, 3]))
    bad = {"m": {1: b2_scan(True, stride=5), 2: b2_scan(False)}}
    assert any("stride" in p for p in b2_fidelity_checks(bad, b1, [1, 2]))
    bad = {"m": {1: b2_scan(True, verified_prefix_ks=[]), 2: b2_scan(False)}}
    assert any("prefix verification" in p for p in b2_fidelity_checks(bad, b1, [1, 2]))
    bad = {"m": {1: b2_scan(True, code_commit="d"), 2: b2_scan(False)}}
    assert any("code commits" in p for p in b2_fidelity_checks(bad, b1, [1, 2]))
    drifted = b2_scan(True)
    drifted["learned_full"] = dict(drifted["learned_full"], survival_s=80.1)
    assert any("survival_s differs from B1" in p
               for p in b2_fidelity_checks({"m": {1: drifted, 2: b2_scan(False)}}, b1, [1, 2]))
