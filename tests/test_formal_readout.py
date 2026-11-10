from math import pi

import pytest

from env.t2_stage import window_half_width
from experiments.formal_readout import PRESCREEN_POLICIES, paired, paired_bootstrap, prescreen, primary


def row(event):
    return {"event": event, "return": 1 if event == "success" else -1, "time_s": 10, "delta_v": 1.0}


def test_paired_bootstrap_and_primary():
    candidate = {(s, o): row("success") for s in (20, 21, 22) for o in range(5)}
    baseline = {key: row("timeout") for key in candidate}
    comparison = paired(candidate, baseline)
    assert comparison["combined"]["capture"]["interval"] == [1, 1]
    assert len(comparison["rescued"]) == 15 and comparison["lost"] == []
    assert primary(comparison) == "significant_increase"
    assert paired_bootstrap([[0], [1, 1]])["interval"] == pytest.approx([2 / 3, 2 / 3])
    assert primary(paired(baseline, candidate)) == "not_established"
    with pytest.raises(ValueError, match="Paired key mismatch"):
        paired(candidate, {})


def test_window_half_width_extremes():
    assert window_half_width(0, pi / 6, pi / 4) == pi
    assert window_half_width(0, pi / 2, pi / 4) is None
    assert window_half_width(pi / 2, pi / 2, pi / 6) == pytest.approx(pi / 6)
    assert window_half_width(pi / 2, pi / 2, pi) == pi
    assert window_half_width(pi / 6, pi / 2, pi / 12) is None


@pytest.mark.parametrize("budget,good,decision", [(2.0, True, "freeze_budget"),
                                                  (1.8, False, "next_budget"),
                                                  (1.6, False, "cancel")])
def test_prescreen_uses_five_stage_policies_only(budget, good, decision):
    rows = {name: {} for name in (*PRESCREEN_POLICIES, "gstar")}
    for policy in rows:
        for opening in range(295100, 295148):
            event = "success" if good or policy == "gstar" else "resource_exhausted"
            rows[policy][opening] = {"mission": {"budget_m_s": budget, "urgent": False}, "event": event,
                                     "delta_v": .95 * budget, "return": 0}
    result = prescreen(rows, budget)
    assert result["decision"] == decision
    assert result["activated"] == 240 and result["runs"] == 240
    assert result["best_captures"] == (48 if good else 0)
