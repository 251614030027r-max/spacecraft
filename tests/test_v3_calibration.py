"""M6 calibration summary: decisive agreement, critical states, wrong picks."""

from __future__ import annotations

from experiments.v3_calibrate_values import summarise


def _row(k, mu_l, mu_b, g_l, g_b, completed_l, completed_b, sd=0.1):
    return dict(k=k, mu_L=mu_l, sd_L=sd, mu_B=mu_b, sd_B=sd, G_L=g_l, G_B=g_b,
                completed_L=completed_l, completed_B=completed_b)


def test_near_ties_do_not_count_and_wrong_picks_are_counted() -> None:
    rows = [
        _row(0, 1.0, 0.0, 10.0, 10.0005, True, True),  # tie: wrong sign, irrelevant
        _row(10, 5.0, 0.0, 15.0, -5.0, True, False),  # right
        _row(20, 0.0, 5.0, 15.0, -5.0, True, False),  # wrong, and hands back into a failure
    ]
    summary = summarise(rows, z=1.0)
    assert summary["decisive_checkpoints"] == 2
    assert summary["sign_agreement_decisive"] == 0.5
    assert summary["critical_checkpoints"] == 2
    assert summary["wrong_picks"] == 1
    assert summary["gate"] == "INCONCLUSIVE"  # only 2 decisive checkpoints
    assert summary["corr_sd_abs_error_L"] is None  # constant spread carries no information


def test_too_few_decisive_checkpoints_is_inconclusive_not_a_pass() -> None:
    summary = summarise([_row(0, 1.0, 0.0, 10.0, 10.2, True, True)], z=1.0)
    assert summary["decisive_checkpoints"] == 0
    assert summary["sign_agreement_decisive"] is None
    assert summary["gate"] == "INCONCLUSIVE"
    few = [_row(k, 5.0, 0.0, 15.0, -5.0, True, False) for k in (0, 10, 20, 30)]
    assert summarise(few, z=1.0)["gate"] == "INCONCLUSIVE"  # 4 < 5, even at 100%


def test_enough_agreeing_decisive_checkpoints_pass() -> None:
    rows = [_row(k, 5.0, 0.0, 15.0, -5.0, True, False) for k in range(5)]
    assert summarise(rows, z=1.0)["gate"] == "PASS"


def test_enough_decisive_checkpoints_below_80_percent_stop() -> None:
    right = [_row(k, 5.0, 0.0, 15.0, -5.0, True, False) for k in range(3)]
    wrong = [_row(k, 0.0, 5.0, 15.0, -5.0, True, False) for k in range(3, 5)]
    assert summarise(right + wrong, z=1.0)["gate"] == "STOP"  # 3/5 = 60%
