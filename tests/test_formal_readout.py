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


def test_eta_counts_delta_v_already_used():
    from experiments.formal_readout import mechanism

    cand = {"b": [0, 0], "event": 1, "kind": "entry", "t": 10.0, "dv": 0.3, "dv_entry": 0.3,
            "certified": False, "binding": False}
    decision = {"cert": {"cands": [cand] * 9, "intervened": False},
                "remaining_s": 100.0, "remaining_dv": 0.5}
    row = {"initial": {"omega": [0.0, 0.0, 0.04]}, "budget_m_s": 3.0, "decisions": [decision]}
    out = mechanism({(20, 1): row})
    assert out["decisions"][0]["eta"] is None           # 0.6 m/s predicted > 0.5 m/s left
    decision["remaining_dv"] = 0.7
    eta = mechanism({(20, 1): row})["decisions"][0]["eta"]
    assert eta == pytest.approx((100.0 - 45.0 - 10.0) * 0.04 / (2 * pi))


def test_numerical_failure_is_a_non_capture_outcome(tmp_path):
    from experiments.formal_readout import opening_rows, summary

    for opening, event in ((1, "success"), (2, "numerical_failure")):
        (tmp_path / f"r_{opening}.json").write_text(
            '{"opening": %d, "event": "%s", "return": 0}' % (opening, event), encoding="utf-8")
    rows = opening_rows(sorted(tmp_path.glob("*.json")))
    assert summary(rows.values())["captures"] == 1
    (tmp_path / "r_3.json").write_text('{"opening": 3, "event": "error"}', encoding="utf-8")
    with pytest.raises(ValueError):
        opening_rows(sorted(tmp_path.glob("*.json")))


def test_episode_table_reads_all_result_kinds(tmp_path):
    from experiments.formal_readout import episode_table

    body = '{"opening": %d, "event": "success", "return": 1.0, "time_s": 9, "delta_v": 1.0, "mission": {"urgent": true}}'
    for folder, opening in (("C/ECC_CERT_LOOP_s20_urgent_nocert", 1), ("S/pure_urgent", 1), ("V/SAC_C_s21_step1000", 2)):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / f"x_{opening}.json").write_text(body % opening, encoding="utf-8")
    rows = {r["kind"]: r for r in episode_table(tmp_path)}
    assert (rows["C"]["config"], rows["C"]["seed"], rows["C"]["switch"]) == ("ECC_CERT_LOOP", "20", "nocert")
    assert (rows["S"]["config"], rows["S"]["condition"]) == ("pure", "urgent")
    assert (rows["V"]["config"], rows["V"]["seed"], rows["V"]["switch"]) == ("SAC_C", "21", "step1000")
