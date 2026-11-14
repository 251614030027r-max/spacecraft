from argparse import Namespace
from hashlib import sha256

import pytest

from experiments.final import check_formal, checked_cases, stage_of, verify_existing


def test_formal_budget_and_smoke_boundaries():
    args = Namespace(stage="formal", budget=0.3, openings="294300-294301")
    with pytest.raises(SystemExit, match="development smoke"):
        checked_cases(args, stage_of("formal"), "urgent", False)
    ops, mission = checked_cases(args, stage_of("formal"), "urgent", True)
    assert ops == [294300, 294301]
    assert mission.sample(294300)["budget_m_s"] == 0.3
    args.openings = "300000-300001"
    with pytest.raises(SystemExit, match="formal evaluation block"):
        checked_cases(args, stage_of("formal"), "urgent", True)


def test_prescreen_budget_and_validation_openings():
    stage = stage_of("formal_prescreen")
    args = Namespace(stage="formal_prescreen", budget=1.8, openings=None)
    ops, mission = checked_cases(args, stage, None, False)
    assert ops == list(range(295100, 295148))
    assert not mission.sample(295100)["urgent"]
    assert mission.sample(295100)["budget_m_s"] == 1.8
    args.budget = 0.3
    with pytest.raises(SystemExit, match="unregistered prescreen"):
        checked_cases(args, stage, None, False)
    args.stage, args.budget = "formal", None
    ops, mission = checked_cases(args, stage_of("formal"), "validation", False)
    assert ops == list(range(295200, 295224)) and mission.sample(295200)["urgent"]


def test_resume_verifies_existing_bytes_before_skip(tmp_path):
    folder = tmp_path / "C" / "trial"
    folder.mkdir(parents=True)
    path = folder / "trial_294300.json"
    path.write_bytes(b'{"opening": 294300}')
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{sha256(path.read_bytes()).hexdigest()}  C/trial/{path.name}\n", encoding="utf-8")
    verify_existing(folder, manifest, [294300, 294301], "trial")
    path.write_bytes(b'{"opening": 294301}')
    with pytest.raises(SystemExit, match="mismatch"):
        verify_existing(folder, manifest, [294300, 294301], "trial")


def test_completeness_report_lists_every_registered_result(tmp_path):
    report = check_formal(tmp_path, stage_of("formal"))
    assert len(report["missing_runs"]) == 18
    assert report["evaluations"]["ECC_CERT_LOOP_s20_economy_nocert"]["expected"] == 96
    assert report["evaluations"]["ECC_CERT_LOOP_s20_economy"]["expected"] == 96
    assert not any("resource" in key for key in (*report["evaluations"], *report["screens"]))
    assert report["screens"]["pure_natural"]["expected"] == 96 and "gstar_natural" not in report["screens"]
    assert report["validation"]["TD3_C_s21_step5000"]["expected"] == 24
    assert not report["complete"]
