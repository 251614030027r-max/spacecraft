"""Training health check: reads only the run's own files and bins by outer decisions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from experiments.v3_stopping_training_health import report
from train.stopping import STOPPING, STOPPING_SAC, stopping_configs
from train.hybrid_configs import serializable_hybrid_hyperparameters

COLUMNS = ("r,l,t,completed,illegal_terminal_entry_count,hybrid_v3_episode_qp_zero_fallbacks,"
           "hybrid_v3_episode_control_steps,handoff,handoff_decision,learned_decisions,suffix_decisions,"
           "simulated_decisions,stop_beta_mean,stop_beta_max,stop_beta_last")


def _run(tmp_path: Path, rows: list[str], **manifest_overrides) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    _, hybrid = stopping_configs()
    manifest = {"method": "learned_stopping_option", "seed": 1, "status": "running", "code_commit": "abc",
                "code_dirty": False, "smoke_overrides": None, "stopping": asdict(STOPPING), "hybrid": asdict(hybrid),
                "hyperparameters": serializable_hybrid_hyperparameters(STOPPING_SAC),
                "requested_outer_decisions": 1000, **manifest_overrides}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "train.monitor.csv").write_text('#{"t_start": 0}\n' + COLUMNS + "\n" + "\n".join(rows) + "\n")
    return run


def test_report_bins_and_flags(tmp_path: Path) -> None:
    early = "-5,60,3600,False,0,0,1200,False,-1,60,0,60,0.01,0.02,0.01"
    handed = "15,11,7200,True,0,0,1000,True,10,10,40,50,0.2,0.6,0.6"
    late = "16,40,10800,True,1,2,800,False,-1,40,0,40,0.1,0.3,0.2"
    r = report(_run(tmp_path, [early, early, handed, late, late]), bin_decisions=100)
    assert r["provenance_problems"] == [] and r["episodes"] == 5
    assert r["outer_decisions_logged"] == 211 and r["suffix_decisions_logged"] == 40
    first, last = r["bins"][0], r["bins"][-1]
    assert first["completed"] == 0.0 and first["learned_ge_30"] == 1.0
    assert last["completed"] == 1.0 and abs(last["handoff_fraction"] - 1 / 3) < 1e-9 and last["completed_after_handoff"] == 1.0
    assert r["eta_h"] is not None and r["eta_h"] > 0


def test_report_flags_a_dirty_or_smoke_run(tmp_path: Path) -> None:
    r = report(_run(tmp_path, ["1,5,10,True,0,0,100,False,-1,5,0,5,0.0,0.0,0.0"], code_dirty=True,
                    smoke_overrides={"net_arch": [16]}), bin_decisions=100)
    assert any("dirty" in p for p in r["provenance_problems"])
    assert any("smoke" in p for p in r["provenance_problems"])
