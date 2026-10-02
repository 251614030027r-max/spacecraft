from __future__ import annotations

from pathlib import Path

from experiments.check_v3b_training_health import check

HEADER = (
    "r,l,t,completed,hybrid_v3_episode_reference_jump_target_max_m,"
    "hybrid_v3_episode_reference_jump_violations,"
    "hybrid_v3_episode_qp_zero_fallbacks,hybrid_v3_episode_control_steps"
)


def _write(run: Path, rows: list[str]) -> Path:
    run.mkdir()
    (run / "train.monitor.csv").write_text(
        '#{"t_start": 0, "env_id": "None"}\n' + HEADER + "\n" + "\n".join(rows) + "\n"
    )
    return run


def test_health_check_passes_clean_run(tmp_path: Path) -> None:
    run = _write(tmp_path / "ok", ["1.0,100,1.0,True,2.5,0,0,2000"] * 5)
    result = check(run)
    assert result["status"] == "OK"
    assert result["decisions"] == 500


def test_health_check_stops_on_any_jump_violation(tmp_path: Path) -> None:
    run = _write(tmp_path / "jump", ["1.0,100,1.0,False,9.0,1,0,2000"])
    assert check(run)["status"] == "STOP"


def test_health_check_stops_on_high_fallback_rate(tmp_path: Path) -> None:
    run = _write(tmp_path / "qp", ["1.0,100,1.0,False,2.5,0,3,2000"])
    result = check(run)
    assert result["status"] == "STOP"
    assert result["qp_zero_fallback_rate_last_window"] > 1e-3
