"""The smooth-nominal row: the V3 learned branch advanced at full rate, no model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def test_v3_nominal_runs_on_the_learned_branch_at_full_rate(tmp_path: Path) -> None:
    output = tmp_path / "nominal.json"
    subprocess.run(
        [
            sys.executable, "-B", "-m", "experiments.evaluate_hybrid_policy",
            "--episodes", "1", "--seed", "265000", "--horizon", "35",
            "--parametrization", "task_state_v3", "--phase-time-observation",
            "--execution-feedback", "--adaptive-task", "--control", "v3_nominal",
            "--max-decisions", "3", "--output", str(output),
        ],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    record = json.loads(output.read_text())["records"][0]
    assert record["branch"] == ["learned"] * 3
    assert all(action == [1.0, 1.0] for action in record["task_action_raw"])
    progress = [state[0] for state in record["task_state_applied"]]
    assert progress == sorted(progress) and progress[-1] > progress[0]
