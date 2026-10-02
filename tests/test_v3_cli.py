from __future__ import annotations

import sys

from experiments import evaluate_hybrid_policy
from train import train_hybrid


def test_training_cli_accepts_task_state_v3(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_hybrid",
            "--steps",
            "1",
            "--seed",
            "262420",
            "--run-name",
            "v3_cli_guard",
            "--parametrization",
            "task_state_v3",
        ],
    )
    assert train_hybrid.parse_args().parametrization == "task_state_v3"


def test_evaluator_cli_accepts_task_state_v3(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_hybrid_policy",
            "--episodes",
            "1",
            "--seed",
            "262000",
            "--output",
            str(tmp_path / "v3.json"),
            "--parametrization",
            "task_state_v3",
        ],
    )
    assert evaluate_hybrid_policy.parse_args().parametrization == "task_state_v3"
