"""Validate the exact-linearization MPC recovery configuration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from controllers.mpc import MPCConfig
from experiments.evaluate_mpc import evaluate


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_NAME = "mpc_recovery_pipeline_exact_v1"


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_default),
        encoding="utf-8",
    )


def status(path: Path, phase: str, **extra: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    value.update(
        phase=phase,
        updated_at_utc=datetime.now(timezone.utc).isoformat(),
        **extra,
    )
    write(path, value)


def main() -> None:
    output = ROOT / "logs" / PIPELINE_NAME
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    status_path = output / "status.json"
    config = MPCConfig(
        horizon_steps=50,
        outer_iterations=1,
        linearization_source="exact",
        exact_linearization_refresh_steps=10,
    )
    status(status_path, "screening")
    candidate = evaluate(config, episodes=20, seed=20285000)
    write(output / "candidate_20.json", candidate)
    status(status_path, "evaluating_official")
    official = evaluate(config, episodes=100, seed=20286000)
    write(output / "official_100.json", official)
    acceptance = {
        "algorithm": "mpc_only",
        "config": candidate["mpc_config"],
        "rates": official["rates"],
        "qp": official["qp"],
        "acceptance": official["acceptance"],
    }
    write(output / "acceptance.json", acceptance)
    status(
        status_path,
        "completed",
        passed=bool(official["acceptance"]["passed"]),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        output = ROOT / "logs" / PIPELINE_NAME
        output.mkdir(parents=True, exist_ok=True)
        status(
            output / "status.json",
            "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
