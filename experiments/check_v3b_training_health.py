"""Preregistered health check for v3b training runs (read-only).

Reads each run's ``train.monitor.csv`` and applies the stop rules in
``docs/V3B_TRAINING_ORDER_20260925.md``:

- STOP if any episode reports a cross-decision reference-jump violation
  (the T0b bound is provable, so a violation is a bug, not a result);
- STOP if the QP zero-fallback rate over the last 100 episodes exceeds 1e-3
  (matched T0/T0b probe: ~4e-5; V2 nominal evaluation: 2.3e-5).

Completion rate, return and channel activity are printed for the record and
are NOT stop criteria.

    python -B -m experiments.check_v3b_training_health logs/v3b_262420 logs/v3b_262421 logs/v3b_262422
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FALLBACK_RATE_LIMIT = 1.0e-3
WINDOW = 100


def check(run_dir: Path) -> dict:
    path = run_dir / "train.monitor.csv"
    if not path.exists():
        return {"run": str(run_dir), "status": "STOP", "stop_reasons": [f"missing {path}"]}
    with path.open(newline="") as handle:
        first = handle.readline()
        if not first.startswith("#"):
            handle.seek(0)
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"run": str(run_dir), "episodes": 0, "status": "NO_EPISODES_YET"}

    def number(row: dict, key: str) -> float:
        value = row.get(key, "")
        if value in ("", None):
            return 0.0
        if value in ("True", "False"):  # Monitor writes booleans as text
            return 1.0 if value == "True" else 0.0
        return float(value)

    violations = sum(int(number(r, "hybrid_v3_episode_reference_jump_violations")) for r in rows)
    recent = rows[-WINDOW:]
    fallbacks = sum(number(r, "hybrid_v3_episode_qp_zero_fallbacks") for r in recent)
    steps = sum(number(r, "hybrid_v3_episode_control_steps") for r in recent)
    rate = fallbacks / steps if steps else 0.0
    completed = sum(number(r, "completed") > 0.5 for r in recent) if "completed" in rows[0] else None
    decisions = int(sum(number(r, "l") for r in rows))
    reasons = []
    if violations:
        reasons.append(f"reference-jump violations = {violations} (must be 0)")
    if rate > FALLBACK_RATE_LIMIT:
        reasons.append(f"QP zero-fallback rate over last {len(recent)} episodes = {rate:.2e} > {FALLBACK_RATE_LIMIT:.0e}")
    return {
        "run": str(run_dir),
        "decisions": decisions,
        "episodes": len(rows),
        "reference_jump_violations_total": violations,
        "qp_zero_fallback_rate_last_window": rate,
        "max_reference_jump_target_m_last_window": max(
            number(r, "hybrid_v3_episode_reference_jump_target_max_m") for r in recent
        ),
        "completed_last_window": completed,
        "window": len(recent),
        "mean_return_last_window": sum(number(r, "r") for r in recent) / len(recent),
        "status": "STOP" if reasons else "OK",
        "stop_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    results = [check(run) for run in args.runs]
    print(json.dumps(results, indent=1))
    sys.exit(1 if any(r["status"] == "STOP" for r in results) else 0)


if __name__ == "__main__":
    main()
