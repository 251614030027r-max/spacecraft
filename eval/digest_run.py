"""Compact one-screen digest of a training run.

The artefacts of a 500k run are several megabytes; the handful of numbers that
actually decide whether to continue, abort or roll back a single-factor
experiment fit on one screen. This prints exactly those, so a run can be
reported without shipping its logs around.

    python -B -m eval.digest_run --run logs/<run-name>

Read-only; writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


# Abort gates from the current experiment protocol; see the gradient_steps
# commit message. Exceeding either is the divergence the lowered update ratio
# was originally patched around.
CRITIC_Q_ABORT = 10.0
ENTROPY_COEFFICIENT_ABORT = 0.1
ZERO_ACTION_SURVIVAL_S = 53.3

REPORTED_HYPERPARAMETERS = (
    "learning_rate",
    "tau",
    "gamma",
    "train_freq",
    "gradient_steps",
    "buffer_size",
    "learning_starts",
    "batch_size",
    "ent_coef",
)


def _load(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_lengths(path: Path) -> list[tuple[float, float]]:
    """Return (cumulative episode index, length) pairs from the monitor CSV."""

    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        handle.readline()  # monitor header comment
        return [
            (float(index), float(row["l"]))
            for index, row in enumerate(csv.DictReader(handle))
        ]


def _window_medians(lengths: list[float], windows: int = 5) -> list[float]:
    if not lengths:
        return []
    size = max(len(lengths) // windows, 1)
    return [
        median(lengths[start : start + size])
        for start in range(0, len(lengths), size)
    ][:windows]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-screen digest of a run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()
    run = args.run

    manifest = _load(run / "manifest.json")
    if manifest is not None:
        hyper = manifest.get("hyperparameters", {})
        print(f"run     {manifest.get('run_name')}  seed {manifest.get('seed')}")
        print(
            f"steps   {manifest.get('actual_model_timesteps')} / "
            f"{manifest.get('requested_timesteps')}   "
            f"mode {manifest.get('phase2_mode')}   "
            f"status {manifest.get('status')}"
        )
        print(
            "hyper   "
            + "  ".join(
                f"{name}={hyper[name]}"
                for name in REPORTED_HYPERPARAMETERS
                if name in hyper
            )
        )
        environment = manifest.get("evaluation_environment", {})
        print(f"schema  {environment.get('phase2_observation_schema')}")

    diagnostics = _load(run / "phase2_diagnostics.json")
    if diagnostics is not None:
        print("\nstep      critic_q1  critic_q2  alpha     act_std  sat")
        for probe in diagnostics.get("probes", []):
            print(
                f"{probe['training_step']:<9} "
                f"{probe['critic_q1']['mean']:>9.3f} "
                f"{probe['critic_q2']['mean']:>10.3f} "
                f"{probe['entropy_coefficient']:>8.5f} "
                f"{probe['actor_action_std']['mean']:>8.3f} "
                f"{probe['actor_action_saturation_fraction']:>5.2f}"
            )
        probes = diagnostics.get("probes", [])
        if probes:
            worst_q = max(probe["critic_q1"]["mean"] for probe in probes)
            worst_alpha = max(probe["entropy_coefficient"] for probe in probes)
            flags = []
            if worst_q > CRITIC_Q_ABORT:
                flags.append(f"ABORT critic_q1 {worst_q:.2f} > {CRITIC_Q_ABORT}")
            if worst_alpha > ENTROPY_COEFFICIENT_ABORT:
                flags.append(
                    f"ABORT alpha {worst_alpha:.4f} > {ENTROPY_COEFFICIENT_ABORT}"
                )
            print(
                "\ngates   "
                + ("; ".join(flags) if flags else "ok (no abort condition hit)")
            )

    lengths = [length for _, length in _episode_lengths(run / "train.monitor.csv")]
    if lengths:
        medians = _window_medians(lengths)
        print(
            "\nsurvival (training episode length, median per fifth, seconds)\n        "
            + "  ".join(f"{value * args.dt:.1f}" for value in medians)
            + f"   | zero-action baseline {ZERO_ACTION_SURVIVAL_S}"
        )
        tail = lengths[-200:]
        print(
            f"        last {len(tail)} episodes median "
            f"{median(tail) * args.dt:.1f} s"
        )

    evaluations = sorted((run / "evaluations").glob("evaluation_*.json"))
    if evaluations:
        print("\nstep      eval rates")
        for path in evaluations:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rates = payload.get("rates", {})
            print(
                f"{payload.get('training_step'):<9} "
                + "  ".join(f"{key}={value}" for key, value in rates.items())
            )

    calibrations = sorted(run.glob("value_calibration*.json"))
    if calibrations:
        print("\ncalibration (independent Monte-Carlo ground truth)")
        for path in calibrations:
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
            print(
                f"{path.stem:<38} "
                f"Q={summary['critic_q_min_s0']:+.3f} "
                f"soft={summary['soft_discounted_return']:+.3f} "
                f"err={summary['calibration_error']:.3f} "
                f"survival={summary['deterministic_survival_s']:.1f}s "
                f"gate={summary['deterministic_gate_rate']:.2f}"
            )


if __name__ == "__main__":
    main()
