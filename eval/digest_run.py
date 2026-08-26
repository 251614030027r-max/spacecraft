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


# Abort thresholds from the current experiment protocol; see the gradient_steps
# commit message. Exceeding either is the divergence the lowered update ratio
# was originally patched around.
CRITIC_Q_ABORT = 10.0
ENTROPY_COEFFICIENT_ABORT = 0.1
# Zero-action survival differs per task: the two-phase modes only enforce the
# terminal constraints after the Waypoint, single_phase enforces them from step one.
ZERO_ACTION_SURVIVAL_S = 53.3
SINGLE_PHASE_ZERO_ACTION_SURVIVAL_S = 16.9
SCRIPTED_COMPLETION_S = 95.3
OUTCOME_RATES = (
    "episode_completion",
    "waypoint_acquisition",
    "final_completion",
    "constraint_success",
)

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


def _cell(summary: dict[str, Any] | None, key: str, scale: float = 1.0) -> str:
    """Format one summary field, or a dash when the sample does not exist."""

    if not summary or summary.get(key) is None:
        return "--"
    return f"{float(summary[key]) * scale:.3f}"


def _print_main_table(evaluation: dict[str, Any]) -> None:
    """Print the four columns that separate the three methods.

    Completion rate cannot: the scripted controller is already 20/20 and both
    optimizer-based methods should reach it. These four can, so they are printed
    together, at the latest checkpoint, in the shared conventions -- controller
    compute without the simulator, effort and time over completed episodes.
    """

    table = evaluation.get("main_table")
    if not table:
        return
    step = evaluation.get("training_step")
    completed = table["completed_episodes"]
    print(
        f"\nmain table  (step {step}; {completed}/{table['episodes']} completed)"
    )
    time_s = table.get("completion_time_s")
    print(
        "        completion time (completed only)  "
        f"mean {_cell(time_s, 'mean')} s   median {_cell(time_s, 'median')} s"
        f"   max {_cell(time_s, 'max')} s"
        + ("" if completed else "   (no completed episode yet)")
    )
    force = table["force_impulse_n_s"]
    print(
        "        force impulse                     "
        f"mean {_cell(force['completed_only'], 'mean')} N*s (completed)"
        f"   {_cell(force['all_episodes'], 'mean')} N*s (all episodes)"
    )
    print(
        "        worst constraint margin (all)     "
        + "  ".join(
            f"{name.replace('_margin_m_s','').replace('_margin_m','').replace('_margin_rad','')}"
            f"={value:+.3f}"
            for name, value in table["worst_constraint_margin"].items()
        )
    )
    compute = table["per_step_compute_s"]
    period = compute["control_period_s"]
    controller = compute["controller"]
    over = compute.get("controller_mean_over_budget")
    print(
        "        per-step controller compute       "
        f"mean {_cell(controller, 'mean', 1e3)} ms   p95 "
        f"{_cell(controller, 'p95', 1e3)} ms"
        f"   of a {period * 1e3:.0f} ms period"
        + (f"   ({over:.2f}x budget)" if over is not None else "")
    )
    print(
        "        (environment RK45 step, excluded) "
        f"mean {_cell(compute['environment_step'], 'mean', 1e3)} ms"
    )
    returns = table.get("discounted_return")
    if returns:
        print(
            "        discounted return                 "
            f"mean {_cell(returns['all_episodes'], 'mean')} (all episodes)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-screen digest of a run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()
    run = args.run

    manifest = _load(run / "manifest.json")
    mode = manifest.get("phase2_mode") if manifest is not None else None
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
                "\nguards  "
                + ("; ".join(flags) if flags else "ok (no abort condition hit)")
            )

    lengths = [length for _, length in _episode_lengths(run / "train.monitor.csv")]
    if lengths:
        medians = _window_medians(lengths)
        print(
            "\nsurvival (training episode length, median per fifth, seconds)\n        "
            + "  ".join(f"{value * args.dt:.1f}" for value in medians)
            + "   | zero-action baseline "
            + (
                f"{SINGLE_PHASE_ZERO_ACTION_SURVIVAL_S}"
                if mode == "single_phase"
                else f"{ZERO_ACTION_SURVIVAL_S}"
            )
            + "\n        "
            + {
                "phase1_pretrain": (
                    "(in phase1_pretrain a Waypoint success ends the episode, so short"
                    " episodes at a high Waypoint rate are success, not failure)"
                ),
                "single_phase": (
                    "(in single_phase the task constraints are live from step one:"
                    f" scripted completion runs ~{SCRIPTED_COMPLETION_S} s, a short"
                    " episode is a death, and ~200 s means the cap was hit without"
                    " completing -- read it with the failure mix below)"
                ),
            }.get(
                mode,
                "(in full_mission the Waypoint does not end the episode:"
                f" scripted completion runs ~{SCRIPTED_COMPLETION_S} s, so a short"
                " episode is a death, and ~200 s means the cap was hit without"
                " completing)",
            )
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
            outcome = {k: v for k, v in rates.items() if k in OUTCOME_RATES}
            failures = {k: v for k, v in rates.items() if k not in OUTCOME_RATES}
            print(
                f"{payload.get('training_step'):<9} "
                + "  ".join(f"{key}={value}" for key, value in outcome.items())
            )
            if failures:
                print(
                    " " * 10
                    + "  ".join(f"{key}={value}" for key, value in failures.items())
                )

    if evaluations:
        latest = json.loads(evaluations[-1].read_text(encoding="utf-8"))
        _print_main_table(latest)

    calibrations = sorted(run.glob("value_calibration*.json"))
    if calibrations:
        print("\ncalibration (independent Monte-Carlo ground truth)")
        for path in calibrations:
            summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
            # The soft return carries alpha * entropy accumulated per step, so a
            # long-surviving policy collects a large bonus that has nothing to do
            # with task quality. Print it, and the hard return, so a calibration
            # error can be attributed rather than just read off.
            print(
                f"{path.stem:<34} "
                f"Q={summary['critic_q_min_s0']:+.3f} "
                f"hard={summary['hard_discounted_return']:+.3f} "
                f"soft={summary['soft_discounted_return']:+.3f} "
                f"(entropy {summary['entropy_contribution']:+.3f}) "
                f"err={summary['calibration_error']:+.3f} "
                f"survival={summary['deterministic_survival_s']:.1f}s"
            )


if __name__ == "__main__":
    main()
