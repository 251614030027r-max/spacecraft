"""Read-only health check of a learned-stopping training run, safe while it trains.

Reads only files the run writes itself -- ``manifest.json``,
``train.monitor.csv`` and (if ``tensorboard`` is installed) the event file --
and never touches the process, the model or the environment.

What it can and cannot say. Training episodes are flown by the *behaviour*
policy: stochastic task actions and a handoff sampled with
``clip(beta, 0.002, 0.01)``. Their completion rate is therefore not the
method's performance (that is the deterministic formal evaluation) and is not
a gate. This script answers a narrower question: is the run the frozen
configuration, is it healthy (no NaN, solver sane, budget accounting right),
and are the learning signals moving -- does the task policy improve with
budget, does the learned branch keep flying into the mid/late episode, does
the stopping head's beta depend on the state rather than sitting at 0 or 1?

    python -B -m experiments.v3_stopping_training_health --run-dir logs/stop_262430 \
        --run-dir logs/stop_262431 --run-dir logs/stop_262432 [--reference-run logs/v3e_262420]

``--reference-run`` adds a V3e learned-only run for the same budget bins (its
episodes are all learned, so only the completion and return columns compare).
Bins are by cumulative outer decisions (Monitor ``l``), so runs line up by
budget, not by episode count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from train.stopping import STOPPING, stopping_configs

BIN_DECISIONS = 5_000
TIMEOUT_DECISIONS = 150  # 300 s / 2 s
LONG_LEARNED = 30        # 60 s, the D2 reading
TB_TAGS = (
    "stop/beta_mean",
    "stop/q_handoff_minus_v_continue",
    "stop/q_handoff_loss",
    "train/critic_loss",
    "train/actor_loss",
    "rollout/ep_rew_mean",
)


def read_monitor(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="") as handle:
        handle.readline()
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)
    return rows


def provenance(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    problems = []
    if manifest.get("method") is None:
        return manifest, problems  # a V3e reference run: nothing to check here
    if manifest.get("method") != "learned_stopping_option":
        problems.append(f"method is {manifest.get('method')!r}, not learned_stopping_option")
        return manifest, problems
    if manifest.get("code_dirty"):
        problems.append("training checkout was dirty (code_dirty=true)")
    if manifest.get("smoke_overrides"):
        problems.append(f"smoke overrides present: {manifest['smoke_overrides']}")
    if manifest.get("stopping") != asdict(STOPPING):
        problems.append("stopping config differs from the frozen STOPPING")
    _, hybrid = stopping_configs()
    if manifest.get("hybrid") != asdict(hybrid):
        problems.append("hybrid config differs from stopping_configs() of this checkout")
    if manifest.get("hyperparameters", {}).get("gamma") != STOPPING.gamma:
        problems.append("SAC gamma is not the stopping gamma")
    return manifest, problems


def _f(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else None


def bin_rows(rows: list[dict[str, Any]], bin_decisions: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cumulative = 0.0
    current: list[dict[str, Any]] = []
    edge = bin_decisions

    def flush(upper: float) -> None:
        if not current:
            return
        has_stop = "handoff" in current[0]
        handed = [r for r in current if has_stop and r["handoff"]]
        alone = [r for r in current if not has_stop or not r["handoff"]]
        learned = [r["learned_decisions"] if has_stop else r["l"] for r in current]
        simulated = [r["simulated_decisions"] if has_stop else r["l"] for r in current]
        out.append({
            "outer_decisions_upto": int(upper),
            "episodes": len(current),
            "completed": _f([float(r["completed"]) for r in current]),
            "return": _f([r["r"] for r in current]),
            "timeout_like": _f([float(s >= TIMEOUT_DECISIONS and not r["completed"]) for s, r in zip(simulated, current)]),
            "illegal_entries_per_ep": _f([r.get("illegal_terminal_entry_count", 0.0) for r in current]),
            "qp_fallback_rate": (
                float(sum(r["hybrid_v3_episode_qp_zero_fallbacks"] for r in current)
                      / max(1.0, sum(r["hybrid_v3_episode_control_steps"] for r in current)))
                if "hybrid_v3_episode_qp_zero_fallbacks" in current[0] else None
            ),
            "learned_decisions_mean": _f(learned),
            "learned_ge_30": _f([float(v >= LONG_LEARNED) for v in learned]),
            "handoff_fraction": (len(handed) / len(current)) if has_stop else None,
            "completed_after_handoff": _f([float(r["completed"]) for r in handed]) if has_stop else None,
            "completed_learned_alone": _f([float(r["completed"]) for r in alone]) if has_stop else None,
            "handoff_k_median": float(np.median([r["handoff_decision"] for r in handed])) if handed else None,
            "beta_mean": _f([r["stop_beta_mean"] for r in current]) if has_stop else None,
            "beta_max_over_half": _f([float(r["stop_beta_max"] >= 0.5) for r in current]) if has_stop else None,
            "beta_mean_over_half": _f([float(r["stop_beta_mean"] >= 0.5) for r in current]) if has_stop else None,
        })
        current.clear()

    for row in rows:
        cumulative += row["l"]
        current.append(row)
        if cumulative >= edge:
            flush(cumulative)
            while edge <= cumulative:
                edge += bin_decisions
    flush(cumulative)
    return out


def tensorboard_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]] | None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    files = sorted((run_dir / "tensorboard").glob("**/events.out.tfevents.*"))
    if not files:
        return None
    out: dict[str, list[tuple[int, float]]] = {}
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in TB_TAGS:
            if tag in accumulator.Tags().get("scalars", []):
                out.setdefault(tag, []).extend((e.step, e.value) for e in accumulator.Scalars(tag))
    return {tag: sorted(values) for tag, values in out.items()}


def flags(manifest: dict[str, Any], rows: list[dict[str, Any]], bins: list[dict[str, Any]],
          tb: dict[str, list[tuple[int, float]]] | None) -> list[str]:
    """Exploratory warnings. None of these is a preregistered gate."""

    notes = []
    if any(math.isnan(r["r"]) for r in rows):
        notes.append("NaN episode return")
    if tb:
        for tag, values in tb.items():
            if any(not math.isfinite(v) for _, v in values):
                notes.append(f"non-finite values in {tag}")
    if len(bins) >= 2:
        last = bins[-1]
        if last["learned_ge_30"] is not None and last["learned_ge_30"] < 0.25:
            notes.append(f"latest bin: only {last['learned_ge_30']:.0%} of episodes fly >= 30 learned decisions (D2a reads >= 25%)")
        if last["beta_mean_over_half"] is not None and last["beta_mean_over_half"] > 0.9:
            notes.append("latest bin: beta >= 0.5 on average in > 90% of episodes -- the stopping head wants to hand off almost everywhere")
        if last["beta_max_over_half"] is not None and last["beta_max_over_half"] < 0.05:
            notes.append("latest bin: beta never reaches 0.5 in > 95% of episodes -- the stopping head almost never wants to hand off")
        if last["qp_fallback_rate"] is not None and last["qp_fallback_rate"] > 0.05:
            notes.append(f"latest bin: QP zero-fallback rate {last['qp_fallback_rate']:.1%} of control steps")
        if last["timeout_like"] is not None and last["timeout_like"] > 0.5:
            notes.append(f"latest bin: {last['timeout_like']:.0%} timeout-like episodes (the hover pathology?)")
        full = [b for b in bins if b["episodes"] >= 5]
        if len(full) >= 3 and full[-1]["completed_learned_alone"] is not None and full[0]["completed_learned_alone"] is not None:
            if full[-1]["completed_learned_alone"] <= full[0]["completed_learned_alone"]:
                notes.append("learned-alone completion has not risen since the first bin")
    if tb and "stop/beta_mean" in tb and len(tb["stop/beta_mean"]) >= 10:
        tail = [v for _, v in tb["stop/beta_mean"][-max(1, len(tb["stop/beta_mean"]) // 10):]]
        if max(tail) - min(tail) < 1e-3 and (np.mean(tail) < 0.01 or np.mean(tail) > 0.99):
            notes.append("stop/beta_mean is flat at a corner in the latest tenth of logging")
    return notes


def report(run_dir: Path, bin_decisions: int) -> dict[str, Any]:
    manifest, problems = provenance(run_dir)
    rows = read_monitor(run_dir / "train.monitor.csv")
    bins = bin_rows(rows, bin_decisions)
    tb = tensorboard_scalars(run_dir)
    outer = float(sum(r["l"] for r in rows))
    suffix = float(sum(r.get("suffix_decisions", 0.0) for r in rows))
    elapsed_h = (rows[-1]["t"] / 3600.0) if rows else 0.0
    requested = (manifest.get("requested_outer_decisions") or manifest.get("requested_simulated_decisions")
                 or manifest.get("requested_decision_steps"))
    rate = outer / elapsed_h if elapsed_h > 0 else None
    checkpoints = sorted(p.name for p in (run_dir / "checkpoints").glob("*.zip")) if (run_dir / "checkpoints").exists() else []
    return {
        "run_dir": str(run_dir),
        "seed": manifest.get("seed"),
        "method": manifest.get("method", "v3e_learned_only"),
        "status": manifest.get("status"),
        "code_commit": manifest.get("code_commit"),
        "provenance_problems": problems,
        "episodes": len(rows),
        "outer_decisions_logged": outer,
        "suffix_decisions_logged": suffix,
        "suffix_over_outer": suffix / outer if outer else None,
        "requested": requested,
        "elapsed_h_at_last_episode": elapsed_h,
        "outer_decisions_per_h": rate,
        "eta_h": ((requested - outer) / rate) if (rate and requested) else None,
        "checkpoints": checkpoints,
        "bins": bins,
        "tensorboard_last": {tag: values[-1] for tag, values in (tb or {}).items()},
        "tensorboard_available": tb is not None,
        "flags": flags(manifest, rows, bins, tb) if rows else ["no finished episode yet"],
    }


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "   -  "
    return f"{value:6.0%}" if pct else f"{value:6.1f}"


def _fmt3(value: float | None) -> str:
    return "   -  " if value is None else f"{value:6.3f}"


def print_report(r: dict[str, Any]) -> None:
    print(f"\n=== {r['run_dir']}  seed {r['seed']}  ({r['method']}, status {r['status']}, commit {str(r['code_commit'])[:7]})")
    print("provenance: " + ("OK" if not r["provenance_problems"] else "PROBLEMS: " + "; ".join(r["provenance_problems"])))
    eta = f", ETA ~{r['eta_h']:.1f} h" if r["eta_h"] is not None else ""
    ratio = f" ({r['suffix_over_outer']:.2f}x of outer)" if r["suffix_over_outer"] is not None else ""
    print(f"progress: {r['episodes']} episodes, {r['outer_decisions_logged']:.0f}/{r['requested']} outer decisions "
          f"(finished episodes only), suffix {r['suffix_decisions_logged']:.0f}{ratio}, "
          f"{r['elapsed_h_at_last_episode']:.1f} h{eta}; checkpoints {len(r['checkpoints'])}")
    header = ("  upto  eps  compl return t/out  learned >=30  handoff c|hand c|alone k_med  beta b>.5avg b>.5max qpfb")
    print(header)
    for b in r["bins"]:
        print(f"{b['outer_decisions_upto']:6d} {b['episodes']:4d} {_fmt(b['completed'], True)} {_fmt(b['return'])}"
              f" {_fmt(b['timeout_like'], True)} {_fmt(b['learned_decisions_mean'])} {_fmt(b['learned_ge_30'], True)}"
              f" {_fmt(b['handoff_fraction'], True)} {_fmt(b['completed_after_handoff'], True)}"
              f" {_fmt(b['completed_learned_alone'], True)} {_fmt(b['handoff_k_median'])}"
              f" {_fmt3(b['beta_mean'])}"
              f" {_fmt(b['beta_mean_over_half'], True)} {_fmt(b['beta_max_over_half'], True)}"
              f" {_fmt(b['qp_fallback_rate'], True)}")
    if r["tensorboard_last"]:
        print("tensorboard (last): " + ", ".join(f"{k}={v[1]:.3g}@{v[0]}" for k, v in r["tensorboard_last"].items()))
    elif not r["tensorboard_available"]:
        print("tensorboard: not read (package missing or no event file)")
    print("flags (exploratory, not gates): " + ("none" if not r["flags"] else ""))
    for note in r["flags"]:
        print(f"  - {note}")


LEGEND = """
columns: upto = bin end in cumulative outer decisions; compl = completed (training behaviour, NOT the
method's score; violations not checked here); t/out = timeout-like (>= 150 simulated decisions, not
completed); learned = mean learned decisions per episode; >=30 = share flying >= 60 s on the learned
branch (D2a reads >= 25%); handoff = share of episodes that handed off (capped by the behaviour clip,
<= 1% per decision); c|hand / c|alone = completion of handed-off / learned-alone episodes; k_med = median
handoff decision; beta = mean of the stopping head's beta over the episode's learned decisions;
b>.5avg / b>.5max = share of episodes whose mean / max beta is >= 0.5 (the head would hand off there at
deployment); qpfb = QP zero-fallback share of control steps.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--reference-run", type=Path, action="append", default=[],
                        help="optional V3e learned-only run(s) for the same budget bins")
    parser.add_argument("--bin-decisions", type=int, default=BIN_DECISIONS)
    parser.add_argument("--output", type=Path, default=None, help="optional JSON dump")
    args = parser.parse_args(argv)

    reports = [report(run_dir, args.bin_decisions) for run_dir in args.run_dir + args.reference_run]
    print(LEGEND)
    for r in reports:
        print_report(r)
    stopping = [r for r in reports if r["method"] == "learned_stopping_option"]
    if len(stopping) > 1:
        print("\n=== across seeds (latest bin)")
        for r in stopping:
            b = r["bins"][-1] if r["bins"] else {}
            print(f"  seed {r['seed']}: upto {b.get('outer_decisions_upto')}, completed {_fmt(b.get('completed'), True)}, "
                  f"learned alone {_fmt(b.get('completed_learned_alone'), True)}, >=30 {_fmt(b.get('learned_ge_30'), True)}, "
                  f"b>.5max {_fmt(b.get('beta_max_over_half'), True)}, flags {len(r['flags'])}")
    if args.output is not None:
        args.output.write_text(json.dumps(reports, indent=1, default=str))


if __name__ == "__main__":
    main()
