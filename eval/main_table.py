"""Assemble the three-way main table from evaluation JSONs.

Completion rate cannot separate Pure SAC, Pure MPC and the hybrid; the four
columns that can -- completion time, force impulse, worst constraint margin and
per-step controller compute -- are emitted on every measurement path under the
conventions ``eval.metrics`` pins, each inside a ``main_table`` block. This reads
those blocks and prints them side by side, so the comparison is one table rather
than three files a reader has to reconcile.

Each ``--row`` is ``label=path`` to a JSON that carries a ``main_table`` block:

    python -B -m eval.main_table \
        --row "Scripted=logs/single_phase_scripted.json" \
        --row "Pure MPC=logs/mpc_single_phase.json" \
        --row "Pure SAC=logs/sac_260817_eval.json"

The learned row comes from ``eval.evaluate_policy``, the MPC row from
``experiments.evaluate_mpc --task single_phase --reference-source corridor_guidance``,
and the scripted reference row from ``eval.validate_single_phase_semantics``.
Rows are printed in the order given. Read-only; writes nothing unless --output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.metrics import MARGIN_KEYS


def _cell(summary: dict[str, Any] | None, key: str, scale: float = 1.0) -> str:
    if not summary or summary.get(key) is None:
        return "--"
    return f"{float(summary[key]) * scale:.3f}"


def _load_table(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    table = payload.get("main_table")
    if table is None:
        raise ValueError(f"{path} carries no main_table block")
    return table


def _provenance(path: Path) -> dict[str, Any]:
    """Read the alignment fields three rows must share to be comparable.

    Different tasks, seed blocks or episode counts make a table apples to
    oranges, so surface them for a human to confirm rather than assuming the
    three JSONs were produced against the same conditions. Fields absent on
    older files simply print as a dash.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    environment = payload.get("environment", {})
    mission = environment.get("phase2_mission", {}) if isinstance(environment, dict) else {}
    distance = "--"
    if mission.get("initial_distance_min_m") is not None:
        distance = (
            f"{mission['initial_distance_min_m']:.0f}-"
            f"{mission['initial_distance_max_m']:.0f} m"
        )
    seed = payload.get("base_seed", payload.get("seed"))
    reference = payload.get("mpc_config", {}).get("reference_source")
    return {
        "episodes": payload.get("episodes", payload.get("reachability", {}).get("episodes")),
        "seed": seed,
        "distance": distance,
        "reference": reference,
    }


def _row(label: str, table: dict[str, Any]) -> dict[str, str]:
    compute = table["per_step_compute_s"]
    controller = compute.get("controller") or {}
    period = compute.get("control_period_s")
    mean_over = compute.get("controller_mean_over_budget")
    p95_over = compute.get("controller_p95_over_budget")
    # Real-time is bounded by the worst case, not the mean, so the headline
    # budget columns are p95 and max. The exact-linearization refresh step fires
    # once every few seconds and dominates both while barely moving the mean;
    # reporting only the mean (as this table used to) hides an over-budget
    # controller behind an in-budget average. max/period is derived here so the
    # column works on evaluation files written before the metric existed.
    max_s = controller.get("max")
    max_over = (max_s / period) if (max_s is not None and period) else None
    force = table["force_impulse_n_s"]["completed_only"]
    returns = table.get("discounted_return", {}).get("completed_only")
    worst = table["worst_constraint_margin"]

    def budget(value: float | None) -> str:
        return "--" if value is None else f"{value:.2f}x"

    return {
        "method": label,
        "completion": f"{table['completed_episodes']}/{table['episodes']}",
        "time_s": _cell(table["completion_time_s"], "mean"),
        "force_ns": _cell(force, "mean"),
        "worst_margin": (
            "--"
            if not worst
            else f"{min(worst.values()):+.3f}"
        ),
        "compute_ms": _cell(controller, "mean", 1e3),
        "p95_ms": _cell(controller, "p95", 1e3),
        "budget_x": budget(mean_over),
        "budget_p95_x": budget(p95_over),
        "budget_max_x": budget(max_over),
        "return": _cell(returns, "mean"),
    }


def _render(rows: list[dict[str, str]]) -> str:
    columns = [
        ("method", "method", "<"),
        ("completion", "completed", ">"),
        ("time_s", "time s", ">"),
        ("force_ns", "force N*s", ">"),
        ("worst_margin", "worst margin", ">"),
        ("compute_ms", "mean ms", ">"),
        ("p95_ms", "p95 ms", ">"),
        ("budget_p95_x", "budget p95", ">"),
        ("budget_max_x", "budget max", ">"),
        ("return", "return", ">"),
    ]
    widths = {
        key: max(len(header), *(len(row[key]) for row in rows))
        for key, header, _ in columns
    }
    def line(values: dict[str, str]) -> str:
        return "  ".join(
            f"{values[key]:{align}{widths[key]}}" for key, _, align in columns
        )
    header = line({key: header for key, header, _ in columns})
    rule = "  ".join("-" * widths[key] for key, _, _ in columns)
    return "\n".join([header, rule, *(line(row) for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way main table")
    parser.add_argument(
        "--row",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a labelled evaluation JSON; repeat for each method",
    )
    parser.add_argument("--output", type=Path, help="also write the rows as JSON")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    provenance: list[tuple[str, dict[str, Any]]] = []
    for entry in args.row:
        if "=" not in entry:
            raise ValueError(f"--row must be LABEL=PATH, got {entry!r}")
        label, _, raw_path = entry.partition("=")
        path = Path(raw_path)
        rows.append(_row(label.strip(), _load_table(path)))
        provenance.append((label.strip(), _provenance(path)))

    print(
        "\nThree-way main table -- completion rate does not separate the methods;"
        "\nthese four columns do. Compute is the controller only (the RK45 truth"
        "\nstep is excluded); force impulse and time are over completed episodes;"
        "\nthe worst margin spans every episode.\n"
    )
    print(_render(rows))
    print(
        "\nworst margin is the single tightest of the five task margins over all"
        " episodes. compute is the controller only; budget is controller time /"
        " control period (>1 = over budget). Real-time is bounded by the worst"
        " case, so budget p95 and budget max are the headline -- an in-budget"
        " mean can still hide an over-budget refresh step (mean ms shown for"
        " reference)."
    )
    print("\nalignment (confirm the rows are one comparison):")
    for label, item in provenance:
        print(
            f"  {label:<22} episodes={item['episodes']}  seed={item['seed']}"
            f"  start={item['distance']}"
            + (f"  reference={item['reference']}" if item['reference'] else "")
        )
    episode_counts = {item["episodes"] for _, item in provenance if item["episodes"]}
    seeds = {item["seed"] for _, item in provenance if item["seed"] is not None}
    if len(episode_counts) > 1 or len(seeds) > 1:
        print(
            "  WARNING: rows differ in episode count or seed block -- not a"
            " like-for-like comparison."
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"rows": rows, "margins": list(MARGIN_KEYS)}, indent=2),
            encoding="utf-8",
        )
        print(f"\nrows written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
