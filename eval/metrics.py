"""Shared metric conventions for the three-way main table.

Completion rate cannot separate Pure SAC, Pure MPC and the hybrid -- the
scripted controller is already 20/20 on ``single_phase`` and both optimizer-based
methods should reach it -- so the table rests on completion time, force impulse,
worst constraint margin and per-step compute. Those four have to mean the same
thing on every path that fills a row, which is what this module is for. It is
imported by ``eval.evaluate_policy`` (the learned rows), by
``experiments.evaluate_mpc`` (the Pure MPC row) and by
``eval.validate_single_phase_semantics`` (the scripted reference row), so a row is
never assembled from a private convention.

Three conventions it pins, each because the alternative was already producing a
number that could not be compared:

* **Compute is the controller, not the simulator.** ``controller_time_s`` covers
  only the call that produces an action; ``environment_step_time_s`` covers the
  RK45 truth propagation. Timed together they over-report the controller by
  about 8.6x for Pure SAC -- measured on CPU, ``predict`` is 1.09 ms against
  ``env.step`` at 8.32 ms -- while the MPC path has always timed
  ``controller.command`` alone. A merged figure on one row and a controller-only
  figure on another is not a comparison.
* **Effort and time are read over completed episodes.** Force impulse is a time
  integral, so an episode that dies at 10 s scores a smaller one than an episode
  that flies the whole mission; ranking rows by the raw mean rewards dying early.
  The all-episode figures are kept beside them, never instead of them.
* **The worst margin is the worst over every episode**, completed or not. A
  method that reaches the pose by grazing a boundary on the episodes it loses
  has not earned a clean margin column.

Read-only and dependency-free: no environment, no model, no I/O.
"""

from __future__ import annotations

from statistics import median, pstdev
from typing import Any, Iterable, Sequence

import numpy as np


# The five signed task margins, in the order the environment reports them.
MARGIN_KEYS = (
    "corridor_axial_margin_m",
    "corridor_lateral_margin_m",
    "fov_margin_rad",
    "total_speed_margin_m_s",
    "closing_speed_margin_m_s",
)
PRECAPTURE_MARGIN_KEYS = (
    "keepout_margin_m",
    "fov_margin_rad",
    "outer_inertial_speed_margin_m_s",
    "target_frame_speed_margin_m_s",
    "corridor_axial_margin_m",
    "corridor_lateral_margin_m",
    "terminal_total_speed_margin_m_s",
    "closing_speed_margin_m_s",
)


def summarize(values: Iterable[float]) -> dict[str, float] | None:
    """Return the standard summary of a sample, or ``None`` if it is empty.

    ``None`` rather than zeros: a completion time over zero completed episodes
    does not exist, and a zero there would read as "instant" in a table.
    """

    sample = [float(value) for value in values]
    if not sample:
        return None
    ordered = sorted(sample)
    index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "count": len(ordered),
        "mean": float(np.mean(ordered)),
        "median": float(median(ordered)),
        "std": float(pstdev(ordered)) if len(ordered) > 1 else 0.0,
        "p95": float(ordered[max(index, 0)]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _completed(records: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [record for record in records if bool(record.get(key, False))]


def _split(
    records: Sequence[dict[str, Any]], completed: Sequence[dict[str, Any]], field: str
) -> dict[str, dict[str, float] | None]:
    """Summarize one per-episode quantity over completed and all episodes."""

    return {
        "completed_only": summarize(
            record[field] for record in completed if record.get(field) is not None
        ),
        "all_episodes": summarize(
            record[field] for record in records if record.get(field) is not None
        ),
    }


def main_table_metrics(
    records: Sequence[dict[str, Any]],
    *,
    controller_times_s: Sequence[float],
    environment_step_times_s: Sequence[float],
    control_period_s: float,
    completed_key: str = "completed",
) -> dict[str, Any]:
    """Assemble the four differentiating columns from per-episode records.

    ``records`` must carry ``survival_s``, ``force_impulse_n_s``,
    ``torque_impulse_nm_s``, ``minimum_margins`` and the flag named by
    ``completed_key``; ``discounted_return`` is used when present. The two time
    sequences are the per-step samples pooled over the whole evaluation, which
    the caller holds and the per-episode summaries have already collapsed.
    """

    if control_period_s <= 0.0:
        raise ValueError("control_period_s must be positive")
    episodes = list(records)
    if not episodes:
        raise ValueError("main-table metrics need at least one episode record")
    done = _completed(episodes, completed_key)
    controller = summarize(controller_times_s)
    environment = summarize(environment_step_times_s)
    worst = {
        key: min(
            float(record["minimum_margins"][key])
            for record in episodes
            if record.get("minimum_margins")
        )
        for key in (*MARGIN_KEYS, *PRECAPTURE_MARGIN_KEYS)
        if all(key in record.get("minimum_margins", {}) for record in episodes)
    }
    metrics: dict[str, Any] = {
        "episodes": len(episodes),
        "completed_episodes": len(done),
        "completion_rate": len(done) / len(episodes),
        # Completion time only exists for episodes that completed; an episode
        # that ran to the 200 s cap has a survival time, not a completion time.
        "completion_time_s": summarize(record["survival_s"] for record in done),
        "survival_s": summarize(record["survival_s"] for record in episodes),
        "force_impulse_n_s": _split(episodes, done, "force_impulse_n_s"),
        "torque_impulse_nm_s": _split(episodes, done, "torque_impulse_nm_s"),
        "worst_constraint_margin": worst,
        "per_step_compute_s": {
            "control_period_s": float(control_period_s),
            "controller": controller,
            "environment_step": environment,
            # The budget ratio is the headline: > 1 means the controller cannot
            # run in real time at this control period, whatever the simulator costs.
            "controller_mean_over_budget": (
                controller["mean"] / control_period_s if controller else None
            ),
            "controller_p95_over_budget": (
                controller["p95"] / control_period_s if controller else None
            ),
        },
    }
    if any("equivalent_delta_v_m_s" in record for record in episodes):
        metrics["equivalent_delta_v_m_s"] = _split(
            episodes, done, "equivalent_delta_v_m_s"
        )
    if any("minimum_normalized_margin" in record for record in episodes):
        metrics["minimum_normalized_margin"] = summarize(
            record["minimum_normalized_margin"]
            for record in episodes
            if record.get("minimum_normalized_margin") is not None
        )
    if any("constraint_violated" in record for record in episodes):
        metrics["constraint_violation_rate"] = float(
            np.mean([bool(record.get("constraint_violated", False)) for record in episodes])
        )
    for field in (
        "terminal_region_entry_time_s",
        "entry_target_frame_speed_m_s",
        "entry_attitude_error_rad",
        "entry_angular_velocity_error_rad_s",
        "entry_corridor_margin_m",
    ):
        if any(field in record for record in episodes):
            metrics[field] = _split(episodes, done, field)
    if any("discounted_return" in record for record in episodes):
        metrics["discounted_return"] = _split(episodes, done, "discounted_return")
    return metrics


__all__ = [
    "MARGIN_KEYS",
    "PRECAPTURE_MARGIN_KEYS",
    "main_table_metrics",
    "summarize",
]
