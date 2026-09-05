"""Derivative-free S5 search over a small piecewise 3D waypoint family."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from controllers.mpc import precapture_mpc_config
from env.phase2_env import precapture_planning_environment_config
from experiments.evaluate_mpc import evaluate
from experiments.piecewise_guidance import (
    PIECEWISE_PLANNING_FOV_HALF_ANGLE_RAD,
    PiecewiseGuidanceParameters,
)


CANDIDATES = (
    PiecewiseGuidanceParameters(8.0, 8.0, 70.0, 150.0),
    PiecewiseGuidanceParameters(6.0, 6.0, 70.0, 150.0),
    PiecewiseGuidanceParameters(8.0, 6.0, 60.0, 140.0),
    PiecewiseGuidanceParameters(10.0, 6.0, 80.0, 160.0),
    PiecewiseGuidanceParameters(6.0, 6.0, 100.0, 180.0),
    PiecewiseGuidanceParameters(8.0, 8.0, 100.0, 180.0),
    PiecewiseGuidanceParameters(10.0, 6.0, 40.0, 100.0),
    PiecewiseGuidanceParameters(8.0, 6.0, 40.0, 100.0),
    PiecewiseGuidanceParameters(10.0, 6.0, 20.0, 80.0),
    PiecewiseGuidanceParameters(8.0, 6.0, 20.0, 80.0),
    PiecewiseGuidanceParameters(10.0, 8.0, 40.0, 100.0),
    PiecewiseGuidanceParameters(8.0, 6.0, 60.0, 120.0),
)


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    table = result["main_table"]
    return {
        "rates": result["rates"],
        "completion_time_s": table["completion_time_s"],
        "equivalent_delta_v_m_s": table["equivalent_delta_v_m_s"],
        "controller_compute_s": table["per_step_compute_s"]["controller"],
        "worst_constraint_margin": table["worst_constraint_margin"],
        "episodes": [
            {
                key: record.get(key)
                for key in (
                    "seed",
                    "completed",
                    "steps",
                    "distance_failure",
                    "time_failure",
                    "illegal_terminal_entry_count",
                    "first_violation",
                )
            }
            for record in result["episode_records"]
        ],
    }


def _score(summary: dict[str, Any], episodes: int) -> float:
    rates = summary["rates"]
    unsafe = episodes - round(float(rates["constraint_success"]) * episodes)
    incomplete = episodes - round(float(rates["completed"]) * episodes)
    mean_survival = sum(item["steps"] for item in summary["episodes"]) * 0.1 / episodes
    delta_v = summary["equivalent_delta_v_m_s"]["all_episodes"]["mean"]
    return float(unsafe * 1.0e8 + incomplete * 1.0e6 + mean_survival * 100.0 + delta_v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--max-candidates", type=int, default=len(CANDIDATES))
    parser.add_argument("--start-candidate", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_candidates <= len(CANDIDATES):
        raise ValueError("max-candidates is outside the deterministic search set")
    if not 0 <= args.start_candidate < args.max_candidates:
        raise ValueError("start-candidate must precede max-candidates")

    environment = precapture_planning_environment_config()
    config = precapture_mpc_config(
        horizon_steps=20,
        terminal_weight=1000.0,
        reference_source="external_local",
        external_reference_hold_steps=20,
    )
    config = replace(
        config,
        precapture_task=replace(
            config.precapture_task,
            fov_half_angle_rad=PIECEWISE_PLANNING_FOV_HALF_ANGLE_RAD,
        ),
    )
    trials = []
    best_result = None
    best_score = float("inf")
    def write_output(selected_result: dict[str, Any] | None) -> None:
        output = {
            "schema_version": 1,
            "method": "deterministic_derivative_free_candidate_search",
            "objective_priority": "truth_safety_and_completion_then_time_then_delta_v",
            "seed": args.seed,
            "episodes": args.episodes,
            "trials": trials,
            "selected_candidate": (
                min(trials, key=lambda item: item["lexicographic_score"])["candidate"]
                if trials
                else None
            ),
            "selected_full_evaluation": selected_result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                output,
                indent=2,
                default=lambda value: value.tolist() if hasattr(value, "tolist") else value,
            )
            + "\n",
            encoding="utf-8",
        )

    for index in range(args.start_candidate, args.max_candidates):
        parameters = CANDIDATES[index]
        result = evaluate(
            config,
            episodes=args.episodes,
            seed=args.seed,
            environment_config=environment,
            external_guidance="piecewise",
            piecewise_parameters=parameters,
            progress=False,
        )
        summary = _summary(result)
        score = _score(summary, args.episodes)
        trials.append(
            {
                "candidate": index,
                "parameters": asdict(parameters),
                "lexicographic_score": score,
                "summary": summary,
            }
        )
        print(
            f"candidate={index} completion={summary['rates']['completed']:.3f} "
            f"safe={summary['rates']['constraint_success']:.3f} score={score:.3f}",
            flush=True,
        )
        if score < best_score:
            best_score = score
            best_result = result
        write_output(best_result)

    assert best_result is not None
    write_output(best_result)
    print(f"S5 result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
