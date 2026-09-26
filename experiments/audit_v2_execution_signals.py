"""Offline audit of MPC execution signals for the V2 task-state interface.

This is deliberately not a controller or training experiment.  It samples 24
states from eight existing nominal seed trajectories, probes nine synthetic
task references at each state, and asks whether any already-computed MPC signal
changes monotonically and above repeat-solve noise.  The preregistered rule is
encoded in :func:`summarize_signal` and written verbatim to the JSON artifact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from controllers.mpc import MPCController
from controllers.mpc.prediction import relative_to_vector
from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv, _slerp
from env.phase2_env import precapture_adaptive_capture_environment_config
from env.scenarios import chaser_parameters, target_parameters
from controllers.mpc.prediction import LocalRelativePredictionModel, RelativePredictionModel
from dynamics.gravity import GravityOptions
from dynamics.integrator import RK45Settings


SEEDS = tuple(range(262000, 262048, 6))
DECISION_INDICES = (0, 15, 30)
REPEATS = 3
MONOTONIC_FRACTION = 0.70
NUMERICAL_FLOOR = 1.0e-9

# The audit base is intentionally inside the legal, non-saturated task box.
# rho follows the sampled vehicle range but is clipped away from both bounds;
# c=0.5 leaves room to perturb commitment in both directions.
RHO_BASE_BOUNDS_M = (6.0, 20.0)
RHO_SMALL_M = 0.5
RHO_LARGE_M = 2.0
C_SMALL = 0.10
C_LARGE = 0.30

CHEAP_SIGNALS = (
    "maximum_successful_slack",
    "total_successful_slack",
    "objective",
    "actuator_usage",
)
ORACLE_SIGNAL = "predicted_minimum_margin"
SIGNALS = CHEAP_SIGNALS + (ORACLE_SIGNAL,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/adp/v2_execution_signal_audit.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/V2_EXECUTION_SIGNAL_AUDIT_20260922.md"),
    )
    return parser.parse_args()


def task_waypoint(
    hold_direction_target: np.ndarray,
    commit_direction: np.ndarray,
    rho_m: float,
    commitment: float,
) -> np.ndarray:
    direction = _slerp(
        np.asarray(hold_direction_target, dtype=np.float64),
        np.asarray(commit_direction, dtype=np.float64),
        float(np.clip(commitment, 0.0, 1.0)),
    )
    return float(rho_m) * direction


def candidate_tasks(rho_m: float, commitment: float = 0.5) -> dict[str, tuple[float, float]]:
    return {
        "rho_backward_large": (rho_m + RHO_LARGE_M, commitment),
        "rho_backward_small": (rho_m + RHO_SMALL_M, commitment),
        "unchanged": (rho_m, commitment),
        "rho_forward_small": (rho_m - RHO_SMALL_M, commitment),
        "rho_forward_large": (rho_m - RHO_LARGE_M, commitment),
        "commit_decrease_large": (rho_m, commitment - C_LARGE),
        "commit_decrease_small": (rho_m, commitment - C_SMALL),
        "commit_increase_small": (rho_m, commitment + C_SMALL),
        "commit_increase_large": (rho_m, commitment + C_LARGE),
    }


def monotonic_triplet(values: list[float], tolerance: float) -> bool:
    a, b, c = values
    return bool(
        (a <= b + tolerance and b <= c + tolerance)
        or (a + tolerance >= b and b + tolerance >= c)
    )


def summarize_signal(states: list[dict[str, Any]], signal: str) -> dict[str, Any]:
    informative: list[bool] = []
    monotonic: list[bool] = []
    distinguishable: list[bool] = []
    directions = {
        "rho_forward": ("unchanged", "rho_forward_small", "rho_forward_large"),
        "rho_backward": ("unchanged", "rho_backward_small", "rho_backward_large"),
        "commit_increase": ("unchanged", "commit_increase_small", "commit_increase_large"),
        "commit_decrease": ("unchanged", "commit_decrease_small", "commit_decrease_large"),
    }
    per_state = []
    for state in states:
        means = {
            name: float(np.mean(candidate["signals"][signal]))
            for name, candidate in state["candidates"].items()
        }
        repeat_noise = max(
            float(np.ptp(candidate["signals"][signal]))
            for candidate in state["candidates"].values()
        )
        tolerance = max(NUMERICAL_FLOOR, repeat_noise)
        monotonic_directions = [
            name
            for name, keys in directions.items()
            if monotonic_triplet([means[key] for key in keys], tolerance)
            and max(means[key] for key in keys) - min(means[key] for key in keys)
            > tolerance
        ]
        candidate_span = max(means.values()) - min(means.values())
        is_monotonic = bool(monotonic_directions)
        is_distinguishable = candidate_span > tolerance
        monotonic.append(is_monotonic)
        distinguishable.append(is_distinguishable)
        informative.append(is_monotonic and is_distinguishable)
        per_state.append(
            {
                "seed": state["seed"],
                "decision_index": state["decision_index"],
                "monotonic_directions": monotonic_directions,
                "candidate_span": candidate_span,
                "repeat_noise": repeat_noise,
                "informative": informative[-1],
            }
        )
    fraction = float(np.mean(informative))
    return {
        "informative_states": int(sum(informative)),
        "state_count": len(states),
        "informative_fraction": fraction,
        "monotonic_state_fraction": float(np.mean(monotonic)),
        "distinguishable_state_fraction": float(np.mean(distinguishable)),
        "passes_preregistered_rule": fraction >= MONOTONIC_FRACTION,
        "per_state": per_state,
    }


def make_audit_controller(env: PrecaptureHybridEnv) -> MPCController:
    config = replace(env.mpc_config, runtime_diagnostics=True)
    reference = RelativePredictionModel(
        target_parameters=target_parameters(),
        chaser_parameters=chaser_parameters(),
        dt_s=env.environment_config.dt_s,
        gravity_options=GravityOptions(include_j2=env.environment_config.include_j2),
        solver_settings=RK45Settings(
            rtol=env.environment_config.solver_rtol,
            atol=env.environment_config.solver_atol,
            max_step=env.environment_config.dt_s,
        ),
    )
    return MPCController(
        config,
        LocalRelativePredictionModel(reference.chaser_parameters, env.environment_config.dt_s),
        reference,
    )


def probe_state(
    env: PrecaptureHybridEnv,
    controller: MPCController,
    *,
    seed: int,
    decision_index: int,
    hold_inertial: np.ndarray,
) -> dict[str, Any]:
    assert env.env.relative is not None and env.env.target_state is not None
    state = relative_to_vector(env.env.relative)
    target = env.env.target_state
    rotation = np.asarray(target.rotation, dtype=np.float64)
    position = env._current_position()
    sampled_range = float(np.linalg.norm(position))
    rho_base = float(np.clip(sampled_range, *RHO_BASE_BOUNDS_M))
    commitment_base = 0.5
    hold_direction_target = rotation.T @ hold_inertial
    desired = np.asarray(env.environment_config.precapture_task.desired_position, dtype=np.float64)
    commit_direction = desired / np.linalg.norm(desired)
    candidates: dict[str, Any] = {}
    for name, (rho_m, commitment) in candidate_tasks(rho_base, commitment_base).items():
        waypoint = task_waypoint(
            hold_direction_target, commit_direction, rho_m, commitment
        )
        signals = {key: [] for key in SIGNALS}
        statuses: list[str] = []
        fallbacks: list[bool] = []
        for _ in range(REPEATS):
            controller.reset()
            wrench, diagnostics = controller.command(
                state,
                target_state=target,
                time_seconds=env.env.time_seconds,
                terminal_latched=bool(env._last_info["terminal_region_active"]),
                external_reference=waypoint,
            )
            statuses.append(diagnostics.status)
            fallbacks.append(bool(diagnostics.used_zero_fallback))
            # Slack is meaningful only on a successful solve.
            if diagnostics.used_zero_fallback:
                successful_max = float("nan")
                successful_total = float("nan")
            else:
                successful_max = float(diagnostics.maximum_slack)
                successful_total = float(diagnostics.total_slack)
            signals["maximum_successful_slack"].append(successful_max)
            signals["total_successful_slack"].append(successful_total)
            signals["objective"].append(float(diagnostics.objective))
            signals["actuator_usage"].append(
                max(
                    float(np.max(np.abs(wrench[:3])))
                    / env.environment_config.max_torque_per_axis_nm,
                    float(np.max(np.abs(wrench[3:])))
                    / env.environment_config.max_force_per_axis_n,
                )
            )
            signals["predicted_minimum_margin"].append(
                float(diagnostics.predicted_minimum_margin)
            )
        candidates[name] = {
            "rho_m": rho_m,
            "commitment": commitment,
            "waypoint_target_frame_m": waypoint.tolist(),
            "statuses": statuses,
            "fallbacks": fallbacks,
            "signals": signals,
        }
    return {
        "seed": seed,
        "decision_index": decision_index,
        "time_s": float(env.env.time_seconds),
        "sampled_range_m": sampled_range,
        "rho_base_m": rho_base,
        "commitment_base": commitment_base,
        "target_phase_rotation": rotation.tolist(),
        "remaining_time_fraction": float(
            (env.environment_config.max_time_s - env.env.time_seconds)
            / env.environment_config.max_time_s
        ),
        "candidates": candidates,
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# V2 离线执行信号审计",
        "",
        "本审计不改变任务、训练、reward 或 MPC，只在已有 nominal 种子轨迹的代表状态上读取单步求解信号。",
        "",
        "## 预注册协议",
        "",
        f"- 状态：种子 `{list(SEEDS)}`，每个种子决策点 `{list(DECISION_INDICES)}`，共 24 点。",
        "- 候选：距离前进/后撤各小步与大步，承诺增加/减少各小步与大步，加不动基准，共 9 个。",
        f"- 重复：每个候选独立 reset 后求解 {REPEATS} 次。",
        f"- 判据：至少 {MONOTONIC_FRACTION:.0%} 状态上至少一个方向随幅度单调，且候选跨度大于同状态重复波动。",
        "",
        "## 结果",
        "",
        "| 信号 | 类型 | 有信息状态 | 比例 | 通过 |",
        "|---|---|---:|---:|---|",
    ]
    for signal, summary in result["signal_summary"].items():
        kind = "离线 oracle" if signal == ORACLE_SIGNAL else "便宜"
        lines.append(
            f"| `{signal}` | {kind} | {summary['informative_states']}/{summary['state_count']} | "
            f"{summary['informative_fraction']:.1%} | {'是' if summary['passes_preregistered_rule'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 三选一结论",
            "",
            result["conclusion"],
            "",
            "V2 第一版仍使用固定不对称限速；本审计不授权接入 governor。完整逐状态数据见 `eval/adp/v2_execution_signal_audit.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    env = PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="arrival_condition",
            runtime_diagnostics=False,
        ),
    )
    controller = make_audit_controller(env)
    states: list[dict[str, Any]] = []
    desired = np.asarray(env.environment_config.precapture_task.desired_position)
    try:
        for seed in SEEDS:
            env.reset(seed=seed)
            initial_position = env._current_position()
            assert env.env.target_state is not None
            hold_inertial = env.env.target_state.rotation @ (
                initial_position / np.linalg.norm(initial_position)
            )
            for decision_index in range(max(DECISION_INDICES) + 1):
                if decision_index in DECISION_INDICES:
                    states.append(
                        probe_state(
                            env,
                            controller,
                            seed=seed,
                            decision_index=decision_index,
                            hold_inertial=hold_inertial,
                        )
                    )
                action = env.action_for_waypoint(desired)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    if decision_index < max(DECISION_INDICES):
                        raise RuntimeError(
                            f"seed {seed} ended before decision {max(DECISION_INDICES)}"
                        )
                    break
    finally:
        env.close()

    signal_summary = {signal: summarize_signal(states, signal) for signal in SIGNALS}
    cheap_passes = [
        signal
        for signal in CHEAP_SIGNALS
        if signal_summary[signal]["passes_preregistered_rule"]
    ]
    oracle_passes = signal_summary[ORACLE_SIGNAL]["passes_preregistered_rule"]
    if cheap_passes:
        conclusion = (
            "有便宜信号满足预注册判据："
            + "、".join(f"`{name}`" for name in cheap_passes)
            + "。governor 有离线信号地基，但 V2 第一版仍不接入。"
        )
        category = "cheap_signal_passes"
    elif oracle_passes:
        conclusion = (
            "只有 `predicted_minimum_margin` 满足预注册判据；唯一有用信号需要完整时域推演，"
            "当前在线路径用不起。V2 第一版采用固定限速。"
        )
        category = "oracle_only"
    else:
        conclusion = "没有信号满足预注册判据，governor 当前没有信号地基；V2 第一版采用固定限速。"
        category = "no_signal_passes"

    result = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline_signal_audit_only_no_training_no_task_or_reward_change",
        "sampling": {
            "seeds": list(SEEDS),
            "decision_indices": list(DECISION_INDICES),
            "state_count": len(states),
            "candidate_count_per_state": 9,
            "repeats_per_candidate": REPEATS,
            "selection_rule": (
                "eight evenly spaced seeds from 262000-262047; decisions 0,15,30 "
                "cover initial range, target phase, and remaining time"
            ),
        },
        "candidate_definition": {
            "rho_base_bounds_m": list(RHO_BASE_BOUNDS_M),
            "rho_small_m": RHO_SMALL_M,
            "rho_large_m": RHO_LARGE_M,
            "commitment_base": 0.5,
            "commitment_small": C_SMALL,
            "commitment_large": C_LARGE,
        },
        "preregistered_rule": {
            "minimum_informative_state_fraction": MONOTONIC_FRACTION,
            "monotonicity": "at least one perturbation direction has a monotonic baseline-small-large triplet",
            "distinguishability": "candidate mean span exceeds maximum within-candidate repeat range and 1e-9",
        },
        "signal_summary": signal_summary,
        "conclusion_category": category,
        "conclusion": conclusion,
        "states": states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    args.report.write_text(render_report(result), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "conclusion": conclusion}, ensure_ascii=False))


if __name__ == "__main__":
    main()
