"""Exact base-policy rollout oracle for the V3 feasibility gate (P1/P2).

The simulator, the target tumble and the CLARABEL MPC are deterministic, so a
rollout from a recorded state under a fixed candidate gives that candidate's
exact return. This probe replays a recorded V2 episode's *applied* task states
for K decisions and then hands control back to Pure MPC for the rest of the
episode:

    B     all decisions rejected                  (V_b(s_0), the baseline row)
    K<n>  first n decisions as recorded, then Pure MPC
    Kinf  every decision as recorded             (replay fidelity check)

Handoff calls ``controller.reset()`` at the switch, i.e. "Pure MPC started
fresh from this state". Without it the lower layer reads the reference jump
as a reference velocity (waypoint difference / 2 s), which is an interface
side effect and not Pure MPC's own behaviour.

No model is loaded and nothing is trained. See
``docs/V3_P1_P2_PREREGISTRATION_20260924.md`` for the reading rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import precapture_adaptive_capture_environment_config

DECISION_GAMMA = 0.99


def _make_env() -> PrecaptureHybridEnv:
    return PrecaptureHybridEnv(
        environment_config=precapture_adaptive_capture_environment_config(),
        hybrid_config=PrecaptureHybridConfig(
            horizon_steps=35,
            waypoint_parametrization="task_state_v2",
            runtime_diagnostics=False,
            include_target_phase_and_time_observation=True,
            include_execution_feedback_observation=True,
            include_staging_direction_observation=True,
        ),
    )


def rollout(record: dict, deviate_decisions: int | None) -> dict:
    """``deviate_decisions``: 0 = baseline, n = K-step option, None = full replay."""

    applied = record["task_state_applied"]
    limit = len(applied) if deviate_decisions is None else min(deviate_decisions, len(applied))
    env = _make_env()
    started = perf_counter()
    try:
        env.reset(seed=int(record["seed"]))
        rewards: list[float] = []
        info: dict = {}
        handed_back = False
        replay_mismatch = 0.0
        decision = 0
        while True:
            if decision < limit:
                rho, commitment = applied[decision]
                action = env.action_for_task_state(float(rho), float(commitment))
                _, reward, terminated, truncated, info = env.step_with_proposal(
                    action, proposal_accepted=True
                )
                replay_mismatch = max(
                    replay_mismatch,
                    abs(float(env._task_progress_m) - float(rho)),
                    abs(float(env._task_commitment) - float(commitment)),
                )
            else:
                if not handed_back and decision > 0:
                    env.controller.reset()
                handed_back = True
                _, reward, terminated, truncated, info = env.step_with_proposal(
                    np.zeros(2), proposal_accepted=False
                )
            rewards.append(float(reward))
            decision += 1
            if terminated or truncated:
                break
        discounts = DECISION_GAMMA ** np.arange(len(rewards))
        return {
            "seed": int(record["seed"]),
            "deviate_decisions": deviate_decisions,
            "decisions": len(rewards),
            "return_discounted": float(np.dot(discounts, rewards)),
            "return_undiscounted": float(np.sum(rewards)),
            "completed": bool(info.get("completed", False)),
            "distance_failure": bool(info.get("distance_failure", False)),
            "time_failure": bool(info.get("time_failure", False)),
            "survival_s": float(env.env.time_seconds),
            "max_replay_mismatch": replay_mismatch,
            "wall_clock_s": perf_counter() - started,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--deviate",
        required=True,
        help="0 for the baseline, an integer K, or 'inf' for the full recorded replay",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {r["seed"]: r for r in json.loads(args.records.read_text())["records"]}
    deviate = None if args.deviate == "inf" else int(args.deviate)
    result = rollout(records[args.seed], deviate)
    result["source"] = str(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
