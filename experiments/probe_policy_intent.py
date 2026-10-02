"""What does the trained upper layer *intend*, and what does its critic think?

The two-model interim evaluation measured 0.1-0.5% effective intervention, which
has two very different explanations: either the policy wants to stage and the
deployment gate plus the commit ratchet suppress it, or the policy never intends
to stage in the first place. Telling those apart decides whether the fix is the
gate's semantics or the action parametrisation, and they need work an order of
magnitude apart.

This probe answers it without a single rollout. Under the residual interface the
commit channel maps ``a_commit = clip(1 + 2 * r)`` and the blend is
``0.5 * (a_commit + 1)``, so at the first decision -- where ``_commit_blend`` is
still 0 and the ratchet cannot yet bind -- the reference differs from the
fixed-setpoint Pure MPC reference **if and only if the commit residual is
negative**. So the whole question at the decisive moment is the sign of one
number, and the gate's behaviour is the sign of one advantage. Both are read
from a reset state in microseconds.

Reads, for each evaluation seed: the deterministic commit and radius residuals,
the minimum-twin Q of the policy's own action, the same Q of the zero residual
(the nominal the gate falls back to), and their difference -- the exact quantity
`--deployment-gate` arbitrates on.

Truth geometry adjudicates nothing here and this probe measures no performance:
it reports what the policy and critic *say*, never whether they are right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.phase2_env import (
    precapture_adaptive_capture_environment_config,
    precapture_planning_environment_config,
)
from experiments.evaluate_hybrid_policy import critic_min_q


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--seed", type=int, default=262000)
    parser.add_argument("--horizon", type=int, default=35)
    parser.add_argument("--adaptive-task", action="store_true")
    parser.add_argument(
        "--gate-advantage-margin",
        type=float,
        default=0.0,
        help="Same margin the deployment gate uses, so the fallback fraction "
        "reported here is the one evaluation would produce at step zero.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from stable_baselines3 import SAC

    policy = SAC.load(args.model, device="cpu")
    environment_config = (
        precapture_adaptive_capture_environment_config()
        if args.adaptive_task
        else precapture_planning_environment_config()
    )
    hybrid_config = PrecaptureHybridConfig(
        horizon_steps=args.horizon,
        waypoint_parametrization="arrival_condition",
        runtime_diagnostics=False,
        include_target_phase_and_time_observation=True,
        include_execution_feedback_observation=True,
        baseline_anchored_residual=True,
        include_staging_direction_observation=args.adaptive_task,
    )

    records: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = PrecaptureHybridEnv(
            environment_config=environment_config, hybrid_config=hybrid_config
        )
        if episode == 0:
            expected = int(np.prod(policy.observation_space.shape))
            actual = int(np.prod(env.observation_space.shape))
            if expected != actual:
                raise ValueError(
                    f"the checkpoint expects a {expected}D observation and this "
                    f"configuration builds a {actual}D one; match the run's "
                    "manifest rather than guessing flags"
                )
        observation, _ = env.reset(seed=seed)
        action, _ = policy.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        nominal = np.zeros_like(action)
        q_policy = critic_min_q(policy, observation, action)
        q_nominal = critic_min_q(policy, observation, nominal)
        # Algebra of the residual map at the first decision, where the ratchet
        # cannot yet bind: a_commit = clip(1 + 2r), blend = 0.5 * (a_commit + 1).
        commit_residual = float(np.clip(action[0], -1.0, 1.0))
        a_commit = float(np.clip(1.0 + 2.0 * commit_residual, -1.0, 1.0))
        blend = 0.5 * (a_commit + 1.0)
        advantage = q_policy - q_nominal
        # Cross-check the algebra by measurement rather than trusting it: ask
        # the environment itself what this action commands, and confirm the
        # reference differs from the fixed setpoint exactly when the commit
        # residual is negative. Free here -- it is one mapping call on a fresh
        # reset, no rollout -- and it turns the claim this probe rests on into
        # something the artifact records.
        desired = np.asarray(
            env.environment_config.precapture_task.desired_position,
            dtype=np.float64,
        )
        reference = np.asarray(env.waypoint_from_action(action), dtype=np.float64)
        reference_differs = not bool(np.allclose(reference, desired))
        if reference_differs != (commit_residual < 0.0):
            raise AssertionError(
                "the residual map no longer matches the algebra this probe "
                f"assumes: residual {commit_residual:+.6f} but reference "
                f"{'differs' if reference_differs else 'equals'} the setpoint"
            )
        records.append(
            {
                "episode": episode,
                "seed": seed,
                "commit_residual": commit_residual,
                "radius_residual": float(np.clip(action[1], -1.0, 1.0)),
                "mapped_a_commit": a_commit,
                "first_decision_blend": blend,
                # The whole question at the decisive moment, stated exactly.
                "intends_to_stage": bool(commit_residual < 0.0),
                "reference_differs_measured": reference_differs,
                "first_decision_reference_target_frame": [
                    float(v) for v in reference
                ],
                "q_policy_action": q_policy,
                "q_zero_residual": q_nominal,
                "critic_advantage": advantage,
                "gate_would_deviate": bool(advantage > args.gate_advantage_margin),
                # What actually reaches the MPC at step zero once the gate has
                # had its say: staging survives only if the policy asked for it
                # AND the gate let it through.
                "effective_staging_at_step_zero": bool(
                    commit_residual < 0.0
                    and advantage > args.gate_advantage_margin
                ),
            }
        )

    commit = np.array([r["commit_residual"] for r in records])
    advantages = np.array([r["critic_advantage"] for r in records])
    summary = {
        "model": str(args.model),
        "episodes": args.episodes,
        "seed_block_start": args.seed,
        "gate_advantage_margin": args.gate_advantage_margin,
        "commit_residual_median": float(np.median(commit)),
        "commit_residual_mean": float(np.mean(commit)),
        "commit_residual_min": float(np.min(commit)),
        "commit_residual_max": float(np.max(commit)),
        "intends_to_stage_count": int(sum(r["intends_to_stage"] for r in records)),
        "gate_would_deviate_count": int(
            sum(r["gate_would_deviate"] for r in records)
        ),
        "effective_staging_at_step_zero_count": int(
            sum(r["effective_staging_at_step_zero"] for r in records)
        ),
        "critic_advantage_median": float(np.median(advantages)),
        "critic_advantage_min": float(np.min(advantages)),
        "critic_advantage_max": float(np.max(advantages)),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"model {args.model}")
    print(f"commit residual  median {summary['commit_residual_median']:+.4f}  "
          f"range [{summary['commit_residual_min']:+.4f}, "
          f"{summary['commit_residual_max']:+.4f}]")
    print(f"intends to stage (residual < 0)      "
          f"{summary['intends_to_stage_count']}/{args.episodes}")
    print(f"gate would deviate (advantage > {args.gate_advantage_margin:g})  "
          f"{summary['gate_would_deviate_count']}/{args.episodes}")
    print(f"staging surviving the gate at step 0 "
          f"{summary['effective_staging_at_step_zero_count']}/{args.episodes}")
    print(f"critic advantage median {summary['critic_advantage_median']:+.4f}  "
          f"range [{summary['critic_advantage_min']:+.4f}, "
          f"{summary['critic_advantage_max']:+.4f}]")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
