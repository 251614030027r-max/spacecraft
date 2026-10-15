"""The one canonical configuration of the V3e mainline.

Every V3e number so far (training, the formal learned-only / Pure MPC /
nominal rows, M2/M6/M5) was produced by ``accelerated_training_configs`` with
the flags below, recorded in the three run manifests
(``eval/v3e/manifests/``). This module names that combination once, so new
work (the stage-B handoff scan and whatever follows) builds the mainline from
one entry point instead of re-assembling flags.

It only *names* the existing path: nothing here changes a default, and no
historical parametrisation is removed. ``tests/test_v3_mainline.py`` pins it
bitwise against the manifests and against the formal evaluator's Pure MPC and
nominal paths.

Known, harmless difference, kept visible on purpose: the formal evaluator
(``experiments/evaluate_hybrid_policy.py``) builds its hybrid config without
``decision_discount_factor``, so it uses the dataclass default instead of the
SAC gamma. That field only scales the potential-shaping term of the reward; it
does not enter the dynamics, the MPC or the reference, so trajectories agree
bitwise and only the shaped reward differs. Outcome comparisons across the two
paths are therefore made on outcomes and states, never on shaped reward.
"""

from __future__ import annotations

from typing import Any

from env.hybrid_env import PrecaptureHybridConfig, PrecaptureHybridEnv
from env.se3_rendezvous_env import SE3RendezvousConfig
from train.train_hybrid import accelerated_training_configs

#: Flags of the v3e runs (manifest ``evaluation_flags``: ``--horizon 35
#: --parametrization task_state_v3 --phase-time-observation
#: --execution-feedback --adaptive-task``; ``monotone_commit`` true,
#: ``baseline_anchored_residual`` false).
MAINLINE_V3E: dict[str, Any] = {
    "horizon_steps": 35,
    "waypoint_parametrization": "task_state_v3",
    "execution_feedback": True,
    "monotone_commit": True,
    "baseline_anchored_residual": False,
    "opportunity_task": False,
    "adaptive_task": True,
}

#: The evaluator flags that rebuild the same environment for the formal rows.
MAINLINE_V3E_EVALUATION_FLAGS = (
    "--horizon 35 --parametrization task_state_v3 --phase-time-observation "
    "--execution-feedback --adaptive-task"
)


def mainline_v3e_configs() -> tuple[SE3RendezvousConfig, PrecaptureHybridConfig]:
    return accelerated_training_configs(**MAINLINE_V3E)


def mainline_v3e_env() -> PrecaptureHybridEnv:
    environment_config, hybrid_config = mainline_v3e_configs()
    return PrecaptureHybridEnv(environment_config, hybrid_config)
