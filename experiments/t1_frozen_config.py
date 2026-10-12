"""Print the frozen M0/M1 round-1 configuration *from the code* (not from a document).

    python -B -m experiments.t1_frozen_config --output frozen_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path


def frozen() -> dict:
    from env.t1_env import T1EnvConfig, T1Reward, t1_mpc_config
    from env.t1_gstar import GSTAR_VERSION
    from env.t1_gym import OBSERVATION_NAMES, T1GymConfig
    from experiments import t1_dev_eval, t1_learning_readout
    from train.train_t1 import COUNTERFACTUAL, M1_LAMBDA, M1_WARMUP_FRACTION, T1_SAC, model_kwargs
    from train.train_hybrid import _code_provenance

    env = T1EnvConfig()
    mpc = t1_mpc_config(env)
    openings = Path("experiments/t1_dev_openings_295000.json")
    kwargs = model_kwargs(T1_SAC)
    return {
        **_code_provenance(),
        "gstar_version": GSTAR_VERSION,
        "task": {"max_time_s": env.max_time_s, "dt_s": env.dt_s, "decision_steps": env.decision_steps,
                 "max_force_per_axis_n": env.max_force_per_axis_n, "max_torque_per_axis_nm": env.max_torque_per_axis_nm,
                 "max_distance_m": env.task.max_distance_m},
        "reward": asdict(T1Reward()),
        "mpc": {"horizon_steps": mpc.horizon_steps, "solver": mpc.solver, "outer_iterations": mpc.outer_iterations},
        "gym": asdict(T1GymConfig()) | {"observation": list(OBSERVATION_NAMES), "action": "a0 -> tau (relative with a live plan, keep band +-3 s; absolute 0..160 s without), a1 -> kappa 0.3..1.0"},
        "sac": asdict(T1_SAC) | {"replay_buffer_kwargs": kwargs["replay_buffer_kwargs"]},
        "m1": {"lambda": M1_LAMBDA, "warmup_fraction": M1_WARMUP_FRACTION, "warmup_decisions_at_20k": int(M1_WARMUP_FRACTION * 20_000),
               "loss_normalisation": "full minibatch B", "gate": asdict(COUNTERFACTUAL)},
        "round": {"methods": ["m0", "m1"], "seeds": [0, 1, 2], "decisions_per_run": 20_000,
                  "training_openings": "1_000_000 * (seed + 1) + episode index",
                  "checkpoints": list(t1_dev_eval.CHECKPOINTS)},
        "development_evaluation": {
            "openings_file": str(openings), "sha256": hashlib.sha256(openings.read_bytes()).hexdigest(),
            "seeds": json.loads(openings.read_text())["seeds"],
            "primary": "mean discounted return (gamma 0.999 per decision), deterministic policy, plain env, serial",
            "auc": "trapezoid over 5k/10k/15k/20k / 15000", "rule_level": t1_learning_readout.RULE_LEVEL,
            "rule_level_evidence": "evidence/T1_dev_rule_295000",
        },
        "isolation": {"training": ">= 1_000_000", "development_evaluation": "295000-295054",
                      "R0": "290000-290003 (not used)", "R1": "291000-291066 (not used)",
                      "formal_test": ">= 300000, < 1_000_000: not accessed in this round"},
        "failure_kinds_recorded": ["collision", "operation_sphere_cone", "fov", "radial_envelope", "total_speed",
                                   "axial_envelope", "distance", "timeout", "numerical_failure"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(frozen(), indent=1, default=str)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
