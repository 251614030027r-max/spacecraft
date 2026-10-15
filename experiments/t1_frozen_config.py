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
    from env.t1_gstar import GSTAR_VERSION, GStarRendezvous
    from env.t1_gym import GYM_INTERFACE_VERSION, OBSERVATION_NAMES, T1GymConfig
    from experiments import t1_dev_eval, t1_learning_readout
    from train.train_t1 import COUNTERFACTUAL, M1_LAMBDA, M1_WARMUP_DECISIONS, T1_SAC, model_kwargs
    from train.train_hybrid import _code_provenance

    env = T1EnvConfig()
    mpc = t1_mpc_config(env)
    openings = Path("experiments/t1_dev_openings_295000.json")
    kwargs = model_kwargs(T1_SAC)
    return {
        **_code_provenance(),
        "gstar_version": GSTAR_VERSION,
        "gstar_deadline_margin_s": GStarRendezvous.deadline_margin_s,
        "task": {"max_time_s": env.max_time_s, "dt_s": env.dt_s, "decision_steps": env.decision_steps,
                 "max_force_per_axis_n": env.max_force_per_axis_n, "max_torque_per_axis_nm": env.max_torque_per_axis_nm,
                 "max_distance_m": env.task.max_distance_m},
        "reward": asdict(T1Reward()),
        "mpc": {"horizon_steps": mpc.horizon_steps, "solver": mpc.solver, "outer_iterations": mpc.outer_iterations},
        "gym": asdict(T1GymConfig()) | {"observation": list(OBSERVATION_NAMES), "action": "gym interface v4: a[2] -> inward entry speed of the next new G* plan, 0.06 + 0.04 a[2] m/s (kept plans keep theirs); a0/a1 as interface v2; observation = v2 + 2 tracking errors (v3) + live plan entry speed (v4): a0 -> tau (with a live plan: keep for |a0| <= keep_band, else shift linearly to +-tau_shift_s; without: absolute 0..160 s, <= 2 s earliest), a1 -> kappa 0.3..1.0"},
        "sac": asdict(T1_SAC) | {"replay_buffer_kwargs": kwargs["replay_buffer_kwargs"]},
        "gym_interface_version": GYM_INTERFACE_VERSION,
        "m1": {"lambda": M1_LAMBDA, "warmup_decisions": M1_WARMUP_DECISIONS,
               "loss_normalisation": "full minibatch B", "gate": asdict(COUNTERFACTUAL)},
        "critic_fixes": {"mc_delta": 10.0, "mc_window": 10_000, "lambda_mc_lambda_eq": "frozen by experiments.t1_fix_calibration (revision 2: lam_eq 0.1 fixed, lam_mc in {0.3, 0.1, 0.03} with floor 0.03) on the 2k M0 calibration run; recorded in each run manifest (critic_fixes)",
                         "shared_class": "train.t1_m1.M1SAC for M0 (lam_cf = 0, no counterfactual env) and M2"},
        "round": {"stage": "entry-speed authority + learnability fixes, first training pair (plan 2026-10-15): M0 and M2, seed 0",
                  "decisions": "40000 each, no performance stop (technical stops only)",
                  "checkpoints": "0 (untrained policy), 2000, then every 5000 decisions; dev evaluation at 0, 2000, 5000, 10000, ..., 40000", "n_envs": 6, "updates_per_transition": T1_SAC.updates_per_transition,
                  "training_openings": "1_000_000 * (seed + 1) + env_index * 100000 + episode index"},
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
