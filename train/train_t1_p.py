"""Train one arm of the T1-P 2 x 2 (research plan T1-P sec. 7) from zero.

    R1     review-sample target one step,               no stage counterfactual
    R3     review-sample target = closed 12 s / event interval
    D1     one step                                     + 12 s MPC stage counterfactual
    TCPF   closed interval                              + 12 s MPC stage counterfactual

Common to the four: the T1-P task and interface (``env.t1_p_review``: 3-D 12 s stage goals, local
G*, entry bridge, T1 descent; fuel weight 20; the training reward with the potential of
``env.t1_p_stage``, evaluation with the raw reward), the shared MPC, 256 x 3, fixed entropy
0.005 x review gap, ``uniform_eligibility``, trainer B (one critic update per real transition,
one actor update per 3 real transitions, cosine 1e-4 -> 1e-5), lambda_eq = 0, lambda_mc = 0, the
budget in real 2 s steps and the checkpoints. The counterfactual arms use ``env.t1_p_cf`` and the
5 % / 10 % gradient-share weight. Training openings ``1_000_000 * (seed + 1) + episode``. No resume:
an interrupted run is restarted from zero.

    python -B -m train.train_t1_p --arm tcpf --steps 40000 --seed 0 --run-name tcpf_s0 \
        --checkpoint-freq 10000 --extra-checkpoints 2000,5000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from env.t1_gym import T1GymConfig
from env.t1_n_task import T1N_FUEL_COST, t1n_env_config
from env.t1_p_cf import T1PCFConfig, T1PReviewCFEnv
from env.t1_p_review import INTERFACE_VERSION, T1P_OBSERVATION_NAMES, T1PReviewEnv
from env.t1_p_stage import (
    BRIDGE_ANGLE_DEG, BRIDGE_RANGE_M, BRIDGE_TAUS_S, D_MAX_FRACTION, MOUTH_LOOKAHEAD_S, PLAN_FORCE_N, POTENTIAL_CAP_M,
    STAGE_NODES, STAGE_S, T1P_VERSION,
)
from train.t1_p_sac import T1PSAC
from train.t1_tcpf import CosineLearningRate
from train.train_hybrid import _code_provenance
from train.train_t1 import T1_SAC, Checkpoint, EpisodeLog, _environment_record, model_kwargs

ARMS: dict[str, dict[str, Any]] = {
    "r1": {"interval_bellman": False, "counterfactual": False},
    "r3": {"interval_bellman": True, "counterfactual": False},
    "d1": {"interval_bellman": False, "counterfactual": True},
    "tcpf": {"interval_bellman": True, "counterfactual": True},
}
TRAINER = {"name": "B", "updates_per_transition": 1, "actor_every_transitions": 3,
           "learning_rate": "cosine 1e-4 -> 1e-5 over real transitions"}
LAMBDA_MC = 0.0
LAMBDA_EQ = 0.0
DELTA_WARMUP = 3000
DELTA_TARGET_SHARE = 0.05
DELTA_CAP_SHARE = 0.10


def make_env(arm: str, gym_config: T1GymConfig):
    if ARMS[arm]["counterfactual"]:
        return T1PReviewCFEnv(gym_config, t1n_env_config(T1N_FUEL_COST), cf=T1PCFConfig())
    return T1PReviewEnv(gym_config, t1n_env_config(T1N_FUEL_COST))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--steps", type=int, required=True, help="real 2 s steps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs/t1_p_round1"))
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--extra-checkpoints", default="2000,5000")
    parser.add_argument("--audit-every", type=int, default=100)
    parser.add_argument("--learning-starts", type=int, default=None, help="smoke tests only")
    parser.add_argument("--net-arch", default=None, help="smoke tests only, e.g. 16,16")
    parser.add_argument("--delta-warmup", type=int, default=None, help="smoke tests only")
    parser.add_argument("--delta-calibration-steps", type=int, default=500, help="smoke tests only")
    parser.add_argument("--allow-dirty", action="store_true", help="development checks only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    spec = ARMS[args.arm]
    provenance = _code_provenance()
    if provenance["code_dirty"] is not False and not args.allow_dirty:
        raise SystemExit(f"refusing to train from a modified or unknown checkout: {provenance}")
    log_dir = args.log_root / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    (log_dir / "checkpoints").mkdir(parents=True)
    gym_config = T1GymConfig(seed_base=1_000_000 * (args.seed + 1))
    env = make_env(args.arm, gym_config)
    kwargs = model_kwargs(T1_SAC, [int(x) for x in args.net_arch.split(",")] if args.net_arch else None, n_envs=1)
    kwargs["gradient_steps"] = TRAINER["updates_per_transition"]
    kwargs["learning_rate"] = CosineLearningRate(1e-4, 1e-5)
    kwargs["target_entropy"] = -3.0  # inert while ent_coef is fixed
    if args.learning_starts is not None:
        kwargs["learning_starts"] = args.learning_starts
    warmup = DELTA_WARMUP if args.delta_warmup is None else args.delta_warmup
    model = T1PSAC("MlpPolicy", env, interval_bellman=spec["interval_bellman"],
                   actor_every_transitions=TRAINER["actor_every_transitions"],
                   seed=args.seed, device="cpu", verbose=0, mc_lambda=LAMBDA_MC, eq_lambda=LAMBDA_EQ,
                   audit_every=args.audit_every, delta_lambda=0.0, delta_warmup_decisions=warmup,
                   delta_target_share=DELTA_TARGET_SHARE if spec["counterfactual"] else None,
                   delta_cap_share=DELTA_CAP_SHARE, delta_calibration_steps=args.delta_calibration_steps, **kwargs)
    cfg = t1n_env_config(T1N_FUEL_COST)
    manifest = {
        "experiment": "T1-P 2x2 Round 1 (research plan T1-P, 2026-10-18)", "task_version": T1P_VERSION,
        "arm": args.arm, "arm_spec": spec, "trainer": TRAINER,
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": vars(args) | {"log_root": str(args.log_root)},
        "task": {"initial": asdict(cfg.initial), "reward": asdict(cfg.reward), "stage_s": STAGE_S,
                 "stage_nodes": STAGE_NODES, "plan_force_n": PLAN_FORCE_N, "d_max_fraction": D_MAX_FRACTION,
                 "bridge": {"range_m": BRIDGE_RANGE_M, "angle_deg": BRIDGE_ANGLE_DEG, "taus_s": BRIDGE_TAUS_S},
                 "potential": {"lookahead_s": MOUTH_LOOKAHEAD_S, "cap_m": POTENTIAL_CAP_M}},
        "interface": INTERFACE_VERSION, "observation": list(T1P_OBSERVATION_NAMES), "gym": asdict(gym_config),
        "sac": asdict(T1_SAC) | {"target_entropy": -3.0, "learning_starts": kwargs.get("learning_starts")},
        "learner": {"class": "train.t1_p_sac.T1PSAC", "interval_bellman": spec["interval_bellman"],
                    "uniform_eligibility": True, "lambda_mc": LAMBDA_MC, "lambda_eq": LAMBDA_EQ,
                    "counterfactual": spec["counterfactual"], "delta_target_share": DELTA_TARGET_SHARE if spec["counterfactual"] else None,
                    "delta_cap_share": DELTA_CAP_SHARE, "delta_warmup_steps": warmup, "delta_huber": 10.0,
                    "alpha": "0.005 x review gap (2 s steps, 1..6)"},
        **({"counterfactual": asdict(T1PCFConfig())} if spec["counterfactual"] else {}),
        "training_openings": f"{gym_config.seed_base} + episode index", "n_envs": 1,
        "environment": _environment_record(), **provenance,
    }
    manifest["sac"]["policy_kwargs"] = {"net_arch": kwargs["policy_kwargs"]["net_arch"], "activation_fn": "ReLU"}
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    model.learn(total_timesteps=args.steps, callback=[
        EpisodeLog(log_dir / "episodes.jsonl", log_dir / "decisions.jsonl"),
        Checkpoint(args.checkpoint_freq, log_dir / "checkpoints", args.arm, log_dir,
                   tuple(int(x) for x in args.extra_checkpoints.split(",") if x)),
    ])
    if (log_dir / "ABORTED.json").exists():
        raise SystemExit(f"training stopped: {(log_dir / 'ABORTED.json').read_text()}")
    model.save(log_dir / "final")
    for name in ("fix_stats", "delta_stats", "delta_calibration", "tcpf_stats"):
        (log_dir / f"{name}.json").write_text(json.dumps(getattr(model, name)))
    buffer = model.replay_buffer
    (log_dir / "interval_audit.json").write_text(json.dumps({
        "counts": buffer.interval_counts, "lengths": {str(k): v for k, v in sorted(buffer.interval_lengths.items())},
        "cf_counts": buffer.cf_counts, "actor_updates": model.actor_updates, "critic_updates": model._n_updates}))
    (log_dir / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
                                                       "decisions": int(model.num_timesteps)}))
    env.close()


if __name__ == "__main__":
    main()
