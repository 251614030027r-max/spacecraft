"""Train one arm of the T1-N 2 x 2 (Research Plan R2 sec. 7-8) from zero.

    R1     review-sample target one step,   no relative MPC term
    R3     review-sample target = closed interval (real return over the review interval)
    D1     one step                       + paired MPC relative term
    TCPF   closed interval                + paired MPC relative term

Common to the four (frozen with the run order): the T1-N task and interface
(``env.t1_n_review``, fuel weight 20), conditional G*, the shared MPC, the network (256 x 3),
fixed entropy 0.005 x review gap, ``uniform_eligibility``, trainer B (one critic update per real
transition, one actor update per 3 real transitions on the global clock, cosine 1e-4 -> 1e-5),
the keep-band equivalence term (0.1) and no Monte-Carlo label term (0, see the review), the
budget ``--steps`` in real 2 s steps and the checkpoints. The relative-term arms use the paired
rolling shadow (``env.t1_n_cf``, ``--shadow-steps`` frozen by the run order) and the 5 % / 10 %
gradient-share weight of ``train.t1_option_sac``. Training openings
``1_000_000 * (seed + 1) + env_index * 100000 + episode``. No resume: an interrupted run is
restarted from zero.

    python -B -m train.train_t1_n --arm tcpf --steps 40000 --seed 0 --run-name tcpf_s0 --n-envs 4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import torch

from env.t1_gstar import GSTAR_VERSION
from env.t1_gym import T1GymConfig
from env.t1_n_cf import T1NReviewCFEnv, T1NShadowConfig
from env.t1_n_review import INTERFACE_VERSION, T1N_OBSERVATION_NAMES, T1NReviewEnv
from env.t1_n_task import T1N_ENTRY_SPEED_M_S, T1N_FUEL_COST, T1N_KAPPA, T1N_VERSION, t1n_env_config
from train.t1_n_sac import T1NSAC
from train.t1_tcpf import CosineLearningRate
from train.train_hybrid import _code_provenance
from train.train_t1 import T1_SAC, Checkpoint, EpisodeLog, _environment_record, model_kwargs

ARMS: dict[str, dict[str, Any]] = {
    "r1": {"interval_bellman": False, "relative": False},
    "r3": {"interval_bellman": True, "relative": False},
    "d1": {"interval_bellman": False, "relative": True},
    "tcpf": {"interval_bellman": True, "relative": True},
}
TRAINER = {"name": "B", "updates_per_transition": 1, "actor_every_transitions": 3,
           "learning_rate": "cosine 1e-4 -> 1e-5 over real transitions"}
LAMBDA_MC = 0.0
LAMBDA_EQ = 0.1
DELTA_WARMUP = 3000
DELTA_TARGET_SHARE = 0.05
DELTA_CAP_SHARE = 0.10


def make_env(arm: str, gym_config: T1GymConfig, shadow_steps: int):
    if ARMS[arm]["relative"]:
        return T1NReviewCFEnv(gym_config, t1n_env_config(T1N_FUEL_COST),
                              shadow=T1NShadowConfig(horizon_steps=shadow_steps))
    return T1NReviewEnv(gym_config, t1n_env_config(T1N_FUEL_COST))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--steps", type=int, required=True, help="real 2 s steps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs/t1_n_r2"))
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--shadow-steps", type=int, default=2, help="h of the paired shadow (run order)")
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--extra-checkpoints", default="2000")
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
    if args.n_envs == 1:
        env = make_env(args.arm, gym_config, args.shadow_steps)
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        env = SubprocVecEnv([partial(make_env, args.arm, replace(gym_config, seed_base=gym_config.seed_base + i * 100_000),
                                     args.shadow_steps) for i in range(args.n_envs)], start_method="spawn")
    kwargs = model_kwargs(T1_SAC, [int(x) for x in args.net_arch.split(",")] if args.net_arch else None,
                          n_envs=args.n_envs)
    kwargs["gradient_steps"] = TRAINER["updates_per_transition"] * args.n_envs
    kwargs["learning_rate"] = CosineLearningRate(1e-4, 1e-5)
    if args.learning_starts is not None:
        kwargs["learning_starts"] = args.learning_starts
    warmup = DELTA_WARMUP if args.delta_warmup is None else args.delta_warmup
    model = T1NSAC("MlpPolicy", env, interval_bellman=spec["interval_bellman"],
                   actor_every_transitions=TRAINER["actor_every_transitions"],
                   seed=args.seed, device="cpu", verbose=0, mc_lambda=LAMBDA_MC, eq_lambda=LAMBDA_EQ,
                   audit_every=args.audit_every, delta_lambda=0.0, delta_warmup_decisions=warmup,
                   delta_target_share=DELTA_TARGET_SHARE if spec["relative"] else None,
                   delta_cap_share=DELTA_CAP_SHARE, delta_calibration_steps=args.delta_calibration_steps, **kwargs)
    cfg = t1n_env_config(T1N_FUEL_COST)
    manifest = {
        "experiment": "T1-N 2x2 (Research Plan R2, 2026-10-18)", "arm": args.arm, "arm_spec": spec,
        "trainer": TRAINER | {"critic_updates_per_vector_step": kwargs["gradient_steps"]},
        "started_utc": datetime.now(timezone.utc).isoformat(), "argv": vars(args) | {"log_root": str(args.log_root)},
        "task": {"version": T1N_VERSION, "initial": asdict(cfg.initial), "reward": asdict(cfg.reward),
                 "kappa": T1N_KAPPA, "entry_speed_m_s": T1N_ENTRY_SPEED_M_S},
        "interface": INTERFACE_VERSION, "gstar_version": f"{GSTAR_VERSION}-conditional",
        "observation": list(T1N_OBSERVATION_NAMES), "gym": asdict(gym_config),
        "sac": asdict(T1_SAC) | {k: v for k, v in kwargs.items() if k in ("learning_starts",)},
        "learner": {"class": "train.t1_n_sac.T1NSAC", "interval_bellman": spec["interval_bellman"],
                    "uniform_eligibility": True, "lambda_mc": LAMBDA_MC, "lambda_eq": LAMBDA_EQ,
                    "relative_term": spec["relative"], "delta_target_share": DELTA_TARGET_SHARE if spec["relative"] else None,
                    "delta_cap_share": DELTA_CAP_SHARE, "delta_warmup_steps": warmup, "delta_huber": 10.0,
                    "alpha": "0.005 x review gap"},
        **({"shadow": asdict(T1NShadowConfig(horizon_steps=args.shadow_steps))} if spec["relative"] else {}),
        "training_openings": f"{gym_config.seed_base} + env_index * 100000 + episode index",
        "n_envs": args.n_envs, "environment": _environment_record(), **provenance,
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
        "actor_updates": model.actor_updates, "critic_updates": model._n_updates}))
    (log_dir / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
                                                       "decisions": int(model.num_timesteps)}))
    env.close()


if __name__ == "__main__":
    main()
