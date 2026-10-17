"""Train one arm of the T1-L three-arm experiment (plan 2026-10-15, section 3) or of the T1-L-S
three-arm experiment (plan 2026-10-16) from zero.

T1-L-S arms (``sb2`` / ``sn0`` / ``sndelta``): the same task, interface, reward, MPC, network and
budget, review-interval targets (``train.t1_tcpf``) and one of two trainer configurations
(``--trainer``, chosen by the run order's pre-registered rule): A = the 166244b trainer (UTD 4,
an actor update per critic update, constant 1e-4); B = one critic update per real transition,
one actor update per 3 real transitions on the global transition clock, cosine 1e-4 -> 1e-5.
With ``--n-envs`` N the critic updates per vector step are UTD x N for either trainer.

    B2      review every 2 s step (review_period 1), no relative MPC term
    N0      review every 3 steps (6 s) and after plan events, no relative MPC term
    NDelta  as N0, plus the relative MPC term: after a 3000-step warm-up, weight calibrated once
            from gradients to a 5 % median share of the TD gradient, each update capped at
            10 % (``train.t1_option_sac``; revision of da9be02 after the collaboration review)

Same task (T1-L, 12-18 m), interface (``env.t1_task_review``), reward, MPC, network, SAC
settings and budget for the three; ``--steps`` counts real 2 s steps. Shared critic fixes:
lambda_mc 0.03, lambda_eq 0.1 (frozen in the plan). Training openings
``1_000_000 * (seed + 1) + env_index * 100000 + episode``.

    python -B -m train.train_t1_option --arm n0 --steps 40000 --seed 0 --run-name n0_s0 --log-root logs/t1_l_option_20261015
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from env.t1_counterfactual import CounterfactualConfig
from env.t1_gstar import GSTAR_VERSION
from env.t1_gym import T1GymConfig
from env.t1_l_task import T1L_RANGE_M, t1l_env_config
from env.t1_option_cf import T1LReviewCFEnv
from env.t1_task_review import INTERFACE_VERSION, REVIEW_OBSERVATION_NAMES, T1LReviewEnv
from train.t1_option_sac import OptionSAC
from train.t1_tcpf import CosineLearningRate, TCPFSAC
from train.train_hybrid import _code_provenance
from train.train_t1 import T1_SAC, Checkpoint, EpisodeLog, _environment_record, model_kwargs

ARMS: dict[str, dict[str, Any]] = {
    "b2": {"review_period_steps": 1, "delta_lambda": 0.0, "counterfactual": False},
    "n0": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": False},
    "ndelta": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": True,
               "delta_target_share": 0.05, "delta_cap_share": 0.10},
    # T1-L-S (plan 2026-10-16): the same three arms with review-interval targets
    "sb2": {"review_period_steps": 1, "delta_lambda": 0.0, "counterfactual": False, "interval_bellman": True},
    "sn0": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": False, "interval_bellman": True},
    "sndelta": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": True, "interval_bellman": True,
                "delta_target_share": 0.05, "delta_cap_share": 0.10},
    # T1-O 2 x 2 (research decision 2026-10-17): one eligibility for all four arms; factors =
    # review-interval target (R3 / TCPF) and the relative MPC term (Delta1 / TCPF); trainer A only
    "or1": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": False, "interval_bellman": False,
            "uniform_eligibility": True, "task": "t1o"},
    "or3": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": False, "interval_bellman": True,
            "uniform_eligibility": True, "task": "t1o"},
    "od1": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": True, "interval_bellman": False,
            "uniform_eligibility": True, "task": "t1o", "delta_target_share": 0.05, "delta_cap_share": 0.10},
    "otcpf": {"review_period_steps": 3, "delta_lambda": 0.0, "counterfactual": True, "interval_bellman": True,
              "uniform_eligibility": True, "task": "t1o", "delta_target_share": 0.05, "delta_cap_share": 0.10},
}
TRAINERS: dict[str, dict[str, Any]] = {
    "A": {"updates_per_transition": 4, "actor_every_transitions": None, "learning_rate": "constant 1e-4"},
    "B": {"updates_per_transition": 1, "actor_every_transitions": 3, "learning_rate": "cosine 1e-4 -> 1e-5 over real transitions"},
}
LAMBDA_MC = 0.03
LAMBDA_EQ = 0.1
DELTA_WARMUP = 3000
COUNTERFACTUAL = CounterfactualConfig()


def make_env(arm: str, gym_config: T1GymConfig, profile: str | None = None, fuel_cost: float | None = None):
    spec = ARMS[arm]
    cls = T1LReviewCFEnv if spec["counterfactual"] else T1LReviewEnv
    kwargs = {"counterfactual": COUNTERFACTUAL} if spec["counterfactual"] else {}
    if spec.get("task") == "t1o":
        from env.t1_o_task import T1O_DELAY_MAX_S, t1o_env_config

        if spec["counterfactual"]:
            kwargs["near_field_kappa"] = True
        return cls(gym_config, t1o_env_config(profile, fuel_cost), review_period_steps=spec["review_period_steps"],
                   delay_max_s=T1O_DELAY_MAX_S, **kwargs)
    return cls(gym_config, t1l_env_config(), review_period_steps=spec["review_period_steps"], **kwargs)


def _t1o_task_record(profile: str, fuel_cost: float) -> dict[str, Any]:
    from env.t1_o_task import T1O_DELAY_MAX_S, T1O_PROFILES, t1o_env_config

    cfg = t1o_env_config(profile, fuel_cost)
    return {"profile": f"t1o-{profile}", "spec": T1O_PROFILES[profile], "fuel_cost": fuel_cost,
            "delay_max_s": T1O_DELAY_MAX_S, "near_field_kappa_counterfactual": True,
            "initial": asdict(cfg.initial), "reward": asdict(cfg.reward)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--steps", type=int, required=True, help="real 2 s steps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", type=Path, default=None,
                        help="default logs/t1_l_option_20261015 (T1-L arms) or logs/t1_l_s_tcpf_20261016 (T1-L-S arms)")
    parser.add_argument("--trainer", choices=sorted(TRAINERS), default=None,
                        help="T1-L-S arms only (required there): A or B, per the run order's frozen rule")
    parser.add_argument("--profile", choices=("p1", "p2"), default=None, help="T1-O arms: physical profile")
    parser.add_argument("--fuel-cost", type=float, default=None, help="T1-O arms: fuel weight of the reward")
    parser.add_argument("--n-envs", type=int, default=1)
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
    t1o = spec.get("task") == "t1o"
    interval = bool(spec.get("interval_bellman", False)) or t1o
    if t1o:
        if args.profile is None or args.fuel_cost is None:
            raise SystemExit("T1-O arms need --profile and --fuel-cost (frozen run order)")
        if args.trainer not in (None, "A"):
            raise SystemExit("T1-O arms use trainer A only")
        args.trainer = "A"
    elif args.profile is not None or args.fuel_cost is not None:
        raise SystemExit("--profile / --fuel-cost are for the T1-O arms only")
    if interval and args.trainer is None:
        raise SystemExit("T1-L-S arms need --trainer A or B (run order)")
    if not interval and args.trainer not in (None, "A"):
        raise SystemExit("the T1-L arms (b2 / n0 / ndelta) use the 166244b trainer only")
    if args.log_root is None:
        args.log_root = Path("logs/t1_o_tcpf_20261017" if t1o else
                             "logs/t1_l_s_tcpf_20261016" if interval else "logs/t1_l_option_20261015")
    provenance = _code_provenance()
    if provenance["code_dirty"] is not False and not args.allow_dirty:
        raise SystemExit(f"refusing to train from a modified or unknown checkout: {provenance}")
    log_dir = args.log_root / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    (log_dir / "checkpoints").mkdir(parents=True)
    gym_config = T1GymConfig(seed_base=1_000_000 * (args.seed + 1))
    if args.n_envs == 1:
        env = make_env(args.arm, gym_config, args.profile, args.fuel_cost)
    else:
        from functools import partial

        from stable_baselines3.common.vec_env import SubprocVecEnv

        env = SubprocVecEnv([partial(make_env, args.arm, replace(gym_config, seed_base=gym_config.seed_base + i * 100_000),
                                     args.profile, args.fuel_cost)
                             for i in range(args.n_envs)], start_method="spawn")
    kwargs = model_kwargs(T1_SAC, [int(x) for x in args.net_arch.split(",")] if args.net_arch else None,
                          n_envs=args.n_envs)
    if args.learning_starts is not None:
        kwargs["learning_starts"] = args.learning_starts
    warmup = DELTA_WARMUP if args.delta_warmup is None else args.delta_warmup
    common = dict(seed=args.seed, device="cpu", verbose=0,
                  mc_lambda=LAMBDA_MC, eq_lambda=LAMBDA_EQ, audit_every=args.audit_every,
                  delta_lambda=spec["delta_lambda"], delta_warmup_decisions=warmup,
                  delta_target_share=spec.get("delta_target_share"), delta_cap_share=spec.get("delta_cap_share", 0.10),
                  delta_calibration_steps=args.delta_calibration_steps)
    trainer = TRAINERS[args.trainer] if interval else None
    if interval:
        kwargs["gradient_steps"] = trainer["updates_per_transition"] * args.n_envs
        if args.trainer == "B":
            kwargs["learning_rate"] = CosineLearningRate(1e-4, 1e-5)
        model = TCPFSAC("MlpPolicy", env, interval_bellman=bool(spec.get("interval_bellman", False)),
                        uniform_eligibility=bool(spec.get("uniform_eligibility", False)),
                        actor_every_transitions=trainer["actor_every_transitions"], **common, **kwargs)
    else:
        model = OptionSAC("MlpPolicy", env, **common, **kwargs)
    manifest = {
        "experiment": ("T1-O 2x2 (research decision 2026-10-17)" if t1o else
                       "T1-L-S three-arm (plan 2026-10-16)" if interval else "T1-L three-arm (plan 2026-10-15)"),
        "arm": args.arm, "arm_spec": spec,
        "trainer": None if not interval else {"name": args.trainer, **trainer,
                                              "critic_updates_per_vector_step": kwargs["gradient_steps"],
                                              "actor_clock": "global real transitions (num_timesteps) minus learning_starts"},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": vars(args) | {"log_root": str(args.log_root)},
        "task": (_t1o_task_record(args.profile, args.fuel_cost) if t1o else
                 {"profile": "t1l", "range_m": T1L_RANGE_M, "initial": asdict(t1l_env_config().initial)}),
        "interface": INTERFACE_VERSION, "gstar_version": GSTAR_VERSION,
        "observation": list(REVIEW_OBSERVATION_NAMES), "gym": asdict(gym_config),
        "sac": asdict(T1_SAC) | {k: v for k, v in kwargs.items() if k in ("learning_starts",)},
        "learner": {"class": "train.t1_tcpf.TCPFSAC" if interval else "train.t1_option_sac.OptionSAC",
                    "interval_bellman": bool(spec.get("interval_bellman", False)),
                    "uniform_eligibility": bool(spec.get("uniform_eligibility", False)),
                    "lambda_mc": LAMBDA_MC, "lambda_eq": LAMBDA_EQ,
                    "delta_lambda_fixed": spec["delta_lambda"], "delta_target_share": spec.get("delta_target_share"),
                    "delta_cap_share": spec.get("delta_cap_share"), "delta_calibration_steps": args.delta_calibration_steps,
                    "delta_probe_every_updates": 5, "delta_min_probes": 30,
                    "delta_warmup_steps": warmup, "delta_huber": 10.0,
                    "alpha": "0.005 x review gap (2 s steps closed by the review)", "absolute_cf_lambda": 0.0},
        **({"counterfactual": asdict(COUNTERFACTUAL)} if spec["counterfactual"] else {}),
        "training_openings": f"{gym_config.seed_base} + env_index * 100000 + episode index",
        "n_envs": args.n_envs, "gradient_steps_per_vector_step": kwargs["gradient_steps"],
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
        if hasattr(model, name):
            (log_dir / f"{name}.json").write_text(json.dumps(getattr(model, name)))
    if interval:
        buffer = model.replay_buffer
        (log_dir / "interval_audit.json").write_text(json.dumps({
            "counts": buffer.interval_counts, "lengths": {str(k): v for k, v in sorted(buffer.interval_lengths.items())},
            "actor_updates": model.actor_updates, "critic_updates": model._n_updates}))
    (log_dir / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
                                                       "decisions": int(model.num_timesteps),
                                                       "stopped_on_request": (log_dir / "stopped.json").exists()}))
    env.close()


if __name__ == "__main__":
    main()
