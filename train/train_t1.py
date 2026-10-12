"""Train M0 (plain SAC + G* v0.3.1 + shared MPC) or M1 (same + gated MPC counterfactual critic target) on T1 from zero.

``--steps`` counts 2 s decisions, one gradient update each. Training openings
are ``seed_base + episode`` with ``seed_base = 1_000_000 * (seed + 1)``: never
the R0 (2900xx), R1 (2910xx) or formal (>= 300000, < 1e6) openings.

SAC settings (``T1_SAC``) follow the project's coupled-SAC settings
(``train/hybrid_configs.py``) where the reasons carry over: ``ent_coef`` pinned
at 0.005 (DEAD_ENDS D02), the time limit terminal (time is observed), one
update per decision. They differ in ``gamma`` = 0.999 per 2 s, which is the
T1 reward's own discount (the time cost -0.05 / s keeps the pressure to
finish), and ``target_entropy`` = -2 for the 2-D action (inert while
``ent_coef`` is fixed, Trap 5).

Every finished episode appends one JSON line to ``episodes.jsonl`` (opening,
outcome, time, delta-v, return, how the G* plan was used). That is behaviour-
policy statistics (Trap 11), never a gate.

M1 (``train/t1_m1.py``, ``env/t1_counterfactual.py``) differs only in the
critic loss and in the cached counterfactual it stores; its environment
transitions are bitwise those of M0 for the same actions.

    python -B -m train.train_t1 --method m0 --steps 20000 --seed 0 --run-name t1_m0_s0
    python -B -m train.train_t1 --method m1 --steps 20000 --seed 0 --run-name t1_m1_s0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from env.t1_counterfactual import CounterfactualConfig, T1CounterfactualEnv
from env.t1_gym import OBSERVATION_NAMES, T1GStarEnv, T1GymConfig
from train.t1_m1 import M1SAC
from train.train_hybrid import _code_provenance

#: M1 candidate values (design 2026-10-12 sec. 5.4); frozen only on the user's approval
M1_LAMBDA = 0.5
M1_WARMUP_FRACTION = 0.15
COUNTERFACTUAL = CounterfactualConfig()


@dataclass(frozen=True)
class T1SACConfig:
    learning_rate: float = 1.0e-4
    buffer_size: int = 300_000
    learning_starts: int = 1_000
    batch_size: int = 256
    tau: float = 0.001
    gamma: float = 0.999
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: float = 0.005
    target_entropy: float = -2.0
    target_update_interval: int = 1
    net_arch: tuple[int, ...] = (256, 256, 256)


T1_SAC = T1SACConfig()


def model_kwargs(config: T1SACConfig, net_arch: list[int] | None = None) -> dict[str, Any]:
    values = asdict(config)
    arch = net_arch or list(values.pop("net_arch"))
    values.pop("net_arch", None)
    values["policy_kwargs"] = {"net_arch": arch, "activation_fn": nn.ReLU}
    # The env already returns the 300 s limit as *terminated* (time is
    # observed, the timeout penalty is an outcome); the only truncation it
    # returns is a numerical failure, which SB3 must bootstrap through.
    values["replay_buffer_kwargs"] = {"handle_timeout_termination": True}
    return values


class EpisodeLog(BaseCallback):
    """One JSON line per finished episode plus per-episode plan-usage counters."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self._counts: dict[str, float] = {}
        self._opening: int | None = None

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        execution = info.get("execution", {})
        c = self._counts
        c["decisions"] = c.get("decisions", 0) + 1
        c["new_plans"] = c.get("new_plans", 0) + int(execution.get("new_plan", False))
        c["slips"] = c.get("slips", 0) + int(execution.get("slip_s", 0.0) > 0.0)
        c["holds"] = c.get("holds", 0) + int(str(execution.get("plan_status", "")).startswith("hold"))
        c["plan_wall_s"] = c.get("plan_wall_s", 0.0) + execution.get("plan_wall_s", 0.0)
        cf = info.get("counterfactual")
        if cf is not None:
            c["cf_valid"] = c.get("cf_valid", 0) + int(cf["w"])
            c["cf_available"] = c.get("cf_available", 0) + int(cf["a_tilde"] is not None)
            c["cf_overhead_wall_s"] = c.get("cf_overhead_wall_s", 0.0) + cf["overhead_wall_s"]
        if "episode_summary" in info:
            env = self.training_env.envs[0].unwrapped  # type: ignore[attr-defined]
            record = {
                "num_timesteps": int(self.num_timesteps),
                "opening": env.current_opening,
                **info["episode_summary"],
                **self._counts,
            }
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
            self._counts = {}
        return True


class Checkpoint(BaseCallback):
    def __init__(self, every: int, directory: Path, prefix: str) -> None:
        super().__init__()
        self.every, self.directory, self.prefix = every, directory, prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every == 0:
            self.model.save(self.directory / f"{self.prefix}_{self.num_timesteps}_decisions")
        return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("m0", "m1"), required=True)
    parser.add_argument("--steps", type=int, required=True, help="2 s decisions")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs/t1"))
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--learning-starts", type=int, default=None, help="smoke tests only")
    parser.add_argument("--net-arch", default=None, help="smoke tests only, e.g. 16,16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    log_dir = args.log_root / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    (log_dir / "checkpoints").mkdir(parents=True)
    gym_config = T1GymConfig(seed_base=1_000_000 * (args.seed + 1))
    if args.method == "m1":
        env = T1CounterfactualEnv(gym_config, counterfactual=COUNTERFACTUAL)
    else:
        env = T1GStarEnv(gym_config)
    kwargs = model_kwargs(T1_SAC, [int(x) for x in args.net_arch.split(",")] if args.net_arch else None)
    if args.learning_starts is not None:
        kwargs["learning_starts"] = args.learning_starts
    if args.method == "m1":
        model = M1SAC("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                      cf_lambda=M1_LAMBDA, cf_warmup_fraction=M1_WARMUP_FRACTION, **kwargs)
    else:
        model = SAC("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0, **kwargs)
    manifest = {
        "method": args.method,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": vars(args) | {"log_root": str(args.log_root)},
        "sac": asdict(T1_SAC) | {k: v for k, v in kwargs.items() if k in ("learning_starts", "policy_kwargs")},
        "gym": asdict(gym_config),
        "observation": list(OBSERVATION_NAMES),
        **({"m1": {"lambda": M1_LAMBDA, "warmup_fraction": M1_WARMUP_FRACTION,
                   "counterfactual": asdict(COUNTERFACTUAL)}} if args.method == "m1" else {}),
        **_code_provenance(),
    }
    manifest["sac"]["policy_kwargs"] = {"net_arch": kwargs["policy_kwargs"]["net_arch"], "activation_fn": "ReLU"}
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    model.learn(
        total_timesteps=args.steps,
        callback=[EpisodeLog(log_dir / "episodes.jsonl"), Checkpoint(args.checkpoint_freq, log_dir / "checkpoints", args.method)],
    )
    model.save(log_dir / "final")
    if hasattr(model, "cf_stats"):
        (log_dir / "cf_stats.json").write_text(json.dumps(model.cf_stats))
    (log_dir / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
                                                       "decisions": int(model.num_timesteps)}))


if __name__ == "__main__":
    main()
