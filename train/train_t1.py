"""Train M0 (plain SAC + G* + shared MPC) or M1 (same + gated MPC counterfactual critic target) on T1 from zero.

``--steps`` counts 2 s decisions (over all parallel environments); each collected
decision gets ``updates_per_transition`` gradient updates. Training openings
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

M2 (``train/t1_m1.py``, ``env/t1_counterfactual.py``; ``m1`` is kept as an alias) differs
only in the critic loss and in the cached counterfactual it stores; its environment
transitions are bitwise those of M0 for the same actions. Since the 2026-10-15 learnability
fixes M0 and M2 are the *same* class (``M1SAC``) with the same critic fixes (``--lambda-mc``,
``--lambda-eq``); M0 has ``lam_cf = 0`` and an env without counterfactual.

    python -B -m train.train_t1 --method m0 --steps 20000 --seed 0 --run-name t1_m0_s0 --lambda-mc L --lambda-eq L
    python -B -m train.train_t1 --method m2 --steps 20000 --seed 0 --run-name t1_m2_s0 --lambda-mc L --lambda-eq L
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

from env.t1_counterfactual import CounterfactualConfig, T1CounterfactualEnv
from env.t1_gstar import GSTAR_VERSION
from env.t1_gym import GYM_INTERFACE_VERSION, OBSERVATION_NAMES, T1GStarEnv, T1GymConfig
from train.t1_m1 import M1SAC
from train.train_hybrid import _code_provenance

#: M1 candidate values (design 2026-10-12 sec. 5.4); frozen only on the user's approval
M1_LAMBDA = 0.5
M1_WARMUP_FRACTION = 0.15  # superseded by the absolute warm-up below
M1_WARMUP_DECISIONS = 3000
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
    #: gradient updates per collected transition (round-1 seed 0 ran 1; an update costs
    #: 20-40 ms against ~2 s of simulation per decision, so 4 is nearly free)
    updates_per_transition: int = 4
    ent_coef: float = 0.005
    target_entropy: float = -2.0
    target_update_interval: int = 1
    net_arch: tuple[int, ...] = (256, 256, 256)


T1_SAC = T1SACConfig()


def model_kwargs(config: T1SACConfig, net_arch: list[int] | None = None, n_envs: int = 1) -> dict[str, Any]:
    values = asdict(config)
    values["gradient_steps"] = values.pop("updates_per_transition") * n_envs
    arch = net_arch or list(values.pop("net_arch"))
    values.pop("net_arch", None)
    values["policy_kwargs"] = {"net_arch": arch, "activation_fn": nn.ReLU}
    # The env already returns the 300 s limit as *terminated* (time is
    # observed, the timeout penalty is an outcome); the only truncation it
    # returns is a numerical failure, which SB3 must bootstrap through.
    values["replay_buffer_kwargs"] = {"handle_timeout_termination": True}
    return values


class EpisodeLog(BaseCallback):
    """``episodes.jsonl``: one line per finished episode (outcome, every violated rule, end
    phase, plan usage, counterfactual counts). ``decisions.jsonl``: one line per decision
    (action, tau request / effective, plan status, slip, kappa, reward, event, and for M1
    the counterfactual kind, w, gate reasons and the real-action prediction errors)."""

    def __init__(self, path: Path, decisions_path: Path) -> None:
        super().__init__()
        self.path = path
        self._decisions = decisions_path.open("a")
        self._counts: dict[int, dict[str, float]] = {}
        self._k: dict[int, int] = {}

    def _on_training_end(self) -> None:
        self._decisions.close()

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        actions = np.asarray(self.locals["actions"], dtype=float).reshape(len(infos), -1)
        rewards = np.asarray(self.locals["rewards"], dtype=float).reshape(-1)
        for e, info in enumerate(infos):
            self._one(e, info, actions[e], float(rewards[e]))
        return True

    def _one(self, e: int, info: dict, action: np.ndarray, reward: float) -> None:
        execution = info.get("execution", {})
        cf = info.get("counterfactual")
        k = self._k.get(e, 0)
        line = {
            "t": int(self.num_timesteps), "env": e, "k": k, "a": [round(float(v), 5) for v in action],
            "tau_req": round(float(execution.get("tau_request_s", np.nan)), 3),
            "tau_eff": round(float(execution.get("tau_effective_s", np.nan)), 3),
            "status": execution.get("plan_status"), "slip": execution.get("slip_s"),
            "kappa": round(float(info.get("kappa", np.nan)), 4),
            "r": round(reward, 6), "event": info.get("event"),
            "req": execution.get("request_kind"), "trig": execution.get("new_plan_trigger"),
            "deadline": execution.get("deadline"),
            "dev": None if execution.get("deviation_before_m") is None else round(float(execution["deviation_before_m"]), 4),
        }
        if cf is not None:
            line["cf"] = {"kind": cf["kind"], "w": int(cf["w"]), "reasons": cf["reasons"],
                          "e": None if cf["errors_real"] is None else
                          {key: round(float(v), 6) for key, v in cf["errors_real"].items()},
                          "tau_eff_alt": cf["tau_effective_alt_s"], "alt_status": cf["alt_status"]}
        self._decisions.write(json.dumps(line, default=float) + "\n")
        self._k[e] = k + 1
        c = self._counts.setdefault(e, {})
        c["decisions"] = c.get("decisions", 0) + 1
        c["new_plans"] = c.get("new_plans", 0) + int(execution.get("new_plan", False))
        c["slips"] = c.get("slips", 0) + int(execution.get("slip_s", 0.0) > 0.0)
        c["holds"] = c.get("holds", 0) + int(str(execution.get("plan_status", "")).startswith("hold"))
        c["keeps"] = c.get("keeps", 0) + int(str(execution.get("plan_status", "")).startswith("kept"))
        c["plan_wall_s"] = c.get("plan_wall_s", 0.0) + execution.get("plan_wall_s", 0.0)
        for name in ("keep", "later", "earlier", "no_plan"):
            c[f"req_{name}"] = c.get(f"req_{name}", 0) + int(execution.get("request_kind") == name)
        for name in ("no_plan", "deviation", "deadline", "request"):
            c[f"replan_{name}"] = c.get(f"replan_{name}", 0) + int(execution.get("new_plan_trigger") == name)
        c["deadline_clamped"] = c.get("deadline_clamped", 0) + int(execution.get("deadline") == "clamped")
        if cf is not None:
            c["cf_valid"] = c.get("cf_valid", 0) + int(cf["w"])
            for reason in cf["reasons"]:
                key = "cf_closed_" + reason.split(":")[0]
                c[key] = c.get(key, 0) + 1
            c["cf_available"] = c.get("cf_available", 0) + int(cf["a_tilde"] is not None)
            c["cf_overhead_wall_s"] = c.get("cf_overhead_wall_s", 0.0) + cf["overhead_wall_s"]
        if "episode_summary" in info:
            # opening, end phase and violations come from the env's own terminal record:
            # by now the vectorised wrapper has already reset the environment.
            record = {
                "num_timesteps": int(self.num_timesteps), "env": e,
                **info["episode_summary"],
                "numerical_error": info.get("numerical_error"),
                **c,
            }
            with self.path.open("a") as handle:
                handle.write(json.dumps(record, default=float) + "\n")
            self._decisions.flush()
            self._counts[e] = {}
            self._k[e] = 0


class Checkpoint(BaseCallback):
    """Checkpoints each time the decision count crosses a multiple of ``every`` (saved under
    that multiple), a heartbeat at every multiple of 100 crossed, a technical stop
    (``ABORTED.json``) if any network parameter becomes non-finite, and a clean stop when
    ``STOP_REQUESTED`` exists in the run directory (the plateau rule of the run order)."""

    def __init__(self, every: int, directory: Path, prefix: str, run_dir: Path) -> None:
        super().__init__()
        self.every, self.directory, self.prefix, self.run_dir = every, directory, prefix, run_dir
        self.started = perf_counter()
        self._last_n = 0

    def _on_training_start(self) -> None:
        # the untrained policy is the zero point of the development curve: under interface v2 its
        # deterministic action (a0 ~ 0) keeps the plan, so learning is read against it, not against 0
        self.model.save(self.directory / f"{self.prefix}_0_decisions")

    def _on_step(self) -> bool:
        n, last = self.num_timesteps, self._last_n
        self._last_n = n
        if n // 100 > last // 100:
            finite = all(bool(torch.isfinite(p).all()) for p in self.model.policy.parameters())
            (self.run_dir / "heartbeat.json").write_text(json.dumps({
                "decisions": n, "wall_s": perf_counter() - self.started,
                "utc": datetime.now(timezone.utc).isoformat(), "parameters_finite": finite}))
            if not finite:
                (self.run_dir / "ABORTED.json").write_text(json.dumps(
                    {"reason": "non-finite network parameters", "decisions": n}))
                return False
        for mark in range((last // self.every + 1) * self.every, n + 1, self.every):
            self.model.save(self.directory / f"{self.prefix}_{mark}_decisions")
            if hasattr(self.model, "cf_stats"):
                (self.directory / f"{self.prefix}_{mark}_cf_stats.json").write_text(json.dumps(self.model.cf_stats))
            if hasattr(self.model, "fix_stats"):
                (self.directory / f"{self.prefix}_{mark}_fix_stats.json").write_text(json.dumps(self.model.fix_stats))
        if (self.run_dir / "STOP_REQUESTED").exists():
            (self.run_dir / "stopped.json").write_text(json.dumps({
                "decisions": n, "utc": datetime.now(timezone.utc).isoformat(),
                "request": (self.run_dir / "STOP_REQUESTED").read_text()}))
            return False
        return True


def _make_env(method: str, gym_config: T1GymConfig):
    if method in ("m1", "m2"):
        return T1CounterfactualEnv(gym_config, counterfactual=COUNTERFACTUAL)
    return T1GStarEnv(gym_config)


def _environment_record() -> dict[str, Any]:
    import os
    import platform
    import sys
    from importlib.metadata import version

    packages = {}
    for name in ("numpy", "scipy", "cvxpy", "clarabel", "osqp", "torch", "stable-baselines3", "gymnasium"):
        try:
            packages[name] = version(name)
        except Exception:  # noqa: BLE001
            packages[name] = None
    return {
        "python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "packages": packages, "torch_threads": torch.get_num_threads(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "command": sys.argv,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("m0", "m1", "m2"), required=True, help="m1 = m2 (alias)")
    parser.add_argument("--steps", type=int, required=True, help="2 s decisions")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--log-root", type=Path, default=Path("logs/t1"))
    parser.add_argument("--checkpoint-freq", type=int, default=5_000)
    parser.add_argument("--learning-starts", type=int, default=None, help="smoke tests only")
    parser.add_argument("--net-arch", default=None, help="smoke tests only, e.g. 16,16")
    parser.add_argument("--cf-warmup", type=int, default=None, help="smoke tests only: M1 warm-up decisions")
    parser.add_argument("--save-replay", action="store_true", help="development checks only")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="parallel training environments (one process each); env i uses openings "
                             "seed_base + i * 100000 + episode")
    parser.add_argument("--lambda-mc", type=float, default=0.0, help="critic fix: Monte-Carlo return weight")
    parser.add_argument("--lambda-eq", type=float, default=0.0, help="critic fix: keep-band equivalence weight")
    parser.add_argument("--mc-window", type=int, default=10_000, help="MC labels only for the last n transitions")
    parser.add_argument("--grad-probe-every", type=int, default=0,
                        help="calibration: unweighted aux gradient norms every n updates (read-only)")
    parser.add_argument("--audit-every", type=int, default=0, help="critic audits every n updates (read-only)")
    parser.add_argument("--allow-dirty", action="store_true", help="development checks only")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.set_num_threads(1)
    provenance = _code_provenance()
    if provenance["code_dirty"] is not False and not args.allow_dirty:
        raise SystemExit(f"refusing to train from a modified or unknown checkout: {provenance}")
    log_dir = args.log_root / args.run_name
    if log_dir.exists():
        raise FileExistsError(log_dir)
    (log_dir / "checkpoints").mkdir(parents=True)
    gym_config = T1GymConfig(seed_base=1_000_000 * (args.seed + 1))
    if args.n_envs == 1:
        env = _make_env(args.method, gym_config)
    else:
        from functools import partial

        from stable_baselines3.common.vec_env import SubprocVecEnv

        env = SubprocVecEnv(
            [partial(_make_env, args.method, replace(gym_config, seed_base=gym_config.seed_base + i * 100_000))
             for i in range(args.n_envs)],
            start_method="spawn",
        )
    kwargs = model_kwargs(T1_SAC, [int(x) for x in args.net_arch.split(",")] if args.net_arch else None,
                          n_envs=args.n_envs)
    if args.learning_starts is not None:
        kwargs["learning_starts"] = args.learning_starts
    warmup = M1_WARMUP_DECISIONS if args.cf_warmup is None else args.cf_warmup
    counterfactual = args.method in ("m1", "m2")
    fixes = {"mc_lambda": args.lambda_mc, "eq_lambda": args.lambda_eq, "mc_window": args.mc_window,
             "mc_delta": 10.0, "grad_probe_every": args.grad_probe_every, "audit_every": args.audit_every}
    model = M1SAC("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                  cf_lambda=M1_LAMBDA if counterfactual else 0.0, cf_warmup_decisions=warmup, **fixes, **kwargs)
    manifest = {
        "method": args.method,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "argv": vars(args) | {"log_root": str(args.log_root)},
        "sac": asdict(T1_SAC) | {k: v for k, v in kwargs.items() if k in ("learning_starts", "policy_kwargs")},
        "gym": asdict(gym_config),
        "observation": list(OBSERVATION_NAMES),
        "gstar_version": GSTAR_VERSION,
        "training_openings": f"{gym_config.seed_base} + env_index * 100000 + episode index",
        "n_envs": args.n_envs, "gradient_steps_per_vector_step": kwargs["gradient_steps"],
        "gym_interface_version": GYM_INTERFACE_VERSION,
        "critic_fixes": fixes, "algorithm_class": "train.t1_m1.M1SAC",
        "environment": _environment_record(),
        **({"m1": {"lambda": M1_LAMBDA, "warmup_decisions": warmup,
                   "counterfactual": asdict(COUNTERFACTUAL)}} if counterfactual else {}),
        **provenance,
    }
    manifest["sac"]["policy_kwargs"] = {"net_arch": kwargs["policy_kwargs"]["net_arch"], "activation_fn": "ReLU"}
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    model.learn(
        total_timesteps=args.steps,
        callback=[EpisodeLog(log_dir / "episodes.jsonl", log_dir / "decisions.jsonl"),
                  Checkpoint(args.checkpoint_freq, log_dir / "checkpoints", args.method, log_dir)],
    )
    if (log_dir / "ABORTED.json").exists():
        raise SystemExit(f"training stopped: {(log_dir / 'ABORTED.json').read_text()}")
    model.save(log_dir / "final")
    if hasattr(model, "cf_stats"):
        (log_dir / "cf_stats.json").write_text(json.dumps(model.cf_stats))
        (log_dir / "fix_stats.json").write_text(json.dumps(model.fix_stats))
    if args.save_replay:
        model.save_replay_buffer(log_dir / "replay_buffer")
    (log_dir / "finished.json").write_text(json.dumps({"finished_utc": datetime.now(timezone.utc).isoformat(),
                                                       "decisions": int(model.num_timesteps),
                                                       "stopped_on_request": (log_dir / "stopped.json").exists()}))
    env.close()


if __name__ == "__main__":
    main()
