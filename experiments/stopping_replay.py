"""Deterministic replay of the stopping row, for the paper's mechanism analysis.

Not a gate: nothing here passes or fails the method. Run order:
``docs/collaboration/upper/run_orders/FINAL_EVIDENCE_20261007.md`` (collab).

    replay     re-run the deterministic stopping episode of each opening and
               check it bitwise against the formal result file. At every
               learned decision, record:
                 * distance, entry-axis misalignment, task progress and
                   commitment, remaining time;
                 * beta, Q_H, and V_C under four definitions -- deterministic
                   / soft, online / target critic -- to audit the
                   continuation-value semantics;
               and record the per-decision reward.
               With ``--counterfactual``, also roll out from two boundary
               states of each mid-episode handoff (the handoff decision k*
               and the one before it, k*-1):
                 A  hand off to Pure MPC now                  -> realized G_A vs Q_H(s)
                 B  continue one step with mu(s), then follow
                    the deployed stopping policy               -> realized G_B vs V_C(s)
               giving samples of eps_H = Q_H - G_A and eps_C = V_C - G_B,
               and of the true sign of A = G_A - G_B against the estimate.
    summarize  handoff distribution over state, Q_H - V_C trajectories,
               semantics audit, counterfactual errors, replay fidelity

Returns are discounted with the model's gamma on the environment reward (the
shaped reward the critics are trained on).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from experiments.v3_common import parse_seed_range
from experiments.v3_handoff_scan import ImpulseMeter, _outcome
from experiments.v3_stage_c import _pairs, _scans, _write_json
from experiments.v3_stopping import env_for_stopping_run, load_stopping_model
from train.stopping import discounted_returns_to_go
from train.train_hybrid import _code_provenance
from train.v3_values import sha256_file

SOFT_SAMPLES = 64
COUNTERFACTUAL_EPISODES = 16  # per model: the first mid-episode handoffs in seed order


def _obs_tensor(model: Any, observation: np.ndarray) -> th.Tensor:
    return th.as_tensor(np.asarray(observation, dtype=np.float32), device=model.device).reshape(1, -1)


def continuation_values(model: Any, observation: np.ndarray, sample_seed: int) -> dict[str, float]:
    """V_C at one state under the four definitions in play.

    ``v_det_online`` is what the stopping rule and the deployment use;
    ``v_det_target`` is what beta uses inside the Bellman target;
    ``v_soft_target`` is the SAC continuation term of the Bellman target
    (sampled action, entropy bonus); ``v_soft_online`` is its online twin.
    """

    obs = _obs_tensor(model, observation)
    ent = float(model.ent_coef_tensor.item())
    with th.no_grad():
        mu = model.actor(obs, deterministic=True)
        out = {
            "v_det_online": float(th.min(*model.critic(obs, mu)).item()),
            "v_det_target": float(th.min(*model.critic_target(obs, mu)).item()),
        }
        generator_state = th.get_rng_state()
        th.manual_seed(int(sample_seed))
        rep = obs.repeat(SOFT_SAMPLES, 1)
        actions, log_prob = model.actor.action_log_prob(rep)
        th.set_rng_state(generator_state)
        log_prob = log_prob.reshape(-1, 1)
        out["v_soft_online"] = float((th.min(*model.critic(rep, actions)) - ent * log_prob).mean().item())
        out["v_soft_target"] = float((th.min(*model.critic_target(rep, actions)) - ent * log_prob).mean().item())
    return out


def decision_record(env: Any, model: Any, observation: np.ndarray, k: int, sample_seed: int) -> dict[str, Any]:
    position = np.asarray(env._current_position(), dtype=np.float64)
    desired = np.asarray(env.environment_config.precapture_task.desired_position, dtype=np.float64)
    distance = float(np.linalg.norm(position))
    cos = float(np.dot(position, desired) / max(distance * np.linalg.norm(desired), 1e-12))
    values = model.stopping_values(observation)
    record = {
        "k": k,
        "time_s": float(env.env.time_seconds),
        "remaining_time_s": float(env.environment_config.max_time_s - env.env.time_seconds),
        "distance_m": distance,
        "entry_axis_misalignment_rad": float(np.arccos(np.clip(cos, -1.0, 1.0))),
        "task_progress_m": None if env._task_progress_m is None else float(env._task_progress_m),
        "task_commitment": None if env._task_commitment is None else float(env._task_commitment),
        "beta": values["beta"],
        "q_handoff": values["q_handoff"],
        **continuation_values(model, observation, sample_seed),
    }
    return record


def _rollout(env: Any, model: Any, mode: str, gamma: float,
             max_decisions: int | None = None) -> tuple[float, dict[str, Any], int]:
    """From the copy's current state: 'handoff' (Pure MPC to the end) or
    'continue' (mu now, then the deployed stopping policy)."""

    zero = np.zeros(env.action_space.shape, dtype=np.float64)
    threshold = model.stopping_config.deployment_threshold
    observation = env._policy_observation_cached
    branch = "baseline" if mode == "handoff" else "learned"
    rewards: list[float] = []
    first = True
    info: dict[str, Any] = {}
    while True:
        if branch == "learned" and not first and model.stopping_values(observation)["beta"] >= threshold:
            branch = "baseline"
        if branch == "learned":
            action, _ = model.predict(observation, deterministic=True)
        else:
            action = zero
        first = False
        observation, reward, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch=branch)
        rewards.append(float(reward))
        if terminated or truncated or (max_decisions is not None and len(rewards) >= max_decisions):
            break
    return float(discounted_returns_to_go(rewards, gamma)[0]), _outcome(env, info), len(rewards)


def replay_episode(env: Any, model: Any, seed: int, meter: ImpulseMeter, controller_b: Any,
                   counterfactual_at: set[int] | None = None, max_decisions: int | None = None,
                   rollout_max_decisions: int | None = None) -> dict[str, Any]:
    """The deterministic stopping episode (same decisions as ``run_stopping_episode``)."""

    observation, _ = env.reset(seed=int(seed))
    meter.force_impulse_n_s = 0.0
    zero = np.zeros(env.action_space.shape, dtype=np.float64)
    threshold = model.stopping_config.deployment_threshold
    branch, handoff_k = "learned", None
    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    snapshots: dict[int, Any] = {}
    info: dict[str, Any] = {}
    decision = learned_decisions = 0
    while max_decisions is None or decision < max_decisions:
        if branch == "learned":
            record = decision_record(env, model, observation, decision, sample_seed=int(seed) * 1000 + decision)
            records.append(record)
            if counterfactual_at and decision in counterfactual_at:
                snapshot = copy.deepcopy(env, memo={id(env.controller): controller_b})
                snapshot._policy_observation_cached = np.asarray(observation).copy()
                snapshots[decision] = snapshot
            if record["beta"] >= threshold:
                branch, handoff_k = "baseline", decision
        if branch == "learned":
            action, _ = model.predict(observation, deterministic=True)
            learned_decisions += 1
        else:
            action = zero
        observation, reward, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch=branch)
        rewards.append(float(reward))
        decision += 1
        if terminated or truncated:
            break
    result = {
        "seed": int(seed), **_outcome(env, info), "decisions": decision, "handoff_k": handoff_k,
        "learned_decisions": learned_decisions,
        "equivalent_delta_v_m_s": meter.force_impulse_n_s / float(env.env.chaser_parameters.mass),
        "records": records, "rewards": rewards,
    }
    if snapshots:
        gamma = float(model.gamma)
        returns = discounted_returns_to_go(rewards, gamma)
        rows = []
        for k, snapshot in sorted(snapshots.items()):
            rec = next(r for r in records if r["k"] == k)
            out = {"k": k, "is_handoff_decision": k == handoff_k, "q_handoff": rec["q_handoff"],
                   "v_c": rec["v_det_online"], "realized_return_original": float(returns[k])}
            for mode in ("handoff", "continue"):
                trial = copy.deepcopy(snapshot, memo={id(snapshot.controller): controller_b})
                controller_b.reset()
                g, outcome, n = _rollout(trial, model, mode, gamma, rollout_max_decisions)
                out[mode] = {"return": g, "decisions": n, **outcome}
            out["eps_H"] = out["q_handoff"] - out["handoff"]["return"]
            out["eps_C"] = out["v_c"] - out["continue"]["return"]
            out["A_true"] = out["handoff"]["return"] - out["continue"]["return"]
            out["A_hat"] = out["q_handoff"] - out["v_c"]
            out["sign_agrees"] = (out["A_true"] >= 0) == (out["A_hat"] >= 0)
            rows.append(out)
        result["counterfactual"] = rows
    return result


FIDELITY_KEYS = ("handoff_k", "survival_s", "completed", "clean_completion", "equivalent_delta_v_m_s", "decisions")


def cmd_replay(args: argparse.Namespace) -> None:
    th.set_num_threads(1)
    manifest, env = env_for_stopping_run(args.run_dir, args.allow_incomplete_run)
    _, env_b = env_for_stopping_run(args.run_dir, args.allow_incomplete_run)
    model = load_stopping_model(args.run_dir, args.model_name)
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    formal = _scans(args.formal_dir) if args.formal_dir else {}
    model_sha = sha256_file(args.run_dir / args.model_name)
    seeds = parse_seed_range(args.seeds)
    cf_seeds: set[int] = set()
    if args.counterfactual:
        mids = [s for s in seeds if s in formal and (formal[s].get("handoff_k") or 0) >= 1]
        cf_seeds = set(mids[:COUNTERFACTUAL_EPISODES])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in seeds:
            path = args.output_dir / f"seed_{seed}.json"
            if path.exists():
                continue
            lock = path.with_suffix(".lock")
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            at = None
            if seed in cf_seeds:
                k = int(formal[seed]["handoff_k"])
                at = {k - 1, k}
            result = replay_episode(env, model, seed, meter, env_b.controller, at, args.max_decisions)
            reference = formal.get(seed)
            if reference is not None:
                if reference.get("model_sha256") != model_sha:
                    raise RuntimeError(f"seed {seed}: formal file was produced by a different model")
                result["fidelity"] = {key: result[key] == reference[key] for key in FIDELITY_KEYS}
                result["fidelity_ok"] = all(result["fidelity"].values())
            result.update(run_dir=str(args.run_dir), model_sha256=model_sha, max_decisions=args.max_decisions,
                          **_code_provenance())
            _write_json(path, result)
            lock.unlink()
            print(f"seed {seed}: replayed" + ("" if reference is None else f", fidelity_ok={result['fidelity_ok']}")
                  + (", counterfactual" if at else ""), flush=True)
    finally:
        env.close()
        env_b.close()


# -- summary ------------------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "median_abs": float(np.median(np.abs(a))), "max_abs": float(np.max(np.abs(a)))}


def summarize(episodes: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {"models": {}}
    for model, rows in episodes.items():
        handoff_states, flips, diffs = [], {"det_target": 0, "soft_target": 0, "soft_online": 0}, {
            "det_target": [], "soft_target": [], "soft_online": []}
        decisions = 0
        cf = [c for r in rows.values() for c in r.get("counterfactual", [])]
        for r in rows.values():
            for rec in r["records"]:
                decisions += 1
                gap = rec["q_handoff"] - rec["v_det_online"]
                for key in flips:
                    other = rec["q_handoff"] - rec[f"v_{key}"]
                    flips[key] += (gap >= 0) != (other >= 0)
                    diffs[key].append(rec[f"v_{key}"] - rec["v_det_online"])
            k = r["handoff_k"]
            if k is not None:
                rec = next(x for x in r["records"] if x["k"] == k)
                handoff_states.append({key: rec[key] for key in (
                    "k", "time_s", "remaining_time_s", "distance_m", "entry_axis_misalignment_rad",
                    "task_progress_m", "task_commitment")} | {"seed": r["seed"], "clean": r["clean_completion"]})
        ks = [r["handoff_k"] for r in rows.values()]
        mid = [h for h in handoff_states if h["k"] >= 1]
        out["models"][model] = {
            "episodes": len(rows),
            "fidelity_ok": all(r.get("fidelity_ok", False) for r in rows.values()),
            "handoff": {"k0": sum(k == 0 for k in ks), "never": sum(k is None for k in ks), "mid": len(mid),
                        "distinct_mid_k": len({h["k"] for h in mid}),
                        "mid_state_stats": {key: _stats([h[key] for h in mid if h[key] is not None]) for key in (
                            "k", "distance_m", "entry_axis_misalignment_rad", "task_progress_m",
                            "task_commitment", "remaining_time_s")},
                        "states": handoff_states},
            "continuation_value_semantics": {
                "learned_decisions": decisions,
                "sign_flips_vs_deployed_gap": flips,
                "difference_to_v_det_online": {key: _stats(v) for key, v in diffs.items()},
            },
            "counterfactual": {
                "states": len(cf),
                "eps_H": _stats([c["eps_H"] for c in cf]),
                "eps_C": _stats([c["eps_C"] for c in cf]),
                "sign_agreement": (sum(bool(c["sign_agrees"]) for c in cf) / len(cf)) if cf else None,
                "rows": cf,
            },
        }
    return out


def write_trajectories(episodes: dict[str, dict[int, dict[str, Any]]], path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "seed", "k", "time_s", "distance_m", "entry_axis_misalignment_rad",
                         "task_progress_m", "task_commitment", "beta", "q_handoff", "v_det_online",
                         "v_det_target", "v_soft_online", "v_soft_target", "handoff_k", "clean"])
        for model, rows in episodes.items():
            for r in rows.values():
                for rec in r["records"]:
                    writer.writerow([model, r["seed"], rec["k"], rec["time_s"], rec["distance_m"],
                                     rec["entry_axis_misalignment_rad"], rec["task_progress_m"],
                                     rec["task_commitment"], rec["beta"], rec["q_handoff"], rec["v_det_online"],
                                     rec["v_det_target"], rec["v_soft_online"], rec["v_soft_target"],
                                     r["handoff_k"], r["clean_completion"]])


def cmd_summarize(args: argparse.Namespace) -> None:
    episodes = {m: _scans(d) for m, d in _pairs(args.replay).items()}
    result = {"note": "mechanism analysis only; not a gate", **summarize(episodes), **_code_provenance()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    write_trajectories(episodes, args.output.with_suffix(".trajectories.csv"))
    print(json.dumps({m: {"fidelity_ok": v["fidelity_ok"], "handoff": {k: v["handoff"][k] for k in ("k0", "mid", "never")},
                          "sign_flips": v["continuation_value_semantics"]["sign_flips_vs_deployed_gap"],
                          "eps_H": v["counterfactual"]["eps_H"].get("median_abs"),
                          "eps_C": v["counterfactual"]["eps_C"].get("median_abs")}
                      for m, v in result["models"].items()}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-name", default="final_model.zip")
    p.add_argument("--seeds", required=True)
    p.add_argument("--formal-dir", default=None, help="the formal stopping-row JSONs to check against")
    p.add_argument("--counterfactual", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=None, help="tests only")
    p.add_argument("--allow-incomplete-run", action="store_true")
    p.set_defaults(func=cmd_replay)
    p = sub.add_parser("summarize")
    p.add_argument("--replay", action="append", required=True, help="MODEL=dir")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_summarize)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
