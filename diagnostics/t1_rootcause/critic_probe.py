"""P2/P3: critic ranking of keep / earlier 15 s / later 15 s at reconstructed states, against
closed-loop returns of the same actions from a cloned state; the M1 counterfactual target
of each action under each critic.

    # 1) pre-registered state selection from the published development logs (no model needed)
    python -B -m diagnostics.t1_rootcause.critic_probe select --evidence evidence/T1_V2_S0 \
        --output diagnostics/t1_rootcause/states.json
    # 2) probe (needs the checkpoints)
    python -B -m diagnostics.t1_rootcause.critic_probe probe --states diagnostics/t1_rootcause/states.json \
        --checkpoints 0k=<m1_0_decisions.zip>,20k=<m1_20000_decisions.zip>,30k=<m1_30000_decisions.zip> \
        --followups rule,traj_det,traj_stoch:2 --output-dir <dir> --workers 6

A state is reached by replaying its trajectory checkpoint deterministically from the opening
(the development evaluation is deterministic, so this reproduces the logged state; the
logged request at k is checked). Branches start from a clone of that state
(env.t1_counterfactual.CounterfactualPredictor.sync, validated bitwise for one 2 s step in
evidence/T1_M1_validationB); --verify-branch also checks a whole identity continuation.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from diagnostics.t1_rootcause.common import NEW_PLAN_PREFIXES, actor_policy, load_model, rule_policy, run_from

SHIFT_S = 15.0


# ---------------------------------------------------------------------- selection
def _log_requests(log: list[dict]) -> list[dict]:
    """Request kind of each logged decision with correct arrival bookkeeping."""

    arrival, out = None, []
    for k, x in enumerate(log):
        t = 2.0 * k
        remaining = max(arrival - t, 0.0) if arrival is not None else 0.0
        if x["status"] == "descent":
            kind = "descent"
        elif remaining <= 0.0:
            kind = "no_plan"
        else:
            d = x["tau_request_s"] - remaining
            kind = "keep" if abs(d) <= 1.0 else ("later" if d > 0 else "earlier")
        if x["status"].startswith(NEW_PLAN_PREFIXES):
            arrival = t + x["tau_effective_s"]
        elif x["status"] == "descent" or x["status"].startswith("hold"):
            arrival = None
        out.append({"k": k, "t_s": t, "kind": kind, "status": x["status"], "remaining_s": remaining,
                    "tau_request_s": x["tau_request_s"], "arrival_after_s": arrival})
    return out


def select(evidence: Path) -> list[dict]:
    """Pre-registered rule. For every development opening that failed at 30k: k1 = first
    executed postponement (a 'later' request adopted as a new plan); k2 = first executed
    postponement at t >= 120 s (if different). Controls: the same opening under 20k at the
    largest k <= k1 (and <= k2) that has a live plan."""

    with gzip.open(evidence / "dev_eval.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    by = {(r["mark"], r["opening"]): r for r in rows}
    states = []
    for (mark, opening), r in sorted(by.items()):
        if mark != 30000 or r["event"] == "success":
            continue
        req = _log_requests(r["decision_log"])
        postpone = [q for q in req if q["kind"] == "later" and q["status"].startswith(NEW_PLAN_PREFIXES)]
        if not postpone:
            continue
        picks = [("first_postponement", postpone[0])]
        late = [q for q in postpone if q["t_s"] >= 120.0 and q["k"] != postpone[0]["k"]]
        if late:
            picks.append(("first_postponement_after_120s", late[0]))
        control = _log_requests(by[(20000, opening)]["decision_log"])
        for why, q in picks:
            states.append({"traj": "30k", "opening": opening, "k": q["k"], "why": why, "logged": q})
            live = [c for c in control if c["k"] <= q["k"] and c["kind"] in ("keep", "earlier", "later")]
            if live:
                c = live[-1]
                states.append({"traj": "20k", "opening": opening, "k": c["k"], "why": f"control_for_{why}",
                               "logged": c})
    return states


# ---------------------------------------------------------------------- probe
def _reach(model, opening: int, k: int):
    from env.t1_gym import T1GStarEnv

    env = T1GStarEnv()
    obs, _ = env.reset(options={"opening": opening})
    for _ in range(k):
        a = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float64)
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            raise RuntimeError(f"episode ended before decision {k}")
    return env, obs


def _q(model, obs: np.ndarray, actions: dict[str, np.ndarray], grid_a1: float) -> dict:
    import torch as th

    policy = model.policy
    o = th.as_tensor(obs[None], dtype=th.float32)
    out: dict = {"actions": {}}
    with th.no_grad():
        for name, a in actions.items():
            q1, q2 = (float(v) for v in policy.critic(o, th.as_tensor(a[None], dtype=th.float32)))
            out["actions"][name] = {"a": [float(v) for v in a], "q1": q1, "q2": q2, "q_min": min(q1, q2)}
        own = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float64)
        grid = np.linspace(-1.0, 1.0, 81)
        acts = th.as_tensor(np.stack([grid, np.full_like(grid, grid_a1)], axis=1), dtype=th.float32)
        q1g, q2g = policy.critic(o.repeat(len(grid), 1), acts)
        out["a0_grid"] = grid.round(3).tolist()
        out["q_min_on_a0_grid"] = th.minimum(q1g, q2g).flatten().tolist()
        mean_actions, log_std, _ = policy.actor.get_action_dist_params(o)
        out["actor_mean_pre_tanh"] = mean_actions.flatten().tolist()
        out["actor_log_std"] = log_std.flatten().tolist()
    out["own_action"] = own.tolist()
    a = th.tensor(own[None], dtype=th.float32, requires_grad=True)
    q1, q2 = policy.critic(o, a)
    th.minimum(q1, q2).sum().backward()
    out["dq_da_at_own_action"] = a.grad.flatten().tolist()
    return out


def _m1_target(model, predictor, base_env, action: np.ndarray, gamma: float) -> dict:
    """y_cf = r_hat + gamma * (min target Q(z_hat, pi(z_hat)) - alpha log pi), as in train/t1_m1.py."""

    import torch as th

    predictor.sync(base_env)
    p = predictor.predict(action)
    z = th.as_tensor(np.asarray(p.observation, dtype=np.float32)[None])
    with th.no_grad():
        a_next, logp = model.policy.actor.action_log_prob(z)
        q = th.min(th.cat(model.policy.critic_target(z, a_next), dim=1), dim=1).values
        alpha = float(model.ent_coef) if not isinstance(model.ent_coef, str) else float("nan")
        v = float(q) - alpha * float(logp)
    return {"r_hat": p.reward, "plan_status": p.plan_status, "tau_effective_s": p.tau_effective_s,
            "end_violation": p.end_violation, "v_next": v, "y_cf": p.reward + gamma * v}


def _branch(predictor, base_env, first_action: np.ndarray, follow, k: int, seed: int | None):
    """Clone ``base_env`` into the predictor's shadow (one shadow reused per state: building
    an env compiles the MPC problem) and run the forced first action, then ``follow``."""

    import torch as th

    predictor.sync(base_env)
    env = predictor.shadow
    if seed is not None:
        th.manual_seed(seed)
    state = {"first": True}

    def policy(e, obs):
        if state["first"]:
            state["first"] = False
            return np.asarray(first_action, dtype=np.float64), None
        return follow(e, obs)

    obs = env._observation()
    return run_from(env, obs, policy, k0=k)


def probe_state(state: dict, checkpoints: dict[str, str], followups: list[str], output_dir: str,
                verify_branch: bool) -> None:
    from env.t1_counterfactual import CounterfactualPredictor

    from time import perf_counter

    path = Path(output_dir) / f"{state['traj']}_{state['opening']}_k{state['k']}.json"
    if path.exists():
        return
    started = perf_counter()
    models = {label: load_model(p) for label, p in checkpoints.items()}
    traj = models[state["traj"]]
    env, obs = _reach(traj, state["opening"], state["k"])
    a_traj = np.asarray(traj.predict(obs, deterministic=True)[0], dtype=np.float64)
    tau_now, _ = env.intent(a_traj)
    gamma = env.env.config.reward.gamma
    actions = {
        "keep": np.array([0.0, a_traj[1]]),
        "earlier15": np.array([env.action_for_shift(-SHIFT_S), a_traj[1]]),
        "later15": np.array([env.action_for_shift(SHIFT_S), a_traj[1]]),
    }
    out = {**state, "observation": obs.tolist(), "traj_action": a_traj.tolist(),
           "replay_check": {"tau_request_s": tau_now, "logged_tau_request_s": state["logged"]["tau_request_s"],
                            "match": bool(abs(tau_now - state["logged"]["tau_request_s"]) < 1e-6)},
           "remaining_s": env.generator.remaining_s(env.env), "time_s": env.env.time_s,
           "critics": {}, "m1_targets": {}, "rollouts": {}}
    for label, model in models.items():
        out["critics"][label] = _q(model, obs, actions, float(a_traj[1]))
    predictor = CounterfactualPredictor()
    for label, model in models.items():
        out["m1_targets"][label] = {name: _m1_target(model, predictor, env, a, gamma) for name, a in actions.items()}
    for name, a in actions.items():
        out["rollouts"][name] = {}
        for f in followups:
            if f == "rule":
                fn, seeds = rule_policy(), [None]
            elif f == "traj_det":
                fn, seeds = actor_policy(traj, "A", deterministic=True), [None]
            elif f.startswith("traj_stoch"):
                n = int(f.split(":")[1])
                fn, seeds = actor_policy(traj, "A", deterministic=False), list(range(n))
            else:
                raise ValueError(f)
            runs = []
            for s in seeds:
                r = _branch(predictor, env, a, fn, state["k"], None if s is None else 1000 * s + state["k"])
                if s is not None:
                    r.pop("log")  # stochastic follow-ups: outcomes only
                runs.append(r)
            out["rollouts"][name][f] = runs
    if verify_branch:
        fn = actor_policy(traj, "A", deterministic=True)
        branched = _branch(predictor, env, a_traj, fn, state["k"], None)
        direct = run_from(env, obs, fn, k0=state["k"])
        out["verify_branch"] = {key: (branched[key], direct[key], branched[key] == direct[key])
                                for key in ("event", "time_s", "discounted_return_from_k", "delta_v_m_s")}
    out["wall_s"] = perf_counter() - started
    out["simulated_decisions"] = state["k"] + sum(
        x["decisions"] for v in out["rollouts"].values() for runs in v.values() for x in runs)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    print(state["traj"], state["opening"], state["k"], "replay", out["replay_check"]["match"],
          "wall %.0f s, decisions %d" % (out["wall_s"], out["simulated_decisions"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("select", "probe"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence/T1_V2_S0"))
    parser.add_argument("--states", type=Path)
    parser.add_argument("--checkpoints", help="label=path,... (labels used by states: 20k, 30k)")
    parser.add_argument("--followups", default="rule,traj_det")
    parser.add_argument("--only", default=None, help="comma-separated indices into the state list")
    parser.add_argument("--verify-branch", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "select":
        states = select(args.evidence)
        text = json.dumps(states, indent=1)
        args.output.write_text(text)
        print(len(states), "states")
        return
    states = json.loads(args.states.read_text())
    if args.only:
        states = [states[int(i)] for i in args.only.split(",")]
    checkpoints = dict(item.split("=", 1) for item in args.checkpoints.split(","))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, checkpoints, args.followups.split(","), str(args.output_dir), args.verify_branch) for s in states]
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            pool.starmap(probe_state, jobs)
    else:
        for job in jobs:
            probe_state(*job)


if __name__ == "__main__":
    main()
