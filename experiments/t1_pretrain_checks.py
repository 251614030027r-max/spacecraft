"""P1 checks of a development training run (functional only; never a performance readout).

``stats RUN_DIR``     from decisions.jsonl / episodes.jsonl: opening timing (a0 -> tau
                      request -> tau effective), keep / re-plan / slip / hold shares, M1
                      gate availability and reasons by decision kind and by behaviour
                      (keeping vs changing the plan), outcomes, failure kinds, end phases.
``offline RUN_DIR``   with the run's saved replay buffer and final model (M1 only): one
                      training update on the formal clock just before (2 999) and at
                      (3 000) the warm-up threshold of a 20 000-decision run; finiteness of
                      Q and gradients, aux-loss and gradient norms, and for valid counter-
                      factual samples Q(z,a), Q(z,a~), y_cf, and whether one aux-only step
                      moves Q(z,a~) towards y_cf, split by decision kind. Nothing is tuned.
``reload RUN_DIR``    every saved checkpoint loads and acts on a real observation.

    python -B -m experiments.t1_pretrain_checks stats logs/t1/smoke_m1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _lines(path: Path) -> list[dict]:
    """Complete lines only: a log still being written may end in a partial line."""
    if not path.exists():
        return []
    text = path.read_text()
    lines = text.splitlines()
    if lines and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def stats(run: Path) -> dict:
    d = _lines(run / "decisions.jsonl")
    e = _lines(run / "episodes.jsonl")
    first = [x for x in d if x["k"] == 0]
    out: dict = {"decisions": len(d), "episodes": len(e)}
    out["opening_timing"] = {
        "a0": np.quantile([x["a"][0] for x in first], [0, .25, .5, .75, 1]).round(3).tolist() if first else None,
        "tau_request_s": np.quantile([x["tau_req"] for x in first], [0, .25, .5, .75, 1]).round(1).tolist() if first else None,
        "tau_effective_s": np.quantile([x["tau_eff"] for x in first], [0, .25, .5, .75, 1]).round(1).tolist() if first else None,
        "distinct_effective_tau": len({round(x["tau_eff"]) for x in first}),
        "earliest_requests": sum(x["tau_req"] <= 2.0 for x in first),
    }
    status = Counter(str(x["status"]).split(":")[0] for x in d)
    out["plan_status_share"] = {k: v / len(d) for k, v in status.most_common()}
    out["slip_share"] = sum((x["slip"] or 0) > 0 for x in d) / len(d)
    out["action_saturation_share"] = float(np.mean([max(abs(v) for v in x["a"]) > 0.99 for x in d]))
    cf = [x for x in d if "cf" in x]
    if cf:
        by_kind: dict = defaultdict(lambda: {"n": 0, "w": 0, "reasons": Counter()})
        for x in cf:
            g = by_kind[x["cf"]["kind"]]
            g["n"] += 1
            g["w"] += x["cf"]["w"]
            g["reasons"].update(r.split(":")[0] for r in x["cf"]["reasons"])
        out["gate_by_kind"] = {k: {"n": v["n"], "open": v["w"], "open_share": v["w"] / v["n"],
                                   "reasons": dict(v["reasons"].most_common())} for k, v in by_kind.items()}
        keeping = [x for x in cf if x["cf"]["kind"] == "keep_vs_shift"]
        changing = [x for x in cf if x["cf"]["kind"] in ("replan_vs_keep", "request_vs_earliest", "earliest_vs_later")]
        out["gate_open_share"] = {
            "all_decisions": sum(x["cf"]["w"] for x in cf) / len(cf),
            "policy_keeps_plan": (sum(x["cf"]["w"] for x in keeping) / len(keeping)) if keeping else None,
            "policy_changes_or_starts_plan": (sum(x["cf"]["w"] for x in changing) / len(changing)) if changing else None,
        }
        err = np.array([[x["cf"]["e"][k] for k in ("e_r", "e_p", "e_o")] for x in cf if x["cf"]["e"] is not None])
        out["real_prediction_error_quantiles"] = {
            k: np.quantile(err[:, i], [.5, .9, .99]).round(4).tolist() for i, k in enumerate(("e_r", "e_p", "e_o"))}
    events = Counter(x["event"] for x in e)
    out["outcomes"] = dict(events)
    out["failure_kinds"] = dict(Counter(k for x in e if x["event"] == "unsafe" for k in x.get("violations", {})))
    out["end_phase_by_outcome"] = {ev: dict(Counter(x.get("end_phase") for x in e if x["event"] == ev)) for ev in events}
    rewards = [x["r"] for x in d]
    out["terminal_rewards_seen"] = {"capture": sum(r > 50 for r in rewards), "unsafe": sum(r < -120 for r in rewards),
                                    "timeout": sum(-120 <= r < -50 for r in rewards)}
    out["episodes_without_plan_ever"] = sum(1 for x in e if x.get("new_plans", 0) == 0)
    return out


def offline(run: Path) -> dict:
    import torch as th

    from train.t1_m1 import M1SAC, auxiliary_critic_loss

    model = M1SAC.load(run / "final.zip", device="cpu")
    model.load_replay_buffer(run / "replay_buffer.pkl")
    buffer = model.replay_buffer
    n = buffer.pos if not buffer.full else buffer.buffer_size
    kinds = {x["t"] - 1: x["cf"]["kind"] for x in _lines(run / "decisions.jsonl") if "cf" in x}
    valid = np.flatnonzero(buffer.cf_weights[:n, 0, 0] > 0)  # env 0
    out: dict = {"replay_size": int(n), "valid_counterfactuals": int(len(valid))}
    model._total_timesteps = 20_000
    for clock in (2_999, 3_000):
        model.num_timesteps = clock
        out[f"lambda_at_{clock}"] = model.current_lambda()
    # per-sample audit on all valid samples, model as trained
    idx = valid
    obs = buffer.to_torch(buffer.observations[idx, 0])
    act = buffer.to_torch(buffer.actions[idx, 0])
    alt = buffer.to_torch(buffer.cf_actions[idx, 0])
    r_hat = buffer.to_torch(buffer.cf_rewards[idx, 0])
    z_hat = buffer.to_torch(buffer.cf_next_observations[idx, 0])
    with th.no_grad():
        a_next, logp = model.actor.action_log_prob(z_hat)
        v_next = th.min(th.cat(model.critic_target(z_hat, a_next), dim=1), dim=1, keepdim=True)[0] \
            - model.ent_coef_tensor * logp.reshape(-1, 1)
        y_cf = r_hat + model.gamma * v_next
        q_a = th.min(th.cat(model.critic(obs, act), dim=1), dim=1, keepdim=True)[0]
        q_alt = th.min(th.cat(model.critic(obs, alt), dim=1), dim=1, keepdim=True)[0]
    finite = bool(th.isfinite(y_cf).all() and th.isfinite(q_a).all() and th.isfinite(q_alt).all())
    # one aux-only gradient step on a copy of the critic: does Q(z, a~) move towards y_cf?
    import copy

    critic = copy.deepcopy(model.critic)
    opt = th.optim.Adam(critic.parameters(), lr=model.lr_schedule(1.0))
    w = th.ones_like(r_hat)
    before = [q.detach() for q in critic(obs, alt)]
    loss = auxiliary_critic_loss(list(critic(obs, alt)), y_cf, w, 0.5)
    opt.zero_grad()
    loss.backward()
    grad = float(th.sqrt(sum((p.grad ** 2).sum() for p in critic.parameters() if p.grad is not None)))
    opt.step()
    after = [q.detach() for q in critic(obs, alt)]
    gap_before = th.abs(before[0] - y_cf).flatten().numpy()
    gap_after = th.abs(after[0] - y_cf).flatten().numpy()
    per_kind: dict = defaultdict(list)
    for j, i in enumerate(idx):
        per_kind[kinds.get(int(i), "unknown")].append(j)
    out["audit"] = {
        "finite": finite, "aux_loss": float(loss), "aux_grad_norm": grad,
        "fraction_moved_towards_target": float(np.mean(gap_after < gap_before)),
        "by_kind": {
            k: {"n": len(js),
                "q_alt_minus_q_a_median": float(np.median((q_alt - q_a).flatten().numpy()[js])),
                "y_cf_minus_q_alt_median": float(np.median((y_cf - q_alt).flatten().numpy()[js])),
                "abs_gap_median_before": float(np.median(gap_before[js])),
                "abs_gap_median_after_one_step": float(np.median(gap_after[js]))}
            for k, js in per_kind.items()},
    }
    # formal-clock updates: just before and at the warm-up threshold (fresh diagnostics)
    deltas = {}
    for clock in (2_999, 3_000):
        m = M1SAC.load(run / "final.zip", device="cpu")
        m.load_replay_buffer(run / "replay_buffer.pkl")
        from stable_baselines3.common.logger import configure

        m.set_logger(configure(None, []))
        m._total_timesteps, m.num_timesteps = 20_000, clock
        m.cf_diagnostics_every = 1
        m.cf_stats = {k: [] for k in m.cf_stats}
        m._n_updates = 0
        params = th.cat([p.detach().flatten() for p in m.critic.parameters()])
        np.random.seed(0)
        th.manual_seed(0)
        m.train(gradient_steps=1, batch_size=256)
        moved = th.cat([p.detach().flatten() for p in m.critic.parameters()]) - params
        deltas[clock] = moved
        out[f"update_at_{clock}"] = {
            "lambda": m.current_lambda(), "aux_loss": m.cf_stats["aux_loss"][-1],
            "real_loss": m.cf_stats["real_loss"][-1], "valid_fraction": m.cf_stats["valid_fraction"][-1],
            "critic_change_norm": float(moved.norm()), "parameters_finite": bool(th.isfinite(moved).all()),
            "grad_norm_real": m.cf_stats["grad_norm_real"][-1] if m.cf_stats["grad_norm_real"] else None,
            "grad_norm_aux": m.cf_stats["grad_norm_aux"][-1] if m.cf_stats["grad_norm_aux"] else None,
        }
    a, b = deltas[2_999], deltas[3_000]
    out["update_difference"] = {
        "norm_of_difference": float((b - a).norm()),
        "cosine": float((a @ b) / (a.norm() * b.norm())),
    }
    return out


def reload(run: Path) -> dict:
    from stable_baselines3 import SAC

    from env.t1_gym import T1GStarEnv
    from train.t1_m1 import M1SAC

    env = T1GStarEnv(openings=[9_100_000])
    obs, _ = env.reset()
    out = {}
    final = [run / "final.zip"] if (run / "final.zip").exists() else []
    for path in sorted((run / "checkpoints").glob("*.zip")) + final:
        cls = M1SAC if "m1" in path.name or (run / "cf_stats.json").exists() else SAC
        model = cls.load(path, device="cpu")
        action, _ = model.predict(obs, deterministic=True)
        out[path.name] = {"action": np.asarray(action).round(4).tolist(), "finite": bool(np.all(np.isfinite(action)))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("stats", "offline", "reload"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"stats": stats, "offline": offline, "reload": reload}[args.command](args.run)
    text = json.dumps(result, indent=1, default=str)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
