"""T1-O V2 Gate E: credibility of the MPC counterfactual against true 2 s branches.

For each episode of a fixed candidate (the frozen uniform protocol) in the counterfactual
review wrapper (``T1LReviewCFEnv``, 90 s delay range, near-field kappa alternative, the chosen
fuel weight): at every review, before the real step, two truth shadows are synced from the
real state; after the step (which computes the real and the alternative MPC predictions and
the gate) one shadow flies the alternative for 2 s in truth and the other the executed action
(its difference to the real step is the numerical repeatability). Recorded per review: kind,
the gate's reasons, the executed branch's errors e_r / e_p / e_o, predicted and true reward of
both branches, true observation / position / first-control differences.

``calibrate``: e_r tolerance = p95 of the executed-branch e_r over the calibration reviews
(``experiments.t1o_v2``: Gate A discovery trajectories). ``validate``: with that tolerance and
all other gates unchanged, on independent reviews (Gate C confirmation trajectories): accepted
pairs whose true reward difference is >= 0.10 and whose repeatability is < 1/3 of it must be
>= 20 with sign accuracy >= 80 %; far-field accepted distinct >= 10 % of far-field review
candidates; >= 20 accepted near-field kappa samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.t1o_screen import FixedCandidate

FAR_KINDS = ("delay_shift", "replan_vs_keep", "keep_vs_shift", "entry_speed")


def review_records(args) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    from env.t1_o_task import T1O_DELAY_MAX_S, t1o_env_config, t1o_review_env
    from env.t1_option_cf import ReviewCounterfactualPredictor, T1LReviewCFEnv

    profile, candidate, opening, lam = args
    cfg = t1o_env_config(profile, lam)
    env = T1LReviewCFEnv(env_config=cfg, delay_max_s=T1O_DELAY_MAX_S, near_field_kappa=True)
    env.reset(options={"opening": opening})
    policy = FixedCandidate(candidate)
    shadows = [ReviewCounterfactualPredictor(None, cfg, None, shadow_factory=lambda: t1o_review_env(profile, lam))
               for _ in range(2)]
    out = []
    while True:
        due = bool(env._due)
        action = policy.action(env)
        if due:
            for sh in shadows:
                sh.sync(env)
        obs, reward, terminated, truncated, info = env.step(action)
        if due:
            policy.executed(env, info)
            cf = info["counterfactual"]
            rec = {"opening": opening, "t": env.env.time_s - 2.0, "kind": cf["kind"], "w_original": cf["w"],
                   "reasons": cf["reasons"], "errors_real": cf["errors_real"], "r_real_true": float(reward),
                   "terminated": bool(terminated or truncated)}
            if cf["a_tilde"] is not None and cf["r_hat"] is not None and not (terminated or truncated):
                alt_true = shadows[0].closed_loop(np.asarray(cf["a_tilde"], dtype=np.float64))
                rep = shadows[1].closed_loop(np.asarray(info["executed_action"], dtype=np.float64))
                rec.update({
                    "r_hat_alt": float(cf["r_hat"]), "r_alt_true": alt_true["reward"],
                    "pred_reward_diff": float(cf["alt_reward_diff"]) if cf["alt_reward_diff"] is not None else None,
                    "true_reward_diff": float(alt_true["reward"] - reward),
                    "repeatability": abs(float(rep["reward"]) - float(reward)),
                    "true_observation_diff": float(np.linalg.norm(alt_true["observation"] - obs)),
                    "true_position_diff_m": float(np.linalg.norm(alt_true["relative_position_eci_m"]
                                                                 - (env.env.chaser_state.position - env.env.target_state.position))),
                    "pred_controls_diff_max": cf["alt_controls_diff_max"], "alt_terminated": bool(alt_true["terminated"] or alt_true["truncated"]),
                })
            out.append(rec)
        if terminated or truncated:
            return out


def collect(profile: str, candidate_by_opening: dict[int, str], lam: float, workers: int) -> list[dict]:
    import multiprocessing as mp

    jobs = [(profile, c, s, lam) for s, c in sorted(candidate_by_opening.items())]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(review_records, jobs, chunksize=1)
    return [r for p in parts for r in p]


def calibrate(records: list[dict]) -> dict:
    e = [r["errors_real"]["e_r"] for r in records if r.get("errors_real")]
    return {"reviews": len(records), "with_prediction": len(e), "e_r_p95": float(np.percentile(e, 95)) if e else None,
            "e_r_quantiles": {q: float(np.percentile(e, q)) for q in (50, 90, 95, 99)} if e else {}}


def accepted(r: dict, e_r_tol: float) -> bool:
    if r.get("pred_reward_diff") is None or not r.get("errors_real"):
        return False
    others = [x for x in r["reasons"] if x != "e_r"]
    return not others and r["errors_real"]["e_r"] <= e_r_tol


def validate(records: list[dict], e_r_tol: float) -> dict:
    acc = [r for r in records if accepted(r, e_r_tol)]
    signal = [r for r in acc if abs(r["true_reward_diff"]) >= 0.10 and r["repeatability"] < abs(r["true_reward_diff"]) / 3.0]
    sign = [np.sign(r["pred_reward_diff"]) == np.sign(r["true_reward_diff"]) for r in signal]
    err = [abs(r["pred_reward_diff"] - r["true_reward_diff"]) for r in acc]
    far_candidates = [r for r in records if r["kind"] in FAR_KINDS]
    far_acc = [r for r in acc if r["kind"] in FAR_KINDS]
    near_acc = [r for r in acc if r["kind"] == "near_kappa"]
    funnel = {}
    for kind in (*FAR_KINDS, "near_kappa"):
        rk = [r for r in records if r["kind"] == kind]
        funnel[kind] = {"candidates": len(rk),
                        "physically_distinct": sum(not any(x.startswith(("same_", "same")) for x in r["reasons"]) for r in rk),
                        "no_predicted_violation": sum(not any("predicted_violation" in x for x in r["reasons"]) for r in rk),
                        "accepted": sum(accepted(r, e_r_tol) for r in rk)}
    result = {"e_r_tolerance": e_r_tol, "reviews": len(records), "accepted": len(acc), "signal_pairs": len(signal),
              "sign_accuracy": float(np.mean(sign)) if sign else None,
              "reward_diff_error_quantiles": {q: float(np.percentile(err, q)) for q in (50, 90, 95)} if err else {},
              "far_field_candidates": len(far_candidates), "far_field_accepted": len(far_acc),
              "far_field_accepted_share": len(far_acc) / len(far_candidates) if far_candidates else None,
              "near_field_kappa_accepted": len(near_acc), "funnel": funnel}
    result["pass"] = bool(len(signal) >= 20 and (result["sign_accuracy"] or 0.0) >= 0.80
                          and (result["far_field_accepted_share"] or 0.0) >= 0.10 and len(near_acc) >= 20)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "calibrate", "validate"))
    parser.add_argument("--profile", default="p2")
    parser.add_argument("--openings", type=Path)
    parser.add_argument("--candidate", help="one candidate for every opening")
    parser.add_argument("--assignments", type=Path, help="Gate C assignments.json (uniform arm)")
    parser.add_argument("--fuel-weight", type=float, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--e-r", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "collect":
        if args.assignments:
            picks = {int(k): v for k, v in json.loads(args.assignments.read_text())["arms"]["uniform"].items()}
        else:
            picks = {s: args.candidate for s in json.loads(args.openings.read_text())["seeds"]}
        recs = collect(args.profile, picks, args.fuel_weight, args.workers)
        args.records.write_text(json.dumps(recs, default=float))
        print(len(recs), "review records")
        return
    recs = json.loads(args.records.read_text())
    result = calibrate(recs) if args.command == "calibrate" else validate(recs, float(args.e_r))
    if args.output:
        args.output.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(result, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
