"""M6: calibration check of V_L / V_B against exact rollouts (before M5).

Calibration block 262100-262111. Per seed:

    learned_full           the policy to the end; G_L(k) = its return-to-go at k
    handback at k          the policy for k decisions, then Pure MPC to the end;
                           G_B(k) = that episode's return-to-go at k
    for k in 0, 10, 20     (skipped where the learned episode ended before k)

The simulator is deterministic, so both returns at a checkpoint are exact, and
the observation at k is the same in both episodes. Reported:

1. overall: agreement of sign(mu_L - mu_B) with sign(G_L - G_B), on the
   *decisive* checkpoints (|G_L - G_B| >= 1.0, i.e. 5% of the completion
   reward) and, for the record, on all of them; MAE of each value;
   correlation between ensemble spread and absolute error;
2. critical states (exactly one branch completes): sign accuracy;
3. wrong picks: checkpoints where the M4 rule picks a branch that fails while
   the other one completes (k = 0: initial choice; k > 0: stay vs hand back).

Gate: agreement on the decisive checkpoints >= 0.80 -> proceed to M5 (items
2 and 3 are reported with the result); below -> stop and report. The tie
tolerance was fixed on 2026-09-26, before any v3d data existed: where both
branches finish with returns within 1.0 of each other (typically both
complete, 9.8794 vs 9.8801 in the smoke run) the sign is a coin toss and the
choice does not matter, so scoring it would stop the method on noise.

    python -B -m experiments.v3_calibrate_values --run-dir logs/v3d_262420 --values eval/v3d/values_262420 --output eval/v3d/m6_262420.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.v3_common import (
    always,
    learned_then_baseline,
    load_manifest,
    load_policy,
    make_env_for_run,
    parse_seed_range,
    run_episode,
)
from train.v3_values import DECISION_GAMMA, ValueEnsemble, discounted_returns, one_way_rule, sha256_file

AGREEMENT_GATE = 0.80
TIE_TOLERANCE = 1.0


def summarise(rows: list[dict], z: float) -> dict:
    if not rows:
        return {"checkpoints": 0}
    mu_l = np.array([r["mu_L"] for r in rows])
    mu_b = np.array([r["mu_B"] for r in rows])
    sd_l = np.array([r["sd_L"] for r in rows])
    sd_b = np.array([r["sd_B"] for r in rows])
    g_l = np.array([r["G_L"] for r in rows])
    g_b = np.array([r["G_B"] for r in rows])
    agree = np.sign(mu_l - mu_b) == np.sign(g_l - g_b)
    decisive = np.abs(g_l - g_b) >= TIE_TOLERANCE
    critical = np.array([r["completed_L"] != r["completed_B"] for r in rows])
    wrong = 0
    for r in rows:
        branch = one_way_rule(None if r["k"] == 0 else "learned", r["mu_L"], r["sd_L"], r["mu_B"], r["sd_B"], z)
        picked_ok = r["completed_L"] if branch == "learned" else r["completed_B"]
        other_ok = r["completed_B"] if branch == "learned" else r["completed_L"]
        wrong += int((not picked_ok) and other_ok)
    error_l = np.abs(mu_l - g_l)
    error_b = np.abs(mu_b - g_b)

    def corr(a: np.ndarray, b: np.ndarray) -> float | None:
        return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 1e-12 and b.std() > 1e-12 else None

    decisive_agreement = float(agree[decisive].mean()) if decisive.any() else None
    return {
        "checkpoints": len(rows),
        "decisive_checkpoints": int(decisive.sum()),
        "sign_agreement_decisive": decisive_agreement,
        "sign_agreement_all": float(agree.mean()),
        "mae_L": float(error_l.mean()),
        "mae_B": float(error_b.mean()),
        "corr_sd_abs_error_L": corr(sd_l, error_l),
        "corr_sd_abs_error_B": corr(sd_b, error_b),
        "critical_checkpoints": int(critical.sum()),
        "critical_sign_accuracy": float(agree[critical].mean()) if critical.any() else None,
        "wrong_picks": wrong,
        "gate": (
            "PASS"
            if decisive_agreement is None or decisive_agreement >= AGREEMENT_GATE
            else "STOP"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="final_model.zip")
    parser.add_argument("--values", type=Path, required=True, help="M3 output dir (values_L.pt, values_B.pt)")
    parser.add_argument("--seeds", default="262100-262111")
    parser.add_argument("--checkpoints", default="0,10,20")
    parser.add_argument("--z", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete-run", action="store_true", help="smoke tests only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    manifest = load_manifest(args.run_dir, require_completed=not args.allow_incomplete_run)
    policy = load_policy(args.run_dir, args.model_name)
    value_l = ValueEnsemble.load(args.values / "values_L.pt")
    value_b = ValueEnsemble.load(args.values / "values_B.pt")
    env = make_env_for_run(manifest)
    checkpoints = [int(k) for k in args.checkpoints.split(",")]
    rows: list[dict] = []
    try:
        for seed in parse_seed_range(args.seeds):
            learned = run_episode(env, policy, seed, always("learned"))
            g_learned = discounted_returns(learned["rewards"], DECISION_GAMMA)
            for k in checkpoints:
                if k >= len(learned["rewards"]):
                    continue
                handback = run_episode(env, policy, seed, learned_then_baseline(k))
                if not np.array_equal(handback["observations"][k], learned["observations"][k]):
                    raise RuntimeError("the handback prefix diverged from the learned episode")
                g_handback = discounted_returns(handback["rewards"], DECISION_GAMMA)
                observation = learned["observations"][k]
                mu_l, sd_l = value_l.mean_std(observation)
                mu_b, sd_b = value_b.mean_std(observation)
                rows.append(
                    {
                        "seed": seed,
                        "k": k,
                        "mu_L": mu_l,
                        "sd_L": sd_l,
                        "mu_B": mu_b,
                        "sd_B": sd_b,
                        "G_L": float(g_learned[k]),
                        "G_B": float(g_handback[k]),
                        "completed_L": learned["completed"],
                        "completed_B": handback["completed"],
                        "failure_L": learned["failure"],
                        "failure_B": handback["failure"],
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    result = {
        "run_dir": str(args.run_dir),
        "model_name": args.model_name,
        "code_commit": manifest.get("code_commit"),
        "values_L_sha256": sha256_file(args.values / "values_L.pt"),
        "values_B_sha256": sha256_file(args.values / "values_B.pt"),
        "seeds": args.seeds,
        "z": args.z,
        "agreement_gate": AGREEMENT_GATE,
        "tie_tolerance": TIE_TOLERANCE,
        "summary": summarise(rows, args.z),
        "by_checkpoint": {str(k): summarise([r for r in rows if r["k"] == k], args.z) for k in checkpoints},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
