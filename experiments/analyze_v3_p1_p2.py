"""Read the P1/P2 rollouts against docs/V3_P1_P2_PREREGISTRATION_20260924.md.

Input: one JSON produced by collecting every ``probe_base_policy_rollout``
output (``--collect``), or that collected file for re-analysis.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def collect(directory: Path, groups_path: Path) -> dict:
    groups = json.loads(groups_path.read_text())
    rollouts = []
    for path in sorted(directory.glob("*_K*.json")):
        model, seed, tag = path.stem.split("_")
        row = json.loads(path.read_text())
        row.update(model=int(model), group=groups[f"{model}_{seed}"], tag=tag)
        rollouts.append(row)
    return {"rollouts": rollouts}


def analyse(data: dict, baseline: dict, v2: dict[int, dict]) -> dict:
    by_case: dict[tuple[int, int], dict[str, dict]] = {}
    for row in data["rollouts"]:
        by_case.setdefault((row["model"], row["seed"]), {})[row["tag"]] = row

    # Replay fidelity.
    b_ok = sum(
        cases["K0"]["completed"] == baseline[seed]["completed"]
        and abs(cases["K0"]["survival_s"] - baseline[seed]["survival_s"]) < 1e-6
        for (model, seed), cases in by_case.items()
    )
    kinf_ok = sum(
        cases["Kinf"]["completed"] == v2[model][seed]["completed"]
        for (model, seed), cases in by_case.items()
    )
    mismatch = max(r["max_replay_mismatch"] for r in data["rollouts"])

    # P1: pairs with different outcomes within one initial state.
    concordant = total = 0
    discordant_pairs = []
    for (model, seed), cases in by_case.items():
        for a, b in itertools.combinations(cases.values(), 2):
            if a["completed"] == b["completed"]:
                continue
            total += 1
            done, other = (a, b) if a["completed"] else (b, a)
            if done["return_discounted"] > other["return_discounted"]:
                concordant += 1
            else:
                discordant_pairs.append((model, seed, done["tag"], other["tag"]))
    completed_returns = [r["return_discounted"] for r in data["rollouts"] if r["completed"]]
    timeout_returns = [r["return_discounted"] for r in data["rollouts"] if r["time_failure"]]
    distance_returns = [r["return_discounted"] for r in data["rollouts"] if r["distance_failure"]]

    # P2 per K.
    p2 = {}
    for tag in ("K1", "K5"):
        rescued_captured = 0
        rescued_captured_positive = 0
        lost = 0
        chosen_outcomes = {"R": 0, "D": 0, "T": 0}
        rows = []
        for (model, seed), cases in sorted(by_case.items()):
            group = cases["K0"]["group"]
            b, k = cases["K0"], cases[tag]
            advantage = k["return_discounted"] - b["return_discounted"]
            chosen = k if advantage > 0 else b
            chosen_outcomes[group] += int(chosen["completed"])
            if group == "R" and k["completed"] and not b["completed"]:
                rescued_captured += 1
                rescued_captured_positive += int(advantage > 0)
            if group in ("D", "T") and b["completed"] and not chosen["completed"]:
                lost += 1
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "group": group,
                    "baseline_completed": b["completed"],
                    "option_completed": k["completed"],
                    "option_time_s": k["survival_s"],
                    "baseline_time_s": b["survival_s"],
                    "advantage": advantage,
                    "chosen": "option" if advantage > 0 else "baseline",
                }
            )
        p2[tag] = {
            "rescued_captured": rescued_captured,
            "rescued_captured_with_positive_advantage": rescued_captured_positive,
            "baseline_completions_lost_by_exact_arbitration": lost,
            "exact_arbitration_completions_by_group": chosen_outcomes,
            "passes": rescued_captured_positive >= 3 and lost == 0,
            "rows": rows,
        }
    return {
        "fidelity": {
            "baseline_rollouts_matching_baseline_row": f"{b_ok}/{len(by_case)}",
            "full_replays_matching_v2_outcome": f"{kinf_ok}/{len(by_case)}",
            "max_task_state_replay_mismatch": mismatch,
        },
        "p1": {
            "discordant_outcome_pairs": total,
            "completed_side_higher_return": concordant,
            "fraction": concordant / total if total else None,
            "passes": (concordant / total >= 0.95) if total else None,
            "violating_pairs": discordant_pairs,
            "completed_return_min": min(completed_returns) if completed_returns else None,
            "timeout_return_max": max(timeout_returns) if timeout_returns else None,
            "distance_failure_return_max": max(distance_returns) if distance_returns else None,
        },
        "p2": p2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-dir", type=Path)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("eval/adp/pure_mpc_nominal.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.collect_dir is not None:
        args.rollouts.write_text(json.dumps(collect(args.collect_dir, args.groups), indent=1))
    data = json.loads(args.rollouts.read_text())
    baseline = {r["seed"]: r for r in json.loads(args.baseline.read_text())["records"]}
    v2 = {
        model: {
            r["seed"]: r
            for r in json.loads(Path(f"eval/adp/final_v2_nominal_{model}.json").read_text())["records"]
        }
        for model in {row["model"] for row in data["rollouts"]}
    }
    result = analyse(data, baseline, v2)
    args.output.write_text(json.dumps(result, indent=1))
    summary = {k: v for k, v in result.items() if k != "p2"}
    summary["p2"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in result["p2"].items()}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
