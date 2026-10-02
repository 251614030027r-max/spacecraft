"""Read P2' against docs/V3_P2PRIME_PREREGISTRATION_20260924.md.

Selection rule under test: at state s choose argmax{V^learned(s), V_b(s)},
each policy continuing to the end, compared on the decision-level discounted
return (gamma 0.99). Every rollout shares the recorded prefix up to the
checkpoint, so whole-episode returns compare exactly as returns-to-go.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--p2-rollouts", type=Path, default=Path("eval/adp/v3_p2_rollouts.json"))
    parser.add_argument("--collected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = json.loads(args.groups.read_text())
    rows = []
    for path in sorted(args.dir.glob("*_K*.json")):
        row = json.loads(path.read_text())
        head, tag = path.stem.rsplit("_", 1)
        row["tag"] = tag
        row["model"] = None if head.startswith("base") else int(head.split("_")[0])
        rows.append(row)
    args.collected.write_text(json.dumps({"rollouts": rows}, indent=1))

    base = {r["seed"]: r for r in rows if r["model"] is None}
    learned = {(r["model"], r["seed"]): r for r in rows if r["tag"] == "Kinf"}

    # Group A: protective side on fresh episodes, choice at s_0.
    lost, kept_learned_better, wrong, a_rows = 0, 0, 0, []
    for (model, seed), lrow in sorted(learned.items()):
        group = groups[f"{model}_{seed}"]
        b = base[seed]
        pick = lrow if lrow["return_discounted"] > b["return_discounted"] else b
        is_lost = b["completed"] and not pick["completed"]
        lost += int(is_lost)
        wrong += int((lrow["completed"] or b["completed"]) and not pick["completed"])
        kept_learned_better += int(pick is lrow)
        a_rows.append(
            {
                "model": model,
                "seed": seed,
                "group": group,
                "baseline_completed": b["completed"],
                "learned_completed": lrow["completed"],
                "baseline_return": b["return_discounted"],
                "learned_return": lrow["return_discounted"],
                "picked": "learned" if pick is lrow else "baseline",
                "lost_baseline_completion": is_lost,
            }
        )
    a_total = len(a_rows)
    retained_rows = [r for r in a_rows if r["group"] == "T"]

    # Group B: rescue side at new checkpoints, learned continuation from P2.
    p2 = json.loads(args.p2_rollouts.read_text())["rollouts"]
    p2_learned = {(r["model"], r["seed"]): r for r in p2 if r["tag"] == "Kinf" and r["group"] == "R"}
    b_result = {}
    for tag in ("K10", "K20"):
        kept, detail = 0, []
        for r in rows:
            if r["tag"] != tag:
                continue
            lrow = p2_learned[(r["model"], r["seed"])]
            pick = lrow if lrow["return_discounted"] > r["return_discounted"] else r
            kept += int(pick["completed"])
            detail.append(
                {
                    "model": r["model"],
                    "seed": r["seed"],
                    "handback_completed": r["completed"],
                    "handback_return": r["return_discounted"],
                    "handback_failure": "distance" if r["distance_failure"] else ("time" if r["time_failure"] else ("none" if r["completed"] else "hard_constraint")),
                    "learned_return": lrow["return_discounted"],
                    "picked": "learned" if pick is lrow else "handback",
                }
            )
        b_result[tag] = {"rescues_kept": kept, "of": len(detail), "rows": detail}

    result = {
        "group_A": {
            "episodes": a_total,
            "baseline_completions_lost": lost,
            "passes": lost <= 1,
            "picked_learned": kept_learned_better,
            "retained_where_learned_scored_higher": sum(r["picked"] == "learned" for r in retained_rows),
            "retained_total": len(retained_rows),
            "wrong_picks_any": wrong,
            "rows": a_rows,
        },
        "group_B": {
            **b_result,
            "passes": all(v["rescues_kept"] >= 7 for v in b_result.values()),
        },
    }
    result["passes"] = result["group_A"]["passes"] and result["group_B"]["passes"]
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(
        {
            "group_A": {k: v for k, v in result["group_A"].items() if k != "rows"},
            "group_B": {k: ({kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v) for k, v in result["group_B"].items()},
            "passes": result["passes"],
        },
        indent=1,
    ))


if __name__ == "__main__":
    main()
