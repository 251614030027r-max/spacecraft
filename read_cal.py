"""Summarise the decision-margin calibration runs (runsheet section 1).

Reads the evaluation JSONs written by ``experiments.evaluate_hybrid_policy``
and prints one row per file: completion, mean equivalent delta-v and mean
survival time over the *completed* episodes, and the worst truth-normalized
margin over all of them (the only comparable margin column -- see CLAUDE.md
trap 1). Defaults to ``eval/cal2``; pass a directory to read somewhere else.
"""

import glob
import json
import os
import sys

directory = sys.argv[1] if len(sys.argv) > 1 else "eval/cal2"
paths = sorted(glob.glob(os.path.join(directory, "*.json")))
if not paths:
    raise SystemExit(f"no calibration JSON in {directory!r}")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


rows = []
for path in paths:
    data = json.load(open(path, encoding="utf-8"))
    records = data["records"]
    completed = [r for r in records if r["completed"]]
    margins = [
        r["minimum_truth_normalized_margin"]
        for r in records
        if r.get("minimum_truth_normalized_margin") is not None
    ]
    rows.append(
        (
            os.path.basename(path),
            data.get("tumble_scale"),
            f"{len(completed)}/{len(records)}",
            round(mean([r["equivalent_delta_v_m_s"] for r in completed]), 3),
            round(mean([r["survival_s"] for r in completed]), 1),
            round(min(margins), 4) if margins else float("nan"),
        )
    )

print(f'{"file":26}{"tumble":8}{"complete":10}{"dv":9}{"time":8}{"worstMargin"}')
for row in rows:
    print(
        f"{row[0]:26}{str(row[1]):8}{row[2]:10}"
        f"{str(row[3]):9}{str(row[4]):8}{row[5]}"
    )
