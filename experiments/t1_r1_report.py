"""Descriptive R1 tables, beside the pre-registered readout (which is not changed).

For every row over the full opening block: clean-capture rate with a Wilson
95 % interval, failures by category, time with a non-capture counted as the
300 s horizon, delta-v and discounted return over all episodes, the paired
difference to Pure on the same openings (bootstrap 95 % interval, fixed
resampling seed), minimum truth margins, and how the G* plan was used
(decisions with a new plan / kept plan / descent / hold, plan solve time).

    python -B -m experiments.t1_r1_report --input-dir R1DIR --openings R1DIR/openings.json --output R1DIR/report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.t1_r1_readout import HORIZON_S, capture_time, load


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def bootstrap_mean(values: np.ndarray, draws: int = 10_000, seed: int = 20261011) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def plan_usage(results: list[dict]) -> dict:
    counts: Counter = Counter()
    walls, wall_max = [], 0.0
    for r in results:
        for t in r.get("trace", []):
            status = str(t.get("plan_status", ""))
            if not status:
                continue
            if status.startswith("optimal"):
                counts["new_plan"] += 1
            elif status.startswith("hold"):
                counts["hold"] += 1
            elif status.startswith("go_now"):
                counts["go_now"] += 1
            else:
                counts[status.split(":")[0]] += 1
        if "plan_wall_s_total" in r:
            walls.append(r["plan_wall_s_total"])
            wall_max = max(wall_max, r["plan_wall_s_max"])
    total = sum(counts.values())
    return {
        "decisions": total,
        "fractions": {k: v / total for k, v in sorted(counts.items())} if total else {},
        "episodes_with_hold": sum(any(str(t.get("plan_status", "")).startswith("hold") for t in r.get("trace", []))
                                  for r in results),
        "plan_wall_s_per_episode_median": float(np.median(walls)) if walls else None,
        "plan_wall_s_max_single": wall_max if walls else None,
    }


def report(rows: dict[str, dict[int, dict]], seeds: list[int]) -> dict:
    out = {}
    pure = rows.get("pure", {})
    for tag in sorted(rows):
        results = [rows[tag][s] for s in seeds if s in rows[tag]]
        n = len(results)
        k = sum(r["event"] == "success" for r in results)
        failures = Counter()
        for r in results:
            if r["event"] == "success":
                continue
            if r["event"] == "unsafe":
                for key in (r.get("violations") or {"unknown": 0}):
                    failures[f"unsafe:{key}"] += 1
            else:
                failures[r["event"]] += 1
        times = np.array([capture_time(r) for r in results])
        dv = np.array([r["delta_v_m_s"] for r in results])
        J = np.array([r["discounted_return"] for r in results])
        margins: dict[str, float] = {}
        for r in results:
            for key, value in r["min_margins"].items():
                margins[key] = min(margins.get(key, np.inf), value)
        row = {
            "episodes": n,
            "clean_capture": k,
            "clean_capture_rate": k / n if n else None,
            "wilson95": wilson(k, n),
            "failures": dict(failures),
            "time_s_fail_as_300": {"mean": float(times.mean()), "median": float(np.median(times))},
            "delta_v_m_s_all": {"mean": float(dv.mean()), "median": float(np.median(dv))},
            "J_all": {"mean": float(J.mean()), "median": float(np.median(J))},
            "min_truth_margins": margins,
            "qp_zero_fallback_steps": int(sum(r["qp_zero_fallback_steps"] for r in results)),
            "force_saturated_fraction_median": float(np.median([r["force_saturated_fraction"] for r in results])),
            "plan_usage": plan_usage(results),
        }
        common = [s for s in seeds if s in rows[tag] and s in pure]
        if tag != "pure" and common:
            dT = np.array([capture_time(pure[s]) - capture_time(rows[tag][s]) for s in common])
            dJ = np.array([rows[tag][s]["discounted_return"] - pure[s]["discounted_return"] for s in common])
            ddv = np.array([pure[s]["delta_v_m_s"] - rows[tag][s]["delta_v_m_s"] for s in common])
            both = [s for s in common if pure[s]["event"] == "success" and rows[tag][s]["event"] == "success"]
            row["paired_vs_pure"] = {
                "openings": len(common),
                "time_saved_s_fail_as_300": bootstrap_mean(dT),
                "J_gain": bootstrap_mean(dJ),
                "delta_v_saved_m_s": bootstrap_mean(ddv),
                "captured_by_row_not_pure": sum(rows[tag][s]["event"] == "success" and pure[s]["event"] != "success"
                                                for s in common),
                "captured_by_pure_not_row": sum(pure[s]["event"] == "success" and rows[tag][s]["event"] != "success"
                                                for s in common),
                "both_captured": len(both),
                "time_saved_s_both_captured_median": float(np.median(
                    [pure[s]["time_s"] - rows[tag][s]["time_s"] for s in both])) if both else None,
            }
        out[tag] = row
    return out


def capture_curve(rows: dict[str, dict[int, dict]], seeds: list[int], step_s: float = 10.0) -> dict:
    grid = np.arange(0.0, HORIZON_S + 1e-9, step_s)
    return {
        "t_s": grid.tolist(),
        **{tag: [float(np.mean([rows[tag][s]["event"] == "success" and rows[tag][s]["time_s"] <= t
                                for s in seeds if s in rows[tag]])) for t in grid] for tag in sorted(rows)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = json.loads(args.openings.read_text())["seeds"]
    rows = load(args.input_dir, seeds)
    result = {"rows": report(rows, seeds), "cumulative_clean_capture": capture_curve(rows, seeds)}
    if args.output:
        args.output.write_text(json.dumps(result, indent=1))
    for tag, row in result["rows"].items():
        paired = row.get("paired_vs_pure", {})
        lo, hi = row["wilson95"]
        print(f"{tag:12s} {row['clean_capture']:2d}/{row['episodes']} [{lo:.2f},{hi:.2f}] "
              f"T300 mean {row['time_s_fail_as_300']['mean']:.1f} dv {row['delta_v_m_s_all']['mean']:.2f} "
              f"J {row['J_all']['mean']:.1f} fail {row['failures']} "
              + (f"| vs Pure dT {paired['time_saved_s_fail_as_300'][0]:.1f} "
                 f"[{paired['time_saved_s_fail_as_300'][1]:.1f},{paired['time_saved_s_fail_as_300'][2]:.1f}] "
                 f"+{paired['captured_by_row_not_pure']}/-{paired['captured_by_pure_not_row']}" if paired else ""))


if __name__ == "__main__":
    main()
