"""Review readout of T1_V2_S0 (M1 seed 0, interface v2, G* v0.4): the numbers of the upper review.

    # 1) evidence from the unpacked lower package (collab 13cbf38, T1_V2_S0_FINAL_20261013.zip)
    python -B -m experiments.t1_v2_review pack --package <unzipped dir> --output evidence/T1_V2_S0
    # 2) readout from the evidence only
    python -B -m experiments.t1_v2_review readout --evidence evidence/T1_V2_S0 \
        --r1 evidence/T1_R1_pass2/episodes.jsonl.gz --output evidence/T1_V2_S0/review_readout.json
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TIMING_ROWS = ("rv:0:0.7", "rv:50:0.7", "rv:70:0.7", "rv:90:0.7", "rvbest:0.7", "rvgeo:0.7")


def _gz_lines(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_gz(path: Path, rows: list[dict]) -> None:
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        for r in rows:
            f.write((json.dumps(r) + "\n").encode())


def pack(package: Path, output: Path) -> None:
    run = package / "run"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "finished.json", "stopped.json", "stats_final.json", "reload.json",
                 "episodes.jsonl.gz", "decisions.jsonl.gz"):
        shutil.copyfile(run / name, output / name)
    shutil.copyfile(run / "STOP_REQUESTED", output / "STOP_REQUESTED.json")
    shutil.copyfile(package / "MODEL_SHA256.json", output / "MODEL_SHA256.json")
    with open(run / "cf_stats.json", "rb") as src, gzip.GzipFile(output / "cf_stats.json.gz", "wb", mtime=0) as dst:
        dst.write(src.read())
    lengths = {int(p.name.split("_")[1]): len(json.loads(p.read_text())["real_loss"])
               for p in (run / "checkpoints").glob("m1_*_cf_stats.json")}
    (output / "cf_checkpoint_lengths.json").write_text(json.dumps(dict(sorted(lengths.items()))))
    rows, summaries = [], {}
    for d in sorted((run / "dev_eval").iterdir(), key=lambda p: int(p.name)):
        summaries[int(d.name)] = json.loads((d / "summary.json").read_text())
        rows += [{"mark": int(d.name), **json.loads(f.read_text())} for f in sorted(d.glob("2*.json"))]
    _write_gz(output / "dev_eval.jsonl.gz", rows)
    (output / "dev_summaries.json").write_text(json.dumps(summaries, indent=1))


def _arrival_clock(log: list[dict]) -> list[float]:
    return [2.0 * k + x["tau_effective_s"] for k, x in enumerate(log)]


def readout(evidence: Path, r1: Path) -> dict:
    dev = _gz_lines(evidence / "dev_eval.jsonl.gz")
    out: dict = {"checkpoints": {}, "failing_episode_arrival_clock_s": [], "training_outcomes_by_4k": {},
                 "critic_td_loss_by_checkpoint_segment": {}, "timing_headroom_r1": {}}
    by_mark = defaultdict(list)
    for r in dev:
        by_mark[r["mark"]].append(r)
    for mark, rows in sorted(by_mark.items()):
        status, later, earlier = Counter(), 0, 0
        for r in rows:
            log = r["decision_log"]
            clock = _arrival_clock(log)
            for k in range(1, len(log)):
                status[log[k]["status"]] += 1
                if log[k]["status"].startswith(("optimal", "recovery")):
                    later += clock[k] - clock[k - 1] > 0.5
                    earlier += clock[k] - clock[k - 1] < -0.5
            if r["event"] != "success":
                out["failing_episode_arrival_clock_s"].append({
                    "mark": mark, "opening": r["opening"], "event": r["event"], "end_phase": r["end_phase"],
                    "end_range_m": round(r["end_range_m"], 2), "new_plans": r["new_plans"],
                    "first": round(clock[0], 1), "min": round(min(clock), 1), "last": round(clock[-1], 1),
                    "every_10th": [round(c) for c in clock[::10]]})
        out["checkpoints"][mark] = {
            "clean_capture": sum(r["event"] == "success" for r in rows),
            "mean_discounted_return": float(np.mean([r["discounted_return"] for r in rows])),
            "failures": dict(Counter(r["event"] for r in rows if r["event"] != "success")),
            "first_tau_request_s_mean": float(np.mean([r["first_tau_request_s"] for r in rows])),
            "kappa_mean": float(np.mean([r["kappa_mean"] for r in rows])),
            "status_after_first_decision": dict(status), "replans_later": int(later), "replans_earlier": int(earlier)}
    episodes = _gz_lines(evidence / "episodes.jsonl.gz")
    for lo in range(0, max(e["num_timesteps"] for e in episodes) + 1, 4000):
        chunk = [e for e in episodes if lo <= e["num_timesteps"] < lo + 4000]
        if chunk:
            out["training_outcomes_by_4k"][lo] = dict(Counter(e["event"] for e in chunk)) | {"n": len(chunk)}
    with gzip.open(evidence / "cf_stats.json.gz", "rt") as f:
        real = np.array(json.load(f)["real_loss"])
    start = 0
    for mark, end in sorted((int(k), v) for k, v in json.loads((evidence / "cf_checkpoint_lengths.json").read_text()).items()):
        seg = real[start:end]
        out["critic_td_loss_by_checkpoint_segment"][mark] = {
            "updates": int(len(seg)), "mean": float(seg.mean()), "median": float(np.median(seg)),
            "note": "SB3 critic loss = 0.5 * sum over the two critics of the TD MSE"}
        start = end
    rows = _gz_lines(r1)
    by_seed = defaultdict(dict)
    for e in rows:
        by_seed[e["seed"]][e["row"]] = e["discounted_return"]
    per_rule = {row: float(np.mean([v[row] for v in by_seed.values()])) for row in TIMING_ROWS}
    oracle = float(np.mean([max(v[row] for row in TIMING_ROWS) for v in by_seed.values()]))
    out["timing_headroom_r1"] = {
        "openings": len(by_seed), "kappa": 0.7, "mean_return_per_rule": per_rule,
        "best_single_rule": max(per_rule, key=per_rule.get),
        "per_opening_best_of_six_rules": oracle, "headroom_over_best_single_rule": oracle - max(per_rule.values())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pack", "readout"))
    parser.add_argument("--package", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--r1", type=Path, default=Path("evidence/T1_R1_pass2/episodes.jsonl.gz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.package, args.output)
        return
    text = json.dumps(readout(args.evidence, args.r1), indent=1)
    args.output.write_text(text)
    print(hashlib.sha256(text.encode()).hexdigest())


if __name__ == "__main__":
    main()
