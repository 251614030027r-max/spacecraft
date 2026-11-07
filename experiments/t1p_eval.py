"""T1-P evaluation (research plan T1-P sec. 7): checkpoints and reference rows on fixed openings.

Deterministic, one process per episode, every episode one JSON (skipped if it exists). Returns are
the **raw** task reward (``info["reward_raw"]``), never the shaped training reward.

Rows:
    model:<path.zip>   a T1-P checkpoint, deterministic actor at reviews
    rule               the pre-declared same-interface geometric rule (``env.t1_p_review.geometric_rule``)
    pure, oldgstar     delegated to ``experiments.t1n_eval`` (same physics and initial distribution)

    python -B -m experiments.t1p_eval run --row model:<zip> --openings 300000-300047 --output-dir D --workers W
    python -B -m experiments.t1p_eval table --root eval/t1p_round1 --output table.csv
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from experiments.t1n_eval import cmd_table, parse_openings, summarise


def run_episode(job: tuple[str, int, str]) -> dict[str, Any]:
    row, opening, out_dir = job
    if row in ("pure", "oldgstar"):
        from experiments.t1n_eval import run_episode as t1n_episode

        return t1n_episode((row, opening, out_dir))
    tag = Path(row.split(":", 1)[1]).stem if row.startswith("model:") else row
    path = Path(out_dir) / f"{tag}_{opening}.json"
    if path.exists():
        return json.loads(path.read_text())
    from env.t1_p_review import T1PReviewEnv, geometric_rule

    started = perf_counter()
    env = T1PReviewEnv(openings=[opening])
    obs, _ = env.reset()
    if row.startswith("model:"):
        from train.t1_p_sac import T1PSAC

        model = T1PSAC.load(row.split(":", 1)[1], device="cpu")

        def act(o):
            a, _ = model.predict(o, deterministic=True)
            return np.asarray(a, dtype=np.float64).reshape(-1)
    elif row == "rule":
        def act(o):
            return geometric_rule(env)
    else:
        raise ValueError(row)
    steps, done, info = [], False, {}
    while not done:
        obs, r, term, trunc, info = env.step(act(obs) if env._due else env.continue_action())
        x = info["execution"]
        steps.append({"t": env.env.time_s, "r": info["reward_raw"], "r_train": r, "dv": info["delta_v_step_m_s"],
                      "mode": x["mode"], "status": x["plan_status"], "event": x["generator_event"],
                      "reviewed": info["review"]["reviewed"],
                      "a": None if not info["review"]["reviewed"] else [float(v) for v in info["executed_action"]],
                      "request": x["request"]})
        done = term or trunc
    s = info.get("episode_summary", {})
    record = {"row": row, "opening": opening, "event": info["event"], "return": float(sum(x["r"] for x in steps)),
              "time_s": float(env.env.time_s), "delta_v_m_s": float(sum(x["dv"] for x in steps)),
              "violations": s.get("violations", {}), "end_phase": s.get("end_phase"), "max_range_m": s.get("max_range_m"),
              "min_margins": s.get("min_margins"), "holds": sum(1 for x in steps if x["mode"] == "hold"),
              "first_bridge_s": next((x["t"] for x in steps if x["mode"] == "bridge"), None),
              "wall_s": perf_counter() - started, "steps": steps}
    path.write_text(json.dumps(record, default=float))
    return record


def cmd_run(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.row, k, str(out)) for k in parse_openings(args.openings)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_episode, jobs))
    print(json.dumps(summarise(rows), indent=1))


def main(argv: list[str] | None = None) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--row", required=True)
    p.add_argument("--openings", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("table")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": cmd_run, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
