"""T1-N evaluation (Research Plan R2 sec. 8): checkpoints and non-learning rows on fixed openings.

Deterministic, one process per episode (episodes are independent and give the same result
serially or in a pool). Every episode writes one JSON (skipped if it exists, so an interrupted
evaluation resumes without re-running finished episodes); ``table`` reads them back.

Rows:
    model:<path.zip>      a T1-N checkpoint, deterministic actor at reviews
    earliest:<k>          conditional G*, request the observed earliest feasible arrival + k s
    fixed:<a>             conditional G*, a fixed neutral request a at no-plan reviews, keep plans
    oldgstar              the T1 v0.6 G* with its own earliest search (full G*-earliest reference)
    pure                  Pure MPC (fixed terminal setpoint), T1-N initial states

    python -B -m experiments.t1n_eval run --row model:logs/t1_n_r2/tcpf_s0/checkpoints/tcpf_40000_decisions.zip \
        --openings 300000-300047 --output-dir eval/t1n/tcpf_s0_40000 --workers 12
    python -B -m experiments.t1n_eval table --root eval/t1n --output table.csv

Fixed-rule selection (run order step 3, before any formal evaluation): every row of
``RULE_FAMILY`` on the development openings 292200-292231, then ``select-rule`` picks the largest
mean return (ties: the first in ``RULE_FAMILY``) and writes ``selected_rule.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


RULE_FAMILY = ("fixed:-0.5", "fixed:-0.25", "fixed:0", "fixed:0.25", "fixed:0.5",
               "earliest:0", "earliest:15", "earliest:30", "earliest:45")
RULE_DEVELOPMENT_OPENINGS = "292200-292231"


def parse_openings(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _finish(env, steps, info, row: str, opening: int, started: float) -> dict[str, Any]:
    summary = info.get("episode_summary", {})
    return {"row": row, "opening": opening, "event": info["event"], "return": float(sum(s["r"] for s in steps)),
            "return_by_fuel_weight": None, "time_s": float(env.time_s), "delta_v_m_s": float(sum(s["dv"] for s in steps)),
            "violations": summary.get("violations", {}), "end_phase": summary.get("end_phase"),
            "max_range_m": summary.get("max_range_m"), "min_margins": summary.get("min_margins"),
            "force_saturated_fraction": summary.get("force_saturated_fraction"),
            "rejections": sum(1 for s in steps if s.get("outcome") == "rejected" and s.get("reviewed")),
            "holds": sum(1 for s in steps if str(s.get("status", "")).startswith("hold")),
            "first_tau_request_s": next((s["tau_req"] for s in steps if s.get("reviewed")), None),
            "first_tau_effective_s": next((s["tau_eff"] for s in steps if str(s.get("status", "")).startswith("optimal")), None),
            "wall_s": perf_counter() - started, "steps": steps}


def run_episode(job: tuple[str, int, str]) -> dict[str, Any]:
    row, opening, out_dir = job
    name = row.split(":", 1)[0] if row.startswith("model:") else row.replace(":", "_")
    tag = Path(row.split(":", 1)[1]).stem if row.startswith("model:") else name
    path = Path(out_dir) / f"{tag}_{opening}.json"
    if path.exists():
        return json.loads(path.read_text())
    started = perf_counter()
    steps: list[dict[str, Any]] = []
    if row == "pure":
        from env.t1_env import T1Env
        from env.t1_n_task import t1n_env_config

        env = T1Env(t1n_env_config())
        env.reset(opening)
        done, info = False, {}
        while not done:
            reward, term, trunc, info = env.step(pure=True)
            steps.append({"t": env.time_s, "r": reward, "dv": info["delta_v_step_m_s"]})
            done = term or trunc
        if done:
            info["episode_summary"] = {**env.summary(), "violations": dict(info.get("violations", {}))}
        record = _finish(env, steps, info, row, opening, started)
    elif row == "oldgstar":
        from experiments.t1n_check import run_episode as check_episode

        r = check_episode((opening, "oldgstar", str(out_dir)))
        return r
    else:
        from env.t1_n_review import T1NReviewEnv

        env = T1NReviewEnv(openings=[opening])
        obs, _ = env.reset()
        if row.startswith("model:"):
            from train.t1_n_sac import T1NSAC

            model = T1NSAC.load(row.split(":", 1)[1], device="cpu")

            def act(o):
                a, _ = model.predict(o, deterministic=True)
                return np.asarray(a, dtype=np.float32).reshape(-1)
        else:
            from experiments.t1n_check import _policy

            act = _policy(row, env)
        done, info = False, {}
        while not done:
            obs, reward, term, trunc, info = env.step(act(obs) if env._due else env.continue_action())
            x, rv = info["execution"], info["review"]
            steps.append({"t": env.env.time_s, "r": reward, "dv": info["delta_v_step_m_s"], "status": x["plan_status"],
                          "tau_req": x["tau_request_s"], "tau_eff": x["tau_effective_s"], "reviewed": rv["reviewed"],
                          "outcome": rv.get("outcome"), "a": None if not rv["reviewed"] else float(info["executed_action"][0])})
            done = term or trunc
        record = _finish(env.env, steps, info, row, opening, started)
    path.write_text(json.dumps(record, default=float))
    return record


def cmd_run(args) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(args.row, k, str(out)) for k in parse_openings(args.openings)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_episode, jobs))
    print(json.dumps(summarise(rows), indent=1))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ret = np.array([r["return"] for r in rows])
    return {"n": len(rows), "success": sum(r["event"] == "success" for r in rows),
            "unsafe": sum(r["event"] == "unsafe" for r in rows), "timeout": sum(r["event"] == "timeout" for r in rows),
            "return_mean": float(ret.mean()) if ret.size else None,
            "time_mean_s": float(np.mean([r["time_s"] for r in rows])) if rows else None,
            "delta_v_mean_m_s": float(np.mean([r["delta_v_m_s"] for r in rows])) if rows else None,
            "wall_s_mean": float(np.mean([r["wall_s"] for r in rows])) if rows else None}


def cmd_table(args) -> None:
    lines = []
    for d in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        rows = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
        if rows:
            lines.append({"dir": d.name, **summarise(rows)})
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lines[0]))
        writer.writeheader()
        writer.writerows(lines)
    print(json.dumps(lines, indent=1))


def cmd_select_rule(args) -> None:
    root = Path(args.root)
    expected = parse_openings(RULE_DEVELOPMENT_OPENINGS)
    table = []
    for rule in RULE_FAMILY:
        d = root / rule.replace(":", "_")
        rows = [json.loads((d / f"{rule.replace(':', '_')}_{k}.json").read_text()) for k in expected]
        table.append({"rule": rule, **summarise(rows)})
    best = max(table, key=lambda r: (r["return_mean"], -RULE_FAMILY.index(r["rule"])))
    out = {"openings": RULE_DEVELOPMENT_OPENINGS, "family": list(RULE_FAMILY),
           "criterion": "largest mean return; ties: first in the family order", "table": table,
           "selected": best["rule"]}
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


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
    p = sub.add_parser("select-rule")
    p.add_argument("--root", required=True, help="directory with one sub-directory per rule (rule name, ':' -> '_')")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"run": cmd_run, "table": cmd_table, "select-rule": cmd_select_rule}[args.cmd](args)


if __name__ == "__main__":
    main()
