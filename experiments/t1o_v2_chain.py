"""T1-O V2: run Gates A -> B -> C (+ zero step) -> E -> D in order, stopping at the first FAIL.

Every decision is the mechanical rule of ``experiments.t1o_v2`` (frozen in 89b779a); this file
only sequences the commands and records each stage in ``<root>/CHAIN.json``.

    python -B -m experiments.t1o_v2_chain --root logs/t1o_v2 --workers 4 [--reproduce-gate-a] [--through-d]

Default: stops before the Gate D pilot (the pilot needs the action space frozen first).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DISC = "experiments/t1o_p2_discovery_openings.json"
CONF = "experiments/t1o_p2_confirmation_openings.json"
DEV = "experiments/t1o_p2_development_openings.json"
PY = [sys.executable, "-B", "-m"]


def sh(*args: str) -> None:
    print(">>", " ".join(args), flush=True)
    subprocess.run([*PY, *args], check=True)


def stage(root: Path, chain: dict, name: str, result: dict, passed: bool) -> bool:
    chain["stages"].append({"stage": name, "pass": bool(passed), "utc": datetime.now(timezone.utc).isoformat(),
                            **result})
    (root / "CHAIN.json").write_text(json.dumps(chain, indent=1, default=float))
    print(name, "PASS" if passed else "FAIL", flush=True)
    return passed


def _rows(directory: Path, seeds: list[int]) -> dict[int, dict]:
    return {s: json.loads((directory / f"{s}.json").read_text()) for s in seeds}


def zero_step(root: Path, lam: float, rule: dict, workers: int) -> dict:
    from env.t1_o_task import t1o_review_env
    from experiments.t1o_gates import dev_subset
    from experiments.t1o_screen import run as run_candidates
    from experiments.t1o_v2 import apply_state_rule

    dev = json.loads(Path(DEV).read_text())
    seeds = dev_subset(dev, 2)
    run_dir = Path("logs/t1_o_tcpf_20261017/or3_zero")
    if not run_dir.exists():
        sh("train.train_t1_option", "--arm", "or3", "--profile", "p2", "--fuel-cost", str(lam), "--steps", "1",
           "--seed", "0", "--run-name", "or3_zero", "--extra-checkpoints", "", "--checkpoint-freq", "100000")
    zero = next((run_dir / "checkpoints").glob("*_0_decisions.zip"))
    sh("experiments.t1o_gates", "eval-model", "--checkpoint", str(zero), "--dev64", DEV, "--per-cell", "2",
       "--output-dir", str(root / "zero_step" / "0k"), "--workers", str(workers))
    picks = {}
    for s in seeds:
        env = t1o_review_env("p2")
        obs, _ = env.reset(options={"opening": s})
        picks[s] = apply_state_rule(rule, obs)
    for c in sorted(set(picks.values())):
        run_candidates("p2", c, sorted(s for s, v in picks.items() if v == c), root / "zero_step" / "rule" / c, workers)
    z = _rows(root / "zero_step" / "0k", seeds)
    r = {s: json.loads((root / "zero_step" / "rule" / picks[s] / f"{s}.json").read_text()) for s in seeds}
    key = str(lam)
    d = [r[s]["discounted_return_by_fuel_weight"][key] - z[s]["discounted_return_by_fuel_weight"][key] for s in seeds]
    cap_z = sum(z[s]["event"] == "success" for s in seeds)
    cap_r = sum(r[s]["event"] == "success" for s in seeds)
    return {"openings": len(seeds), "rule_minus_0k_mean": float(np.mean(d)), "captures_0k": cap_z,
            "captures_rule": cap_r, "picks": picks,
            "pass": bool(np.mean(d) >= 3.0 or cap_r - cap_z >= 3)}


GATE_A_CANDIDATES = ("A0", "A15", "A30", "A45", "A60", "K0", "K30", "V0", "V30", "Vmax")
UPPER_GATE_A = Path("evidence/T1O_V2/gate_a/gate_a.json")


def reproduce_gate_a(root: Path, workers: int) -> dict:
    """Run Gate A on this machine (the 160 discovery episodes) and compare it with the upper's
    committed result: the verdict, the fuel weight and the uniform candidate must agree; the
    share of bit-identical episode outcomes is reported (cross-machine numerics)."""

    import tarfile

    from experiments.t1o_screen import run as run_candidates

    seeds = json.loads(Path(DISC).read_text())["seeds"]
    for c in GATE_A_CANDIDATES:
        run_candidates("p2", c, seeds, root / "gate_a" / c, workers)
    sh("experiments.t1o_v2", "gate-a", "--openings", DISC, "--root", str(root / "gate_a"),
       "--output", str(root / "gate_a.json"))
    mine, upper = json.loads((root / "gate_a.json").read_text()), json.loads(UPPER_GATE_A.read_text())
    lam_m, lam_u = mine["fuel_weight"], upper["fuel_weight"]
    same = {"verdict": mine["verdict"] == upper["verdict"], "fuel_weight": lam_m == lam_u,
            "uniform": (lam_m is not None and lam_u is not None and
                        mine["per_weight"][str(lam_m)]["uniform_best"] == upper["per_weight"][str(lam_u)]["uniform_best"])}
    equal = total = 0
    with tarfile.open("evidence/T1O_V2/gate_a/episodes.tar.gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[0] == "gate_a" and parts[2].endswith(".json"):
                theirs = json.loads(tar.extractfile(member).read())
                ours = json.loads((root / "gate_a" / parts[1] / parts[2]).read_text())
                total += 1
                equal += all(ours[k] == theirs[k] for k in ("event", "time_s", "delta_v_m_s",
                                                            "discounted_return_by_fuel_weight"))
    return {"local": {"verdict": mine["verdict"], "fuel_weight": lam_m}, "upper": {"verdict": upper["verdict"],
            "fuel_weight": lam_u}, "agree": same, "episodes_bit_identical": f"{equal}/{total}",
            "pass": all(same.values())}


def gate_d(root: Path, lam: float, n_envs: int, workers: int) -> dict:
    run = Path("logs/t1_o_tcpf_20261017/or3_pilot_s0")
    if not (run / "finished.json").exists():
        sh("train.train_t1_option", "--arm", "or3", "--profile", "p2", "--fuel-cost", str(lam), "--steps", "12000",
           "--seed", "0", "--run-name", "or3_pilot_s0", "--n-envs", str(n_envs), "--checkpoint-freq", "3000",
           "--extra-checkpoints", "")
    from experiments.t1o_gates import dev_subset

    seeds = dev_subset(json.loads(Path(DEV).read_text()), 1)
    marks = (0, 3000, 6000, 9000, 12000)
    rows = {}
    for m in marks:
        z = next((run / "checkpoints").glob(f"*_{m}_decisions.zip"))
        sh("experiments.t1o_gates", "eval-model", "--checkpoint", str(z), "--dev64", DEV, "--per-cell", "1",
           "--output-dir", str(root / "gate_d" / str(m)), "--workers", str(workers))
        rows[m] = _rows(root / "gate_d" / str(m), seeds)
    key = str(lam)
    J = {m: [rows[m][s]["discounted_return_by_fuel_weight"][key] for s in seeds] for m in marks}
    bad = {m: sum(rows[m][s]["event"] != "success" for s in seeds) for m in marks}
    fix = json.loads((run / "fix_stats.json").read_text())
    at, q = np.asarray(fix["audit_decisions"], float), np.asarray(fix["q_minus_g_plan"], float)
    sel = (at >= 6000) & np.isfinite(q)
    qg = float(q[sel].mean()) if sel.any() else None
    gain = {m: float(np.mean(np.asarray(J[m]) - np.asarray(J[0]))) for m in marks}
    passed = (gain[9000] > 0 and gain[12000] > 0 and qg is not None and qg <= 50.0
              and bad[9000] <= bad[0] + 2 and bad[12000] <= bad[0] + 2)
    return {"mean_return": {m: float(np.mean(J[m])) for m in marks}, "gain_vs_0k": gain, "failures": bad,
            "replay_q_minus_g_plan_6k_12k": qg, "pass": bool(passed)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("logs/t1o_v2"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reproduce-gate-a", action="store_true",
                        help="run Gate A here first and require agreement with the upper's result")
    parser.add_argument("--through-d", action="store_true",
                        help="also run the Gate D pilot (only once the action space is confirmed frozen)")
    parser.add_argument("--pilot-n-envs", type=int, default=4)
    args = parser.parse_args(argv)
    root, w = args.root, args.workers
    root.mkdir(parents=True, exist_ok=True)
    chain = {"stages": []}
    if args.reproduce_gate_a:
        rep = reproduce_gate_a(root, w)
        (root / "gate_a_reproduction.json").write_text(json.dumps(rep, indent=1, default=float))
        if not stage(root, chain, "A_reproduction", rep, rep["pass"]):
            return
    a = json.loads((root / "gate_a.json").read_text())
    if not stage(root, chain, "A", {"verdict": a["verdict"], "fuel_weight": a["fuel_weight"]}, a["verdict"] == "PASS"):
        return
    lam = float(a["fuel_weight"])
    u = a["per_weight"][str(lam)]["uniform_best"]
    sh("experiments.t1o_v2", "state-rule", "--openings", DISC, "--root", str(root / "gate_a"),
       "--fuel-weight", str(lam), "--output", str(root / "state_rule.json"))
    rule = json.loads((root / "state_rule.json").read_text())
    sh("experiments.t1o_gates", "gate-b", "--openings", DISC, "--uniform", u, "--fuel-weight", str(lam),
       "--output", str(root / "gate_b.json"))
    b = json.loads((root / "gate_b.json").read_text())["summary"]
    if not stage(root, chain, "B", {"uniform": u, "in_flight": {k: b["in_flight"][k] for k in ("states", "real_effects", "preferences", "repeatability_max")},
                                    "near_field_abs_dJ_ge_1": b["near_field"]["openings_abs_dJ_ge_1"]}, b["in_flight"]["pass"]):
        return
    sh("experiments.t1o_gates", "gate-c", "--openings", CONF, "--uniform", u, "--rule", str(root / "state_rule.json"),
       "--fuel-weight", str(lam), "--root", str(root / "gate_c"), "--workers", str(w), "--output", str(root / "gate_c.json"))
    c = json.loads((root / "gate_c.json").read_text())
    if not stage(root, chain, "C", {k: c[k] for k in ("outcomes", "mean_return", "state_rule_minus_uniform",
                                                     "gstar_rule_minus_uniform", "state_rule_minus_gstar_rule")}, c["pass"]):
        return
    z = zero_step(root, lam, rule, w)
    (root / "zero_step.json").write_text(json.dumps(z, indent=1, default=float))
    if not stage(root, chain, "C_zero_step", {k: v for k, v in z.items() if k != "picks"}, z["pass"]):
        return
    sh("experiments.t1o_cf_check", "collect", "--openings", DISC, "--candidate", u, "--fuel-weight", str(lam),
       "--records", str(root / "gate_e_calibration.json"), "--workers", str(w))
    sh("experiments.t1o_cf_check", "calibrate", "--fuel-weight", str(lam), "--records",
       str(root / "gate_e_calibration.json"), "--output", str(root / "gate_e_calibrate.json"))
    e_r = json.loads((root / "gate_e_calibrate.json").read_text())["e_r_p95"]
    sh("experiments.t1o_cf_check", "collect", "--assignments", str(root / "gate_c" / "assignments.json"),
       "--fuel-weight", str(lam), "--records", str(root / "gate_e_validation.json"), "--workers", str(w))
    sh("experiments.t1o_cf_check", "validate", "--fuel-weight", str(lam), "--records",
       str(root / "gate_e_validation.json"), "--e-r", str(e_r), "--output", str(root / "gate_e.json"))
    e = json.loads((root / "gate_e.json").read_text())
    if not stage(root, chain, "E", {k: v for k, v in e.items() if k != "funnel"} | {"funnel": e["funnel"]}, e["pass"]):
        return
    if not args.through_d:
        stage(root, chain, "D_not_run", {"reason": "stopped before the pilot (--through-d not given)"}, True)
        return
    d = gate_d(root, lam, args.pilot_n_envs, w)
    (root / "gate_d.json").write_text(json.dumps(d, indent=1, default=float))
    stage(root, chain, "D", d, d["pass"])


if __name__ == "__main__":
    main()
