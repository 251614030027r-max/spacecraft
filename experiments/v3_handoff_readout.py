"""Stage B readout: fidelity checks, handoff windows, and the preregistered gate.

Applies ``docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md`` to the
per-seed files of ``experiments.v3_handoff_scan``. Nothing here is tuned: the
definitions and the gate are fixed in that document and in the constants
below, committed before any stage-B data exists.

Definitions (per learned failure trajectory, i.e. learned_full not a clean
completion):

    K_rescue   {k : handoff at decision k ends in a clean completion}
               (completed with zero truth-violation steps over the whole episode)
    run        a maximal set of consecutive decisions inside K_rescue
    W          the length of the longest run, in decisions (x 2 s)
    rescuable  K_rescue non-empty
    non-degenerate   W >= 2 (at least two consecutive decisions, 4 s)
    new capability   Pure MPC from reset (k = 0) not a clean completion,
                     learned_full not a clean completion, some k >= 1 in K_rescue

Gate, per model (both blocks pooled): at least MIN_NONDEGENERATE failure
trajectories are non-degenerate, AND non-degenerate trajectories are at least
half of the rescuable ones (the phenomenon is not mainly single-decision
needles). Verdict: FIDELITY_FAIL if any fidelity check fails (no mechanism
reading at all); else PROCEED to stage C if at least MIN_MODELS models pass;
else STOP (return to the policy / interface level, no handoff estimator).

    python -B -m experiments.v3_handoff_readout --scan 262420=eval/v3e/stage_b/262420 ... --formal-learned 262420=.../learned_only.json ... --formal-pure .../pure_mpc.json --m2 262420=.../m2_a.json,.../m2_b.json,... --seeds 262000-262047,270000-270047 --output eval/v3e/stage_b/readout.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from experiments.v3_common import parse_seed_range

DECISION_PERIOD_S = 2.0
NONDEGENERATE_MIN_W = 2
MIN_NONDEGENERATE = 2
MIN_MODELS = 2


# -- windows -----------------------------------------------------------------


def runs_of(ks: Iterable[int], stride: int = 1) -> list[tuple[int, int]]:
    """Maximal runs of consecutive scanned decisions: [(first_k, last_k), ...]."""

    runs: list[tuple[int, int]] = []
    for k in sorted(set(ks)):
        if runs and k - runs[-1][1] == stride:
            runs[-1] = (runs[-1][0], k)
        else:
            runs.append((k, k))
    return runs


def run_length(run: tuple[int, int], stride: int = 1) -> int:
    return (run[1] - run[0]) // stride + 1


def failure_trajectory(scan: dict[str, Any]) -> dict[str, Any]:
    """Window summary of one scanned learned failure (stride must be 1)."""

    if scan["stride"] != 1:
        raise ValueError("gate trajectories must be scanned at every decision")
    handoffs = scan["handoffs"]
    learned = scan["learned_full"]
    if [row["k"] for row in handoffs] != list(range(learned["decisions"])):
        raise ValueError(f"seed {scan['seed']}: incomplete scan")
    rescue = [row["k"] for row in handoffs if row["clean_completion"]]
    runs = runs_of(rescue)
    widths = [run_length(run) for run in runs]
    w_max = max(widths, default=0)
    k0_clean = bool(handoffs[0]["clean_completion"])
    return {
        "seed": int(scan["seed"]),
        "learned_decisions": int(learned["decisions"]),
        "learned_failure": learned["failure"],
        "learned_completed_with_violation": bool(
            learned["completed"] and not learned["zero_violation"]
        ),
        "pure_from_reset_clean": k0_clean,
        "k_rescue_count": len(rescue),
        "runs": [
            {
                "first_k": first,
                "last_k": last,
                "width_decisions": run_length((first, last)),
                "width_s": run_length((first, last)) * DECISION_PERIOD_S,
            }
            for first, last in runs
        ],
        "w_max_decisions": w_max,
        "w_max_s": w_max * DECISION_PERIOD_S,
        "earliest_k": min(rescue) if rescue else None,
        "latest_k": max(rescue) if rescue else None,
        "rescuable": bool(rescue),
        "non_degenerate": w_max >= NONDEGENERATE_MIN_W,
        "interior_window": any(first >= 1 for first, _ in runs),
        "new_capability": (not k0_clean) and any(k >= 1 for k in rescue),
    }


def model_gate(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    rescuable = sum(t["rescuable"] for t in trajectories)
    non_degenerate = sum(t["non_degenerate"] for t in trajectories)
    needles_only = rescuable - non_degenerate
    widths = sorted(t["w_max_decisions"] for t in trajectories)
    return {
        "failures": len(trajectories),
        "rescuable": rescuable,
        "rescuable_fraction": rescuable / len(trajectories) if trajectories else None,
        "non_degenerate": non_degenerate,
        "needle_only": needles_only,
        "interior_window": sum(t["interior_window"] for t in trajectories),
        "new_capability": sum(t["new_capability"] for t in trajectories),
        "w_max_distribution_decisions": widths,
        "passes": non_degenerate >= MIN_NONDEGENERATE and non_degenerate >= needles_only,
        # Stage-C accounting: raw critical states vs independent units.
        "critical_states_raw": sum(t["k_rescue_count"] for t in trajectories),
        "independent_windows": sum(len(t["runs"]) for t in trajectories),
        "trajectories_with_window": rescuable,
    }


def verdict(fidelity_ok: bool, gates: dict[str, dict[str, Any]]) -> str:
    if not fidelity_ok:
        return "FIDELITY_FAIL"
    passing = sum(gate["passes"] for gate in gates.values())
    return "PROCEED" if passing >= MIN_MODELS else "STOP"


# -- B2: successes, stage-C accounting only (no gate) --------------------------


def success_trajectory(scan: dict[str, Any]) -> dict[str, Any]:
    stride = int(scan["stride"])
    learned = scan["learned_full"]
    handoffs = scan["handoffs"]
    destroy = [row["k"] for row in handoffs if not row["clean_completion"]]
    faster = [
        row for row in handoffs
        if row["k"] >= 1 and row["clean_completion"] and row["survival_s"] < learned["survival_s"]
    ]
    best = min(faster, key=lambda row: row["survival_s"]) if faster else None
    return {
        "seed": int(scan["seed"]),
        "stride": stride,
        "scanned": len(handoffs),
        "destroy_count": len(destroy),
        "destroy_runs": len(runs_of(destroy, stride)),
        "earlier_clean_handoff_is_faster": best is not None,
        "best_time_saving_s": (learned["survival_s"] - best["survival_s"]) if best else None,
        "delta_v_change_at_best_m_s": (
            best["equivalent_delta_v_m_s"] - learned["equivalent_delta_v_m_s"]
        ) if best else None,
    }


def b2_summary(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    savings = [t["best_time_saving_s"] for t in trajectories if t["best_time_saving_s"] is not None]
    dv = [t["delta_v_change_at_best_m_s"] for t in trajectories if t["delta_v_change_at_best_m_s"] is not None]
    return {
        "successes": len(trajectories),
        "with_destroying_handoff": sum(t["destroy_count"] > 0 for t in trajectories),
        "destroy_states_raw": sum(t["destroy_count"] for t in trajectories),
        "independent_destroy_runs": sum(t["destroy_runs"] for t in trajectories),
        "with_faster_clean_handoff": len(savings),
        "median_best_time_saving_s": median(savings) if savings else None,
        "median_delta_v_change_at_best_m_s": median(dv) if dv else None,
    }


#: B2-lite (amendment of 2026-09-30, committed before any B1 result): one block,
#: learned clean successes only, a handoff every B2_STRIDE decisions.
B2_SEEDS = "262000-262047"
B2_STRIDE = 10


def b2_fidelity_checks(
    b2_scans: dict[str, dict[int, dict[str, Any]]],
    b1_scans: dict[str, dict[int, dict[str, Any]]],
    expected_seeds: list[int],
) -> list[str]:
    """B2 files: complete, one clean commit, the B1 model, stride, verified, and
    the learned episode bitwise equal to the B1 rerun of the same seed."""

    problems: list[str] = []
    for model, by_seed in b2_scans.items():
        missing = sorted(set(expected_seeds) - set(by_seed))
        if missing:
            problems.append(f"B2 {model}: missing seeds {missing}")
        extra = sorted(set(by_seed) - set(expected_seeds))
        if extra:
            problems.append(f"B2 {model}: seeds outside the B2 block {extra}")
        commits = {scan.get("code_commit") for scan in by_seed.values()}
        if len(commits) != 1 or None in commits:
            problems.append(f"B2 {model}: scans come from {len(commits)} code commits ({commits})")
        if any(scan.get("code_dirty") for scan in by_seed.values()):
            problems.append(f"B2 {model}: a scan ran on a dirty checkout")
        b1 = b1_scans.get(model, {})
        b1_models = {scan.get("model_sha256") for scan in b1.values()}
        for seed, scan in sorted(by_seed.items()):
            if scan.get("scan") != "successes" or scan.get("stride") != B2_STRIDE:
                problems.append(f"B2 {model}/{seed}: not a successes scan at stride {B2_STRIDE}")
            if scan.get("max_decisions") is not None:
                problems.append(f"B2 {model}/{seed}: truncated scan")
            if scan.get("model_sha256") not in b1_models:
                problems.append(f"B2 {model}/{seed}: model file differs from B1")
            if scan["scanned"] and not scan["verified_prefix_ks"]:
                problems.append(f"B2 {model}/{seed}: no prefix verification recorded")
            if seed not in b1:
                problems.append(f"B2 {model}/{seed}: no B1 rerun of this seed to check against")
                continue
            for key in ("completed", "decisions", "survival_s", "task_rewards"):
                if scan["learned_full"][key] != b1[seed]["learned_full"][key]:
                    problems.append(f"B2 {model}/{seed}: learned {key} differs from B1")
            if scan["scanned"] != (not b1[seed]["scanned"]):
                problems.append(f"B2 {model}/{seed}: success/failure split differs from B1")
    return problems


# -- fidelity -------------------------------------------------------------------


def fidelity_checks(
    scans: dict[str, dict[int, dict[str, Any]]],
    formal_learned: dict[str, dict[int, dict[str, Any]]],
    formal_pure: dict[int, dict[str, Any]],
    m2: dict[str, dict[str, dict[int, dict[str, Any]]]],
    expected_seeds: list[int],
) -> list[str]:
    """Return the list of problems; empty means every check passed."""

    problems: list[str] = []
    for model, by_seed in scans.items():
        missing = sorted(set(expected_seeds) - set(by_seed))
        if missing:
            problems.append(f"{model}: missing seeds {missing}")
        commits = {scan.get("code_commit") for scan in by_seed.values()}
        models = {scan.get("model_sha256") for scan in by_seed.values()}
        if len(commits) != 1 or None in commits:
            problems.append(f"{model}: scans come from {len(commits)} code commits ({commits})")
        if len(models) != 1:
            problems.append(f"{model}: scans come from {len(models)} model files")
        if any(scan.get("code_dirty") for scan in by_seed.values()):
            problems.append(f"{model}: a scan ran on a dirty checkout")
        for seed, scan in sorted(by_seed.items()):
            learned = scan["learned_full"]
            if scan.get("scan") != "failures" or scan.get("max_decisions") is not None:
                problems.append(f"{model}/{seed}: not a full failures scan")
            if scan["scanned"] and not scan["verified_prefix_ks"]:
                problems.append(f"{model}/{seed}: no prefix verification recorded")
            if seed in formal_learned.get(model, {}):
                ref = formal_learned[model][seed]
                for key in ("completed", "decisions", "survival_s"):
                    if ref[key] != learned[key]:
                        problems.append(f"{model}/{seed}: learned {key} {learned[key]} != formal {ref[key]}")
            elif seed in m2.get(model, {}).get("learned_full", {}):
                ref = m2[model]["learned_full"][seed]
                for key in ("completed", "decisions", "failure"):
                    if ref[key] != learned[key]:
                        problems.append(f"{model}/{seed}: learned {key} {learned[key]} != M2 {ref[key]}")
            else:
                problems.append(f"{model}/{seed}: no formal learned record to check against")
            if not scan["scanned"]:
                continue
            k0 = scan["handoffs"][0]
            if seed in formal_pure:
                ref = formal_pure[seed]
                for key in ("completed", "decisions", "survival_s"):
                    if ref[key] != k0[key]:
                        problems.append(f"{model}/{seed}: k=0 {key} {k0[key]} != Pure MPC {ref[key]}")
            elif seed in m2.get(model, {}).get("baseline_full", {}):
                ref = m2[model]["baseline_full"][seed]
                for key in ("completed", "decisions", "failure"):
                    if ref[key] != k0[key]:
                        problems.append(f"{model}/{seed}: k=0 {key} {k0[key]} != M2 baseline {ref[key]}")
            else:
                problems.append(f"{model}/{seed}: no Pure MPC record to check k=0 against")
    return problems


# -- CLI ------------------------------------------------------------------------------


def _pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        name, _, path = value.partition("=")
        if not path:
            raise ValueError(f"expected MODEL=PATH, got {value!r}")
        out[name] = path
    return out


def _load_scans(directory: Path) -> dict[int, dict[str, Any]]:
    return {
        int(path.stem.split("_")[1]): json.loads(path.read_text())
        for path in sorted(Path(directory).glob("seed_*.json"))
    }


def _records(path: str) -> dict[int, dict[str, Any]]:
    return {int(r["seed"]): r for r in json.loads(Path(path).read_text())["records"]}


def _m2(paths: str) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = {"learned_full": {}, "baseline_full": {}}
    for path in paths.split(","):
        for episode in json.loads(Path(path).read_text())["episodes"]:
            if episode["kind"] in out:
                out[episode["kind"]][int(episode["seed"])] = episode
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", action="append", required=True, help="MODEL=DIR (B1 failures scan)")
    parser.add_argument("--formal-learned", action="append", default=[], help="MODEL=learned_only.json")
    parser.add_argument("--formal-pure", required=True, help="Pure MPC formal row JSON")
    parser.add_argument("--m2", action="append", default=[], help="MODEL=m2_a.json,m2_b.json,...")
    parser.add_argument("--b2", action="append", default=[], help=f"MODEL=DIR (B2-lite: {B2_SEEDS}, stride {B2_STRIDE})")
    parser.add_argument("--seeds", default="262000-262047,270000-270047")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    scans = {model: _load_scans(Path(d)) for model, d in _pairs(args.scan).items()}
    formal_learned = {model: _records(p) for model, p in _pairs(args.formal_learned).items()}
    m2 = {model: _m2(p) for model, p in _pairs(args.m2).items()}
    formal_pure = _records(args.formal_pure)
    expected = parse_seed_range(args.seeds)
    problems = fidelity_checks(scans, formal_learned, formal_pure, m2, expected)

    trajectories: dict[str, list[dict[str, Any]]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for model, by_seed in scans.items():
        trajectories[model] = [
            {"block": (seed // 1000) * 1000, **failure_trajectory(scan)}
            for seed, scan in sorted(by_seed.items())
            if scan["scanned"]
        ]
        gates[model] = model_gate(trajectories[model])
    b2: dict[str, Any] = {}
    b2_scans = {model: _load_scans(Path(d)) for model, d in _pairs(args.b2).items()}
    b2_problems = (
        b2_fidelity_checks(b2_scans, scans, parse_seed_range(B2_SEEDS)) if b2_scans else []
    )
    for model, by_seed in b2_scans.items():
        rows = [success_trajectory(s) for s in by_seed.values() if s["scanned"]]
        b2[model] = {"summary": b2_summary(rows), "trajectories": rows}
    if b2_scans:
        # B2 has no gate and never changes the B1 verdict; its accounting is
        # only usable when its own checks pass.
        b2["fidelity_problems"] = b2_problems
        b2["valid"] = not b2_problems

    result = {
        "preregistration": "docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md",
        "constants": {
            "nondegenerate_min_w_decisions": NONDEGENERATE_MIN_W,
            "min_nondegenerate_trajectories": MIN_NONDEGENERATE,
            "min_models": MIN_MODELS,
            "decision_period_s": DECISION_PERIOD_S,
        },
        "seeds": args.seeds,
        "fidelity_problems": problems,
        "verdict": verdict(not problems, gates),
        "gates": gates,
        "trajectories": trajectories,
        "b2": b2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({"verdict": result["verdict"], "fidelity_problems": len(problems),
                      "b2_valid": b2.get("valid") if b2_scans else None,
                      "b2_fidelity_problems": len(b2_problems),
                      "gates": {m: {k: g[k] for k in ("failures", "rescuable", "non_degenerate", "needle_only", "passes")}
                                for m, g in gates.items()}}, indent=1))


if __name__ == "__main__":
    main()
