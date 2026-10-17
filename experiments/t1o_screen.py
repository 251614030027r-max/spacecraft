"""T1-O physical executability screen (research decision 2026-10-17, sections 1-2).

Pre-registered before any screening episode is run (this file and the opening files are
committed first; the verdict is applied mechanically by ``verdict``).

Profiles are screened in the fixed order P1 then P2 (``env.t1_o_task``); P2 only if P1 fails.
If P1 passes, P2 is not screened; if both fail, the screen stops and the G* tracking /
recovery / hold failure mechanism is reported.

Openings (``openings``): per profile, 32 exploratory openings, 2 per cell of the 4 x 4 initial
geometry (|sin LOS latitude| in four equal-probability bins x port phase in four 90 deg bins),
the first seeds in seed order from the profile's block (P1 from 299500, P2 from 299700; both
disjoint from every registered block: T1-L 297xxx / 2980xx-2981xx, T1-O discovery 2983xx,
confirmation 298400-298599, development 298600-, T1-R 299000-299195, earlier exploratory
299900-). Reset features only, frozen before any rollout; no seed, latitude or failure is
excluded afterwards. The 16-opening subset is the first seed of every cell.

Policies (the 8 fixed task candidates of the plan; the screen runs F1 on the 32 openings and
F3, F7 on the 16-subset): first-plan delay after the feasible anchor, kappa, entry speed.
The delay applies to the first plan only; afterwards every candidate follows the same rule:
a live plan is kept, a review without a live plan asks for the earliest feasible arrival.
Review period 3 (6 s) with plan-event reviews, as the T1-O interface.

Verdict (all of S1-S3 for PASS):
  S1  F1 on the 32 openings: >= 30 captures and 0 unsafe;
  S2  F3 and F7 on the 16-subset: 0 unsafe each;
  S3  physically distinct execution on the 16-subset, for each of F3 and F7: in >= 12 / 16
      openings the request is adopted differently from F1 (F3: first adopted arrival >= 30 s
      later than F1's; F7: first plan with entry speed 0.06 m/s where F1's is 0.02 m/s), and
      in >= 12 / 16 openings both F1 and the candidate capture and differ in closed loop by
      |delta time| >= 2 s or |delta delta-v| >= 0.02 m/s.
A failure of S1 or S2 is a physical NO-GO of the profile; S3 failing means legal requests are
projected onto the same execution. Per opening the screen reports MPC force saturation, plan
tracking error, deviation re-plans, recoveries and holds, radial speed during holds, maximum
range, safety margins, requested and adopted arrival times, and the discounted return at fuel
weights 5 / 10 / 15 / 20 re-scored exactly from per-step delta-v. Zero unsafe is a sample
admission, not a safety guarantee.

    python -B -m experiments.t1o_screen openings --profile p1 --output experiments/t1o_screen_p1_openings.json
    python -B -m experiments.t1o_screen run --profile p1 --policy F1 --openings <file> --output-dir <dir> --workers 4
    python -B -m experiments.t1o_screen run --profile p1 --policy F3 --subset16 --openings <file> --output-dir <dir>
    python -B -m experiments.t1o_screen verdict --profile p1 --openings <file> --root <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from env.t1_o_task import T1O_DELAY_MAX_S, T1O_PROFILES, t1o_env_config, t1o_review_env

#: (first-plan delay s, kappa, entry speed m/s)
CANDIDATES: dict[str, tuple[float, float, float]] = {
    "F1": (0.0, 1.0, 0.02), "F2": (30.0, 1.0, 0.02), "F3": (60.0, 1.0, 0.02), "F4": (90.0, 1.0, 0.02),
    "F5": (0.0, 0.7, 0.02), "F6": (60.0, 0.7, 0.02), "F7": (0.0, 1.0, 0.06), "F8": (60.0, 1.0, 0.06),
}
SCREEN_START = {"p1": 299_500, "p2": 299_700}
FUEL_WEIGHTS = (5.0, 10.0, 15.0, 20.0)
BASE_FUEL = 5.0


# --------------------------------------------------------------------------- #
def select_openings(profile: str, start: int, per_cell: int = 2, bins: int = 4, limit: int = 199) -> list[dict]:
    from env.t1_env import T1Env

    env = T1Env(t1o_env_config(profile))
    cells: dict[tuple[int, int], list[dict]] = {}
    for seed in range(start, start + limit + 1):
        f = env.reset(seed)["features"]
        lat_bin = min(int(abs(np.sin(np.deg2rad(f["latitude_deg"]))) * bins), bins - 1)
        phase_bin = min(int(f["phase_deg"] / (360.0 / bins)), bins - 1)
        cell = cells.setdefault((lat_bin, phase_bin), [])
        if len(cell) < per_cell:
            cell.append({"seed": seed, "latitude_bin": lat_bin, "phase_bin": phase_bin, "features": f})
        if len(cells) == bins * bins and all(len(c) == per_cell for c in cells.values()):
            break
    else:
        raise RuntimeError(f"not every cell filled within {start}..{start + limit}")
    openings = sorted((o for c in cells.values() for o in c), key=lambda o: o["seed"])
    return openings


def subset16(openings: dict) -> list[int]:
    first: dict[tuple[int, int], int] = {}
    for o in openings["openings"]:
        first.setdefault((o["latitude_bin"], o["phase_bin"]), o["seed"])
    return sorted(first.values())


# --------------------------------------------------------------------------- #
class FixedCandidate:
    def __init__(self, name: str) -> None:
        self.name = name
        self.delay_s, self.kappa, self.entry_speed = CANDIDATES[name]
        self.first_done = False

    def action(self, env) -> np.ndarray:
        c = env.config
        a1 = 2.0 * (self.kappa - c.kappa_min) / (c.kappa_max - c.kappa_min) - 1.0
        a2 = (self.entry_speed - c.entry_speed_nominal_m_s) / c.entry_speed_span_m_s
        if env.generator.remaining_s(env.env) > 0.0:
            a0 = 0.0  # keep the live plan
        elif not self.first_done:
            a0 = 2.0 * self.delay_s / env.delay_max_s - 1.0  # delay after the feasible anchor
        else:
            a0 = -1.0  # earliest feasible
        return np.array([a0, a1, a2], dtype=np.float64)

    def executed(self, env, info) -> None:
        rv = info["review"]
        if rv["reviewed"] and ("tau_min_feasible_s" in rv or "no_feasible_anchor" in rv):
            self.first_done = True


def radial_speed(env) -> float:
    e = env.env
    los = e.chaser_state.position - e.target_state.position
    rel = e.chaser_state.rotation @ e.chaser_state.velocity - e.target_state.rotation @ e.target_state.velocity
    return float(los @ rel / np.linalg.norm(los))


def run_episode(profile: str, policy_name: str, opening: int) -> dict:
    env = t1o_review_env(profile, BASE_FUEL)
    policy = FixedCandidate(policy_name)
    env.reset(options={"opening": opening})
    gamma = env.env.config.reward.gamma
    log, k = [], 0
    while True:
        due = bool(env._due)
        action = policy.action(env)
        _, reward, terminated, truncated, info = env.step(action)
        if due:
            policy.executed(env, info)
        x, rv = info["execution"], info["review"]
        m = env.env.metrics()
        log.append({
            "t": round(env.env.time_s, 3), "reviewed": rv["reviewed"], "plan_event": rv["plan_event"],
            "a": [float(v) for v in info["executed_action"]],
            "tau_min_feasible_s": rv.get("tau_min_feasible_s"), "delay_s": rv.get("delay_s"),
            "tau_request_s": x["tau_request_s"], "tau_effective_s": x["tau_effective_s"],
            "status": x["plan_status"], "trigger": x["new_plan_trigger"], "deadline": x["deadline"],
            "slip_s": x["slip_s"], "new_plan": x["new_plan"], "request_kind": x["request_kind"],
            "deviation_before_m": x["deviation_before_m"], "plan_deviation_end_m": x["plan_deviation_end_m"],
            "force_saturated_fraction": x["force_saturated_fraction"], "zero_fallbacks": x["zero_fallbacks"],
            "max_slack_solved": x["max_slack_solved"], "plan_wall_s": x["plan_wall_s"],
            "entry_speed_request_m_s": x["entry_speed_request_m_s"], "entry_speed_applied": x["entry_speed_applied"],
            "entry_speed_plan_m_s": x["entry_speed_plan_m_s"], "kappa": info["kappa"],
            "range_m": float(m.range_m), "radial_speed_m_s": radial_speed(env),
            "delta_v_step_m_s": float(info["delta_v_step_m_s"]), "r": float(reward),
            "descent": x["plan_status"] == "descent",
        })
        k += 1
        if terminated or truncated:
            break
    s = info["episode_summary"]
    returns = {}
    for lam in FUEL_WEIGHTS:
        returns[str(lam)] = float(sum(gamma ** i * (step["r"] - (lam - BASE_FUEL) * step["delta_v_step_m_s"])
                                      for i, step in enumerate(log)))
    plans = [{"t": st["t"] - 2.0, "trigger": st["trigger"], "status": st["status"], "tau_request_s": st["tau_request_s"],
              "tau_effective_s": st["tau_effective_s"], "slip_s": st["slip_s"], "deadline": st["deadline"],
              "entry_speed_plan_m_s": st["entry_speed_plan_m_s"]}
             for st in log if st["status"].startswith(("optimal", "recovery"))]
    live = [st for st in log if st["status"].startswith(("optimal", "kept", "recovery"))]  # steps flying a plan
    holds = [st for st in log if st["status"].startswith("hold")]
    first = log[0]
    return {
        **s, "profile": profile, "policy": policy_name, "candidate": CANDIDATES[policy_name], "opening": opening,
        "steps": len(log), "discounted_return": returns[str(BASE_FUEL)], "discounted_return_by_fuel_weight": returns,
        "first_tau_min_feasible_s": first["tau_min_feasible_s"], "first_delay_s": first["delay_s"],
        "first_tau_request_s": first["tau_request_s"],
        "first_plan": plans[0] if plans else None,
        "new_plans": len(plans), "plans": plans,
        "deviation_replans": sum(st["trigger"] == "deviation" for st in log),
        "recoveries": sum(st["status"].startswith("recovery") for st in log),
        "hold_steps": len(holds),
        "hold_max_outward_speed_m_s": max([st["radial_speed_m_s"] for st in holds], default=None),
        "tracking_error_mean_m": float(np.mean([st["deviation_before_m"] for st in live])) if live else None,
        "tracking_error_max_m": float(np.max([st["deviation_before_m"] for st in live])) if live else None,
        "saturation_mean": float(np.mean([st["force_saturated_fraction"] for st in log])),
        "max_outward_speed_m_s": float(max(st["radial_speed_m_s"] for st in log)),
        "decision_log": log,
    }


def _one(args) -> None:
    import torch

    torch.set_num_threads(1)
    profile, policy, opening, out = args
    row = run_episode(profile, policy, opening)
    Path(out, f"{opening}.json").write_text(json.dumps(row, default=float))
    print(opening, policy, row["event"], round(row["time_s"], 1), round(row["delta_v_m_s"], 3), flush=True)


def run(profile: str, policy: str, seeds: list[int], output_dir: Path, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    todo = [(profile, policy, s, str(output_dir)) for s in seeds if not (output_dir / f"{s}.json").exists()]
    if workers > 1 and todo:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(workers, len(todo))) as pool:
            pool.map(_one, todo, chunksize=1)
    else:
        for job in todo:
            _one(job)


# --------------------------------------------------------------------------- #
def _rows(directory: Path, seeds: list[int]) -> dict[int, dict]:
    return {s: json.loads((directory / f"{s}.json").read_text()) for s in seeds}


def verdict(profile: str, openings_path: Path, root: Path) -> dict:
    openings = json.loads(openings_path.read_text())
    seeds, sub = openings["seeds"], subset16(openings)
    f1 = _rows(root / "F1", seeds)
    out: dict = {"profile": profile, "openings_sha256": hashlib.sha256(openings_path.read_bytes()).hexdigest(),
                 "seeds": seeds, "subset16": sub}
    count = lambda rows, ev: sum(r["event"] == ev for r in rows.values())  # noqa: E731
    out["F1"] = {"captures": count(f1, "success"), "unsafe": count(f1, "unsafe"), "timeout": count(f1, "timeout")}
    s1 = out["F1"]["captures"] >= 30 and out["F1"]["unsafe"] == 0
    s2, s3 = True, True
    for name in ("F3", "F7"):
        rows = _rows(root / name, sub)
        adopted, differs = [], []
        for s in sub:
            a, b = f1[s], rows[s]
            pa, pb = a["first_plan"], b["first_plan"]
            if name == "F3":
                ok = pa is not None and pb is not None and pb["tau_effective_s"] - pa["tau_effective_s"] >= 30.0
            else:
                ok = (pa is not None and pb is not None and pb["entry_speed_plan_m_s"] is not None
                      and abs(pb["entry_speed_plan_m_s"] - 0.06) < 1e-9 and abs(pa["entry_speed_plan_m_s"] - 0.02) < 1e-9)
            adopted.append(bool(ok))
            both = a["event"] == "success" and b["event"] == "success"
            differs.append(bool(both and (abs(b["time_s"] - a["time_s"]) >= 2.0
                                          or abs(b["delta_v_m_s"] - a["delta_v_m_s"]) >= 0.02)))
        out[name] = {"captures": count(rows, "success"), "unsafe": count(rows, "unsafe"), "timeout": count(rows, "timeout"),
                     "request_adopted_differently": int(sum(adopted)), "closed_loop_distinct": int(sum(differs))}
        s2 = s2 and out[name]["unsafe"] == 0
        s3 = s3 and sum(adopted) >= 12 and sum(differs) >= 12
    lat = {o["seed"]: o["features"]["latitude_deg"] for o in openings["openings"]}
    low = [s for s in seeds if abs(lat[s]) <= 20.0]
    out["F1_low_latitude"] = {"openings": len(low), "captures": sum(f1[s]["event"] == "success" for s in low),
                              "unsafe": sum(f1[s]["event"] == "unsafe" for s in low)}
    out["S1"], out["S2"], out["S3"] = bool(s1), bool(s2), bool(s3)
    out["verdict"] = "PASS" if (s1 and s2 and s3) else "FAIL"
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("openings")
    o.add_argument("--profile", choices=sorted(SCREEN_START), required=True)
    o.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=sorted(T1O_PROFILES), required=True)
    r.add_argument("--policy", choices=sorted(CANDIDATES), required=True)
    r.add_argument("--openings", type=Path, required=True)
    r.add_argument("--subset16", action="store_true")
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--workers", type=int, default=1)
    v = sub.add_parser("verdict")
    v.add_argument("--profile", choices=sorted(SCREEN_START), required=True)
    v.add_argument("--openings", type=Path, required=True)
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "openings":
        openings = select_openings(args.profile, SCREEN_START[args.profile])
        doc = {"rule": __doc__.split("\n\n")[3], "profile": args.profile, "profile_spec": T1O_PROFILES[args.profile],
               "start": SCREEN_START[args.profile], "per_cell": 2, "seeds": [x["seed"] for x in openings],
               "openings": openings}
        doc["subset16"] = subset16(doc)
        args.output.write_text(json.dumps(doc, indent=1))
        print(len(openings), "openings", openings[0]["seed"], "...", openings[-1]["seed"])
    elif args.command == "run":
        doc = json.loads(args.openings.read_text())
        assert doc["profile"] == args.profile, "openings belong to another profile"
        run(args.profile, args.policy, doc["subset16"] if args.subset16 else doc["seeds"], args.output_dir, args.workers)
    else:
        result = verdict(args.profile, args.openings, args.root)
        text = json.dumps(result, indent=1)
        if args.output:
            args.output.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
