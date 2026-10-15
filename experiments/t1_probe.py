"""T1 scripted-intent runner (no learning). Used for interface checks and R1.

    python -B -m experiments.t1_probe run --intent C2 --seeds 290000-290003 --output-dir eval/t1/probe
    python -B -m experiments.t1_probe summarize --input-dir eval/t1/probe

Intents (all through the same interface G and the same MPC; trigger fixed in
advance, ``ENTRY_TRIGGER_DEG``):

C0  Pure: a_sigma = +1 every decision (identical to the Pure row).
C1  outer staging at the opening latitude: sigma = 0 at the episode-initial
    line-of-sight latitude; enter (sigma = 1) once the port normal is within
    the trigger angle of the line of sight.
C2  outer staging on the swept band: as C1 with latitude 0 (H0 equator).
C3  early synchronisation: sigma = 0.5 (co-rotating 7 m reference on the port
    axis) until the trigger, then sigma = 1.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from env.t1_env import T1Env, T1EnvConfig

ENTRY_TRIGGER_DEG = 35.0
INTENTS = ("C0", "C1", "C2", "C3")


def _action(latitude_rad: float, sigma: float) -> np.ndarray:
    return np.array([latitude_rad / (0.5 * np.pi), 2.0 * sigma - 1.0])


def scripted_action(intent: str, features: dict[str, float], initial: dict[str, float]) -> np.ndarray | None:
    """None means the Pure row."""

    if intent == "C0":
        return None
    entered = features["port_los_angle_deg"] <= ENTRY_TRIGGER_DEG
    if intent == "C1":
        return _action(np.deg2rad(initial["latitude_deg"]), 1.0 if entered else 0.0)
    if intent == "C2":
        return _action(0.0, 1.0 if entered else 0.0)
    if intent == "C3":
        return _action(0.0, 1.0 if entered else 0.5)
    raise ValueError(f"unknown intent {intent}")


def run_episode(env: T1Env, seed: int, intent: str, gamma: float) -> dict[str, Any]:
    info = env.reset(seed)
    initial = dict(info["features"])
    discounted = 0.0
    trace = []
    event = None
    k = 0
    while True:
        features = env.features()
        action = scripted_action(intent, features, initial)
        reward, terminated, truncated, step_info = env.step(action, pure=action is None)
        discounted += (gamma**k) * reward
        trace.append({
            "k": k,
            "sigma": step_info["sigma"],
            "range_m": features["range_m"],
            "latitude_deg": features["latitude_deg"],
            "phase_deg": features["phase_deg"],
            "port_los_angle_deg": features["port_los_angle_deg"],
            "reward": reward,
        })
        k += 1
        if terminated or truncated:
            event = step_info["event"]
            extra = {key: step_info[key] for key in ("violations", "numerical_error") if key in step_info}
            break
    return {
        "seed": int(seed),
        "intent": intent,
        "event": event,
        "clean_capture": event == "success",
        **extra,
        "discounted_return": discounted,
        "initial_features": initial,
        **env.summary(),
        "trace": trace,
    }


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_run(args: argparse.Namespace) -> None:
    env = T1Env(T1EnvConfig())
    gamma = env.config.reward.gamma
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        path = out / f"{args.intent}_{seed}.json"
        lock = path.with_suffix(".lock")
        if path.exists():
            continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        started = perf_counter()
        result = run_episode(env, seed, args.intent, gamma)
        result["wall_clock_s"] = perf_counter() - started
        path.write_text(json.dumps(result))
        lock.unlink()
        print(args.intent, seed, result["event"], round(result["time_s"], 1),
              round(result["delta_v_m_s"], 2), round(result["discounted_return"], 1), flush=True)


def cmd_summarize(args: argparse.Namespace) -> None:
    rows: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(Path(args.input_dir).glob("C*_*.json")):
        r = json.loads(path.read_text())
        rows.setdefault(r["seed"], {})[r["intent"]] = r
    for seed, by in sorted(rows.items()):
        f = next(iter(by.values()))["initial_features"]
        cells = []
        for intent in INTENTS:
            r = by.get(intent)
            if r is None:
                cells.append(f"{intent}:--")
                continue
            tag = "OK" if r["clean_capture"] else (r["event"] or "?")[:7]
            cells.append(f"{intent}:{tag} {r['time_s']:.0f}s {r['delta_v_m_s']:.2f} R{r['discounted_return']:.0f}")
        print(f"{seed} lat{f['latitude_deg']:+5.0f} ph{f['phase_deg']:4.0f} r{f['range_m']:4.1f} | " + " | ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--intent", choices=INTENTS, required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--output-dir", required=True)
    run.set_defaults(func=cmd_run)
    summ = sub.add_parser("summarize")
    summ.add_argument("--input-dir", required=True)
    summ.set_defaults(func=cmd_summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
