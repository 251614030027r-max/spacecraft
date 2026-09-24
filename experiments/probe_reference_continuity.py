"""Reference continuity of the V2 direction blend vs a memory-axis blend (T0.2).

V2 blends the inertial hold direction ``h`` (which turns in the target frame)
toward the capture direction ``d`` by shortest-arc slerp. When ``h`` passes
near ``-d`` the shortest arc swings to the other side and the reference jumps
by up to 2 r in one decision. An endpoint-only axis (``h x d`` with sign
continuity) does not fix this: the axis rotates fast near the antipode.

The memory-axis blend rotates ``h`` to ``d`` about the admissible axis closest
to the previous one. Every axis ``n`` with ``(h - d) . n = 0`` maps ``h`` onto
``d``; project the previous axis onto that plane, then take the rotation angle
in the plane normal to ``n`` and unwrap it across decisions. At ``c = 1`` the
direction is ``d`` exactly.

Replays recorded V2 applied task states (no model), records ``h`` per
decision, and reports the largest adjacent reference jump under both blends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.probe_base_policy_rollout import _make_env


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * (axis @ v) * (1.0 - np.cos(angle))
    )


def shortest_arc(h: np.ndarray, d: np.ndarray, c: float) -> np.ndarray:
    angle = float(np.arccos(np.clip(h @ d, -1.0, 1.0)))
    if angle < 1.0e-8:
        return d if c >= 1.0 else h
    return (np.sin((1.0 - c) * angle) * h + np.sin(c * angle) * d) / np.sin(angle)


class MemoryAxisBlend:
    """Stateful direction blend; call ``reset()`` at every episode start."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._axis: np.ndarray | None = None
        self._angle: float | None = None

    def __call__(self, h: np.ndarray, d: np.ndarray, c: float) -> np.ndarray:
        w = h - d
        if self._axis is None:
            cross = np.cross(h, d)
            norm = float(np.linalg.norm(cross))
            if norm < 1.0e-9:  # h == d or h == -d at the very first call
                cross = np.cross(h, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(cross) < 1.0e-9:
                    cross = np.cross(h, np.array([0.0, 1.0, 0.0]))
                norm = float(np.linalg.norm(cross))
            axis = cross / norm
        else:
            axis = self._axis
            if float(w @ w) > 1.0e-18:
                axis = axis - (axis @ w) / (w @ w) * w
            axis = axis / np.linalg.norm(axis)
        hp = h - (h @ axis) * axis
        dp = d - (d @ axis) * axis
        angle = float(np.arctan2(axis @ np.cross(hp, dp), hp @ dp))
        if self._angle is not None:
            while angle - self._angle > np.pi:
                angle -= 2.0 * np.pi
            while angle - self._angle < -np.pi:
                angle += 2.0 * np.pi
        self._axis, self._angle = axis, angle
        return _rodrigues(h, axis, c * angle)


def measure(record: dict) -> dict:
    env = _make_env()
    try:
        env.reset(seed=int(record["seed"]))
        desired = np.asarray(env.environment_config.precapture_task.desired_position)
        d = desired / np.linalg.norm(desired)
        blend = MemoryAxisBlend()
        endpoint = MemoryAxisBlend()
        endpoint_error = 0.0
        old, new, natural, max_angle = [], [], [], 0.0
        previous_h = None
        for rho, c in record["task_state_applied"]:
            _, _, terminated, truncated, _ = env.step_with_proposal(
                env.action_for_task_state(float(rho), float(c)), proposal_accepted=True
            )
            h = np.asarray(env.env.target_state.rotation).T @ env._hold_inertial
            radius = max(env._hold_radius_m - float(rho), env.v2_radius_min_m)
            old.append(radius * shortest_arc(h, d, float(c)))
            new.append(radius * blend(h, d, float(c)))
            endpoint_error = max(endpoint_error, float(np.linalg.norm(endpoint(h, d, 1.0) - d)))
            if previous_h is not None:
                natural.append(radius * float(np.linalg.norm(h - previous_h)))
            previous_h = h
            max_angle = max(max_angle, float(np.degrees(np.arccos(np.clip(h @ d, -1, 1)))))
            if terminated or truncated:
                break
        return {
            "seed": int(record["seed"]),
            "v2_max_adjacent_jump_m": float(np.max(np.linalg.norm(np.diff(old, axis=0), axis=1))),
            "memory_axis_max_adjacent_jump_m": float(
                np.max(np.linalg.norm(np.diff(new, axis=0), axis=1))
            ),
            "natural_hold_motion_max_m": float(np.max(natural)),
            "max_hold_capture_angle_deg": max_angle,
            "c1_max_error_to_capture_direction": endpoint_error,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="records.json:seed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in args.case:
        path, seed = case.rsplit(":", 1)
        records = {r["seed"]: r for r in json.loads(Path(path).read_text())["records"]}
        row = measure(records[int(seed)])
        row["source"] = path
        rows.append(row)
        print(json.dumps(row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
