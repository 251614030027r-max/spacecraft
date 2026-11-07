"""Deterministic state-sequence diagnostic for illegal-entry recovery semantics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from dynamics.lie import make_transform
from dynamics.relative import RelativeState, reconstruct_chaser_state, relative_state
from env.phase2_env import precapture_planning_environment_config
from env.scenarios import target_initial_state
from env.se3_rendezvous_env import SE3RendezvousEnv


def _relative(position: list[float], velocity: list[float]) -> RelativeState:
    return RelativeState(
        make_transform(np.eye(3), np.asarray(position, dtype=np.float64)),
        np.asarray([0.0, 0.0, 0.0, *velocity], dtype=np.float64),
    )


def _place(env: SE3RendezvousEnv, relative: RelativeState) -> None:
    assert env.target_state is not None
    env.chaser_state = reconstruct_chaser_state(env.target_state, relative)
    env.relative = relative_state(env.target_state, env.chaser_state)


def run() -> dict[str, object]:
    """Exercise illegal crossing, scripted retreat, then legal re-entry.

    This deliberately prescribes the diagnostic states; it tests reachability of
    the environment state machine and geometry, not controller competence.
    """

    config = replace(
        precapture_planning_environment_config(),
        cache_target_trajectory=False,
        phase2_target_phase_sampling=False,
        include_j2=False,
    )
    task = config.precapture_task
    target = target_initial_state(tumble_scale=0.0)
    outside_disc = task.entry_disc_radius_m + 0.10
    initial = _relative([-6.005, outside_disc, 0.0], [0.10, 0.0, 0.0])
    env = SE3RendezvousEnv(config)
    try:
        env.reset(
            seed=262050,
            options={
                "target_state": target,
                "chaser_state": reconstruct_chaser_state(target, initial),
            },
        )
        _, _, terminated, truncated, illegal = env.step(np.zeros(6, dtype=np.float32))
        if not (
            illegal["illegal_terminal_entry"]
            and illegal["illegal_terminal_entry_count"] == 1
            and not illegal["terminal_region_active"]
            and not terminated
            and not truncated
        ):
            raise RuntimeError("diagnostic did not produce the intended illegal entry")

        retreat = _relative([-6.10, 0.0, 0.0], [-0.10, 0.0, 0.0])
        _place(env, retreat)
        _, _, _, _, retreated = env.step(np.zeros(6, dtype=np.float32))
        if retreated["terminal_region_active"]:
            raise RuntimeError("retreat unexpectedly latched the terminal region")

        legal_approach = _relative([-6.005, 0.0, 0.0], [0.10, 0.0, 0.0])
        _place(env, legal_approach)
        _, _, terminated, truncated, reentered = env.step(
            np.zeros(6, dtype=np.float32)
        )
        if not (
            reentered["terminal_region_active"]
            and env._terminal_region_entered
            and reentered["illegal_terminal_entry_count"] == 1
            and not terminated
            and not truncated
        ):
            raise RuntimeError("legal re-entry did not latch after retreat")

        return {
            "schema_version": 1,
            "diagnostic": "scripted_state_sequence_not_controller_demonstration",
            "initial_condition": "inside entry plane and outside entry disc after illegal crossing",
            "illegal_crossing_count": int(illegal["illegal_terminal_entry_count"]),
            "latched_after_illegal_crossing": bool(illegal["terminal_region_active"]),
            "latched_after_retreat": bool(retreated["terminal_region_active"]),
            "latched_after_legal_reentry": bool(reentered["terminal_region_active"]),
            "illegal_count_after_legal_reentry": int(
                reentered["illegal_terminal_entry_count"]
            ),
            "conclusion": "task_semantics_allow_retreat_and_legal_reentry",
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
