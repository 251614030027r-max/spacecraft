"""Why did episodes end? Read-only inspector for evaluation records.

``digest_run`` reports *that* a run failed on a terminal constraint, but not
which of the five went first, how long the policy held before it, and -- in the
two-phase modes -- whether the violation landed on the Waypoint transition step (a
Waypoint definition defect) or later inside the corridor (a control problem). Those
read identically in the aggrewaypointd rates and differently in the per-episode
``first_violation`` and ``waypoint_entry`` records, so read those.

Works on both task families: two-phase runs are reported relative to Waypoint
entry, single_phase runs relative to episode start, since there is no Waypoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


SCRIPTED_COMPLETION_S = 95.3


def _report(step: Any, records: list[dict[str, Any]]) -> None:
    if not records:
        print(f"{str(step):<9} no episodes")
        return
    arrivals = [r for r in records if r.get("waypoint_entry") is not None]
    kinds: Counter[str] = Counter()
    holds: list[float] = []
    for record in records:
        violation = record.get("first_violation")
        if violation is None:
            kinds["(no violation)"] += 1
            continue
        kinds[str(violation["type"])] += 1
        # Two-phase runs only enforce the constraints after the Waypoint, so the
        # meaningful duration is measured from there; single_phase enforces them
        # from step one, so it is measured from reset.
        entry = record.get("waypoint_entry")
        start = float(entry["time_s"]) if entry is not None else 0.0
        holds.append(float(violation["time_s"]) - start)

    # Where the policy parked matters as much as why it stopped: a run that
    # holds every constraint but never completes is either hovering short of
    # the desired pose or sitting on it and missing the tolerance, and those
    # are different problems.
    clean = [r for r in records if r.get("first_violation") is None]
    line = f"{str(step):<9} episodes={len(records):<3} "
    if arrivals:
        speeds = [float(r["waypoint_entry"]["speed_m_s"]) for r in arrivals]
        line += (
            f"Waypoint arrivals={len(arrivals)} "
            f"(speed mean {sum(speeds)/len(speeds):.3f} max {max(speeds):.3f} m/s)  "
        )
    print(line + "  ".join(f"{name}={count}" for name, count in kinds.most_common()))
    closest = [
        float(r["minimum_position_error_m"])
        for r in records
        if r.get("minimum_position_error_m") is not None
    ]
    if closest:
        note = ""
        if clean:
            clean_closest = [
                float(r["minimum_position_error_m"])
                for r in clean
                if r.get("minimum_position_error_m") is not None
            ]
            if clean_closest:
                note = (
                    f"   of the {len(clean)} that never violated:"
                    f" median {median(clean_closest):.2f} m,"
                    f" best {min(clean_closest):.2f} m"
                )
        print(
            f"{'':9} closest approach to the desired pose: "
            f"median {median(closest):.2f} m  best {min(closest):.2f} m"
            f"   (completion needs <= 0.25 m){note}"
        )
    # Worst margin reached on each constraint. corridor_axial is the distance
    # ahead of the port, and the corridor radius is that distance times
    # tan(half-angle), so a small axial margin means the cone has pinched shut
    # around the chaser -- the signature of overshooting past the desired pose.
    margins = [r["minimum_margins"] for r in records if r.get("minimum_margins")]
    if margins:
        print(
            f"{'':9} worst margin reached:  "
            + "  ".join(
                f"{name.replace('_margin_m_s','').replace('_margin_m','').replace('_margin_rad','')}"
                f"={min(float(m[name]) for m in margins):+.3f}"
                for name in margins[0]
            )
        )

    conditions = [
        r["best_completion_conditions"]
        for r in records
        if r.get("best_completion_conditions")
    ]
    if conditions:
        limits = {
            "position_error_m": 0.25,
            "attitude_error_rad": 0.1745,
            "total_speed_m_s": 0.05,
            "angular_velocity_rad_s": 0.02,
        }
        print(
            f"{'':9} best each completion condition reached (limit):  "
            + "  ".join(
                f"{name.split('_')[0]}={min(float(c[name]) for c in conditions):.3f}"
                f"/{limits[name]}"
                for name in limits
            )
        )
        streaks = [
            int(r.get("best_completion_streak", 0))
            for r in records
            if "best_completion_streak" in r
        ]
        if streaks:
            print(
                f"{'':9} best completion streak {max(streaks)} steps "
                f"of the 10 the 1.0 s hold requires"
            )

    if holds:
        immediate = sum(1 for value in holds if value <= 1.0e-9)
        anchor = "Waypoint entry" if arrivals else "reset"
        print(
            f"{'':9} held from {anchor} to first violation: "
            f"median {median(holds):.1f} s  min {min(holds):.1f} s  "
            f"max {max(holds):.1f} s   of the ~{SCRIPTED_COMPLETION_S} s the "
            f"scripted controller needs"
            + (f"   on the transition step itself: {immediate}/{len(holds)}" if arrivals else "")
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-violation breakdown from evaluation records"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="a single evaluation JSON")
    args = parser.parse_args()

    paths = (
        [args.evaluation]
        if args.evaluation is not None
        else sorted((args.run / "evaluations").glob("evaluation_*.json"))
    )
    if not paths:
        raise FileNotFoundError("no evaluation records found")

    print(f"run     {args.run}")
    print("step      what ended each episode, and how long it held first")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _report(payload.get("training_step", path.stem), payload["episode_records"])


if __name__ == "__main__":
    main()
