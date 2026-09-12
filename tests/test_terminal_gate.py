"""The MPC's terminal rows must not fire on states the task does not judge.

`_predicted_terminal_active` used to be an axial half-space with no bound on
range, so a chaser 16 m from the target and 15 m off the approach axis had the
approach corridor imposed on it.  The linearised facet there wants about four
normalised units of slack against a `constraint_slack_limit` of 2.0, so the QP
reported `infeasible` and the zero-wrench fallback handed the chaser to
`omega * r`.  See `docs/TERMINAL_GATE_DEFECT.md`.
"""

import numpy as np

from controllers.mpc.constraints import (
    _INACTIVE_MARGIN,
    _predicted_terminal_active,
    entry_plane_radius_m,
    normalized_precapture_constraint_margins,
    normalized_precapture_truth_margins,
)
from dynamics.lie import make_transform, se3_log
from env.task import PrecaptureTaskConfig

_CORRIDOR_FACETS = 8
_OMEGA = np.array([0.0, 0.0, 0.041231])


def _state_at(position: np.ndarray) -> np.ndarray:
    state = np.zeros(12, dtype=np.float64)
    state[:6] = se3_log(make_transform(np.eye(3), position))
    return state


def test_entry_plane_radius_comes_from_frozen_task_terms() -> None:
    task = PrecaptureTaskConfig()
    assert np.isclose(
        entry_plane_radius_m(task),
        np.linalg.norm(task.port_position) + task.entry_port_axial_distance_m,
    )
    assert np.isclose(entry_plane_radius_m(task), 6.0)


def test_terminal_rows_do_not_fire_far_from_the_target() -> None:
    """The axial test alone would fire here; the range bound is what stops it."""

    task = PrecaptureTaskConfig()
    # 16.4 m out, 15 m off the approach axis, on the near side of the plane.
    position = np.array([-5.90, -10.91, 10.72])
    axial = float(task.approach_axis @ (position - task.port_position))
    assert axial < task.entry_port_axial_distance_m
    assert float(np.linalg.norm(position)) > entry_plane_radius_m(task)

    assert not _predicted_terminal_active(position, task, terminal_latched=False)

    margins = normalized_precapture_constraint_margins(
        _state_at(position),
        task,
        target_angular_velocity_rad_s=_OMEGA,
        terminal_latched=False,
        corridor_facets=_CORRIDOR_FACETS,
    )
    corridor = margins[4 : 5 + _CORRIDOR_FACETS]
    assert np.all(corridor == _INACTIVE_MARGIN)


def test_mpc_gate_agrees_with_the_truth_gate_outside_the_entry_sphere() -> None:
    """Whatever the optimiser refuses to solve for, the truth must also judge."""

    task = PrecaptureTaskConfig()
    for position in (
        np.array([-5.90, -10.91, 10.72]),
        np.array([-5.50, -10.91, 10.72]),
        np.array([-2.00, -12.00, 9.00]),
        np.array([1.00, -8.00, 14.00]),
    ):
        state = _state_at(position)
        constraint = normalized_precapture_constraint_margins(
            state,
            task,
            target_angular_velocity_rad_s=_OMEGA,
            terminal_latched=False,
            corridor_facets=_CORRIDOR_FACETS,
        )
        truth = normalized_precapture_truth_margins(
            state, task, target_angular_velocity_rad_s=_OMEGA, terminal_latched=False
        )
        constraint_terminal_active = bool(
            np.any(constraint[4 : 5 + _CORRIDOR_FACETS] != _INACTIVE_MARGIN)
        )
        truth_terminal_active = bool(truth[4] != _INACTIVE_MARGIN)
        assert constraint_terminal_active == truth_terminal_active


def test_terminal_rows_still_fire_inside_the_entry_region() -> None:
    """The bound must not switch the corridor off where the task does want it."""

    task = PrecaptureTaskConfig()
    position = np.array([-3.0, 0.4, 0.3])
    assert float(np.linalg.norm(position)) <= entry_plane_radius_m(task)
    assert _predicted_terminal_active(position, task, terminal_latched=False)

    margins = normalized_precapture_constraint_margins(
        _state_at(position),
        task,
        target_angular_velocity_rad_s=_OMEGA,
        terminal_latched=False,
        corridor_facets=_CORRIDOR_FACETS,
    )
    assert np.all(margins[4 : 5 + _CORRIDOR_FACETS] != _INACTIVE_MARGIN)


def test_latched_state_keeps_the_terminal_rows_at_any_range() -> None:
    """Once the truth latch has armed, the corridor applies wherever the chaser is."""

    task = PrecaptureTaskConfig()
    position = np.array([-5.90, -10.91, 10.72])
    assert float(np.linalg.norm(position)) > entry_plane_radius_m(task)
    assert _predicted_terminal_active(position, task, terminal_latched=True)


def test_behind_the_entry_plane_is_still_excluded() -> None:
    """The axial clause has not been weakened, only bounded."""

    task = PrecaptureTaskConfig()
    position = np.array([-8.0, 0.0, 0.0])
    axial = float(task.approach_axis @ (position - task.port_position))
    assert axial > task.entry_port_axial_distance_m
    assert not _predicted_terminal_active(position, task, terminal_latched=False)


def test_the_range_bound_is_what_excludes_the_defect_state() -> None:
    """Isolate the new clause: the axial test alone would have fired."""

    task = PrecaptureTaskConfig()
    position = np.array([-5.90, -10.91, 10.72])
    axial = float(task.approach_axis @ (position - task.port_position))
    assert axial < task.entry_port_axial_distance_m  # the old gate fired here
    assert float(np.linalg.norm(position)) > entry_plane_radius_m(task)
    assert not _predicted_terminal_active(position, task, terminal_latched=False)
