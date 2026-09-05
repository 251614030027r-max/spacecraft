"""Frozen online hand guidance for the precapture external-local interface."""

from __future__ import annotations

import numpy as np

from controllers.mpc.prediction import RelativePredictionModel
from dynamics.lie import se3_exp
from dynamics.types import SpacecraftState
from env.task import PrecaptureTaskConfig
from experiments.evaluate_precapture_oracle import (
    FINAL_DESCENT_TIME_S,
    FINAL_RADIUS_M,
    MATCH_TIME_S,
    _slerp_direction,
    _smoothstep,
)


FINISH_RESERVE_S = 5.0
HAND_PLANNING_FOV_HALF_ANGLE_RAD = float(np.deg2rad(45.0))


def first_local_minimum(values: np.ndarray, *, last_index: int) -> int:
    """Return the first interior discrete local minimum up to ``last_index``."""

    samples = np.asarray(values, dtype=np.float64)
    upper = min(int(last_index), samples.size - 2)
    for index in range(1, upper + 1):
        if samples[index] <= samples[index - 1] and samples[index] < samples[index + 1]:
            return index
    raise ValueError("nominal forecast contains no admissible local-minimum window")


class HandGuidancePlan:
    """Online nominal-forecast version of the coast--match--cross structure.

    The constructor receives only the controller-visible relative state and
    target estimate. It creates its own nominal free-rigid-body forecast and
    freezes one entry window for the episode. It never reads the environment's
    cached future truth and does not optimise separately for an evaluation seed.
    """

    def __init__(
        self,
        command_state: np.ndarray,
        command_target: SpacecraftState,
        prediction_model: RelativePredictionModel,
        task: PrecaptureTaskConfig,
        *,
        max_time_s: float,
    ) -> None:
        self._dt_s = float(prediction_model.dt_s)
        self._axis = np.asarray(task.approach_axis, dtype=np.float64)
        transform = se3_exp(command_state[:6])
        initial_inertial = command_target.rotation @ transform[:3, 3]
        self.initial_radius_m = float(np.linalg.norm(initial_inertial))
        self.initial_direction = initial_inertial / self.initial_radius_m

        finish_step = int(np.ceil(max_time_s / self._dt_s))
        forecast = [command_target.copy()]
        for step in range(finish_step):
            forecast.append(
                prediction_model.propagate_target(
                    forecast[-1], step * self._dt_s
                )
            )
        self._forecast = tuple(forecast)

        self.match_steps = int(round(MATCH_TIME_S / self._dt_s))
        if float(np.linalg.norm(command_target.omega)) <= 1.0e-9:
            raise ValueError("hand guidance requires non-zero target tumble")
        latest_start = int(
            np.floor(
                (
                    max_time_s
                    - MATCH_TIME_S
                    - FINAL_DESCENT_TIME_S
                    - FINISH_RESERVE_S
                ) / self._dt_s
            )
        )
        axis_directions = np.asarray(
            [state.rotation @ self._axis for state in self._forecast]
        )
        angles = np.arccos(
            np.clip(axis_directions @ self.initial_direction, -1.0, 1.0)
        )
        self.match_start_steps = first_local_minimum(
            angles, last_index=latest_start
        )
        self.match_start_time_s = self.match_start_steps * self._dt_s
        self.match_end_steps = self.match_start_steps + self.match_steps
        self.match_end_time_s = self.match_end_steps * self._dt_s
        self.window_alignment_angle_rad = float(angles[self.match_start_steps])

    def _target_rotation(self, time_s: float) -> np.ndarray:
        index = int(round(time_s / self._dt_s))
        index = int(np.clip(index, 0, len(self._forecast) - 1))
        return self._forecast[index].rotation

    def phase(self, time_s: float) -> str:
        if time_s <= self.match_start_time_s:
            return "coast"
        if time_s <= self.match_end_time_s:
            return "match"
        return "cross"

    def _radius(self, time_s: float) -> float:
        if time_s <= self.match_end_time_s:
            return self.initial_radius_m
        return FINAL_RADIUS_M

    def _inertial_direction(self, time_s: float) -> np.ndarray:
        if time_s <= self.match_start_time_s:
            return self.initial_direction
        if time_s >= self.match_end_time_s:
            return self._target_rotation(time_s) @ self._axis
        fraction = _smoothstep(
            (time_s - self.match_start_time_s) / MATCH_TIME_S
        )
        return _slerp_direction(
            self.initial_direction,
            self._target_rotation(time_s) @ self._axis,
            fraction,
        )

    def _position_reference(self, time_s: float) -> np.ndarray:
        time_s = max(float(time_s), 0.0)
        return self._radius(time_s) * self._inertial_direction(time_s)

    def reference(self, time_s: float) -> np.ndarray:
        """Return the declared inertially oriented 3D waypoint action."""

        return self._position_reference(time_s)

    def metadata(self) -> dict[str, float | str]:
        return {
            "policy": "frozen_online_nominal_coast_match_cross",
            "fixed_reposition_radius_m": self.initial_radius_m,
            "match_duration_s": MATCH_TIME_S,
            "selected_match_start_s": self.match_start_time_s,
            "selected_match_end_s": self.match_end_time_s,
            "closed_loop_descent_allowance_s": FINAL_DESCENT_TIME_S,
            "terminal_waypoint_switch_s": self.match_end_time_s,
            "window_alignment_angle_rad": self.window_alignment_angle_rad,
            "window_rule": "first_local_minimum_of_axis_to_chaser_direction",
            "planning_fov_half_angle_rad": HAND_PLANNING_FOV_HALF_ANGLE_RAD,
        }
