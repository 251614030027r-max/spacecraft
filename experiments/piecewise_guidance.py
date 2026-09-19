"""Low-dimensional piecewise 3D waypoint policy for the S5 interface test."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from controllers.mpc.prediction import RelativePredictionModel
from dynamics.lie import se3_exp
from dynamics.types import SpacecraftState
from env.task import PrecaptureTaskConfig


PIECEWISE_PLANNING_FOV_HALF_ANGLE_RAD = float(np.deg2rad(40.0))


@dataclass(frozen=True)
class PiecewiseGuidanceParameters:
    descent_radius_m: float
    reposition_radius_m: float
    first_switch_s: float
    second_switch_s: float

    def __post_init__(self) -> None:
        if self.descent_radius_m <= 0.0 or self.reposition_radius_m <= 0.0:
            raise ValueError("piecewise waypoint radii must be positive")
        if not 0.0 < self.first_switch_s < self.second_switch_s:
            raise ValueError("piecewise switch times must be strictly ordered")


class PiecewiseWaypointPlan:
    """Three-mode policy delivered only through the 3D external-local action.

    The first and second waypoints are co-rotating gates on the approach axis,
    outside or at the entry plane. The third is the co-rotating terminal
    waypoint. The controller still accepts one inertially oriented 3D vector
    every two seconds; no velocity or future truth is exposed.
    """

    def __init__(
        self,
        command_state: np.ndarray,
        command_target: SpacecraftState,
        prediction_model: RelativePredictionModel,
        task: PrecaptureTaskConfig,
        parameters: PiecewiseGuidanceParameters,
        *,
        max_time_s: float,
    ) -> None:
        self.parameters = parameters
        self._dt_s = float(prediction_model.dt_s)
        self._task = task
        se3_exp(command_state[:6])  # Validate the same finite pose convention.
        finish_step = int(np.ceil(max_time_s / self._dt_s))
        forecast = [command_target.copy()]
        for step in range(finish_step):
            forecast.append(
                prediction_model.propagate_target(forecast[-1], step * self._dt_s)
            )
        self._forecast = tuple(forecast)

    def _target_rotation(self, time_s: float) -> np.ndarray:
        index = int(np.clip(round(time_s / self._dt_s), 0, len(self._forecast) - 1))
        return self._forecast[index].rotation

    def phase(self, time_s: float) -> str:
        if time_s < self.parameters.first_switch_s:
            return "outer_gate"
        if time_s < self.parameters.second_switch_s:
            return "reposition"
        return "terminal"

    def reference(self, time_s: float) -> np.ndarray:
        phase = self.phase(time_s)
        rotation = self._target_rotation(time_s)
        if phase == "outer_gate":
            return rotation @ (
                self.parameters.descent_radius_m * self._task.approach_axis
            )
        if phase == "reposition":
            return rotation @ (
                self.parameters.reposition_radius_m * self._task.approach_axis
            )
        return rotation @ self._task.desired_position

    def metadata(self) -> dict[str, float | str]:
        return {
            "policy": "s5_piecewise_3d_waypoints",
            **asdict(self.parameters),
            "segments": "outer_axis_gate|entry_axis_gate|terminal",
            "planning_fov_half_angle_rad": PIECEWISE_PLANNING_FOV_HALF_ANGLE_RAD,
        }
