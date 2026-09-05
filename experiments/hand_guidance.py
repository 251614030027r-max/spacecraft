"""Frozen online hand guidance for the precapture external-local interface."""

from __future__ import annotations

import numpy as np

from controllers.mpc.prediction import RelativePredictionModel
from dynamics.lie import se3_exp, so3_exp
from dynamics.types import SpacecraftState
from env.task import PrecaptureTaskConfig
from experiments.evaluate_precapture_oracle import (
    FINAL_DESCENT_TIME_S,
    FINAL_RADIUS_M,
    MATCH_TIME_S,
    OUTER_STAGING_RADIUS_M,
    _slerp_direction,
    _smoothstep,
)


FINISH_RESERVE_S = 5.0
HAND_OUTER_DESCENT_TIME_S = 80.0


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

        match_steps = int(round(MATCH_TIME_S / self._dt_s))
        tumble_rate = float(np.linalg.norm(command_target.omega))
        if tumble_rate <= 1.0e-9:
            raise ValueError("hand guidance requires non-zero target tumble")
        self.wait_period_s = float(2.0 * np.pi / tumble_rate)
        # The rate-matching segment occupies the final 40 s of the full-period
        # wait. Entry is still withheld until one complete nominal tumble has
        # elapsed, while the lower layer retains enough time to descend.
        first = int(np.ceil(self.wait_period_s / self._dt_s))
        # Leave the same 80 s allowance used by the feasibility construction,
        # but hand only the terminal waypoint to MPC: the lower layer owns the
        # closed-loop descent rate because external_local carries positions,
        # not an open-loop radial velocity profile.
        latest = int(
            np.floor(
                (max_time_s - FINAL_DESCENT_TIME_S - FINISH_RESERVE_S)
                / self._dt_s
            )
        )
        if first > latest:
            raise ValueError("episode is too short for frozen hand guidance")
        # The classical comparison takes the first full-period window. Unlike
        # the feasibility plan it does not search later windows for a cheaper
        # alignment; that decision is left to the learned upper layer.
        self.match_end_steps = first
        self.match_end_time_s = self.match_end_steps * self._dt_s
        self.match_steps = match_steps
        self.match_start_steps = self.match_end_steps - match_steps
        self.match_start_time_s = self.match_start_steps * self._dt_s
        self._match_directions = self._match_direction_track(
            self.match_end_steps, match_steps
        )
        self.match_entry_direction = self._match_directions[0]
        self.match_alignment_angle_rad = float(
            np.arccos(
                np.clip(
                    self.initial_direction @ self.match_entry_direction,
                    -1.0,
                    1.0,
                )
            )
        )

    def _blend(self, offset_steps: int, match_steps: int) -> float:
        return _smoothstep(offset_steps / match_steps)

    def _match_direction_track(self, end_index: int, match_steps: int) -> np.ndarray:
        directions = np.zeros((match_steps + 1, 3), dtype=np.float64)
        direction = self._forecast[end_index].rotation @ self._axis
        directions[match_steps] = direction
        for offset in range(match_steps, 0, -1):
            index = end_index - offset
            sample = self._forecast[max(index, 0)]
            omega_inertial = sample.rotation @ sample.omega
            rate = self._blend(offset, match_steps) * omega_inertial
            direction = so3_exp(-rate * self._dt_s) @ direction
            direction /= np.linalg.norm(direction)
            directions[offset - 1] = direction
        return directions

    def _match_entry_direction(self, end_index: int, match_steps: int) -> np.ndarray:
        return self._match_direction_track(end_index, match_steps)[0]

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
        if time_s <= HAND_OUTER_DESCENT_TIME_S:
            fraction = _smoothstep(time_s / HAND_OUTER_DESCENT_TIME_S)
            return self.initial_radius_m + fraction * (
                OUTER_STAGING_RADIUS_M - self.initial_radius_m
            )
        if time_s <= self.match_end_time_s:
            return OUTER_STAGING_RADIUS_M
        return FINAL_RADIUS_M

    def _inertial_direction(self, time_s: float) -> np.ndarray:
        if time_s <= self.match_start_time_s:
            fraction = _smoothstep(
                time_s / max(self.match_start_time_s, self._dt_s)
            )
            return _slerp_direction(
                self.initial_direction, self.match_entry_direction, fraction
            )
        if time_s >= self.match_end_time_s:
            return self._target_rotation(time_s) @ self._axis
        offset = int(round((time_s - self.match_start_time_s) / self._dt_s))
        offset = int(np.clip(offset, 0, self.match_steps))
        return self._match_directions[offset]

    def _position_reference(self, time_s: float) -> np.ndarray:
        time_s = max(float(time_s), 0.0)
        return self._radius(time_s) * self._inertial_direction(time_s)

    def reference(self, time_s: float) -> np.ndarray:
        """Return one inertially oriented position/velocity waypoint."""

        time_s = max(float(time_s), 0.0)
        position = self._position_reference(time_s)
        step = self._dt_s
        if time_s < step:
            velocity = (self._position_reference(time_s + step) - position) / step
        elif time_s < self.match_end_time_s <= time_s + step:
            velocity = (position - self._position_reference(time_s - step)) / step
        else:
            velocity = (
                self._position_reference(time_s + step)
                - self._position_reference(time_s - step)
            ) / (2.0 * step)
        return np.concatenate((position, velocity))

    def metadata(self) -> dict[str, float | str]:
        return {
            "policy": "frozen_online_nominal_coast_match_cross",
            "outer_staging_radius_m": OUTER_STAGING_RADIUS_M,
            "outer_descent_time_s": HAND_OUTER_DESCENT_TIME_S,
            "full_tumble_wait_s": self.wait_period_s,
            "match_duration_s": MATCH_TIME_S,
            "selected_match_start_s": self.match_start_time_s,
            "selected_match_end_s": self.match_end_time_s,
            "closed_loop_descent_allowance_s": FINAL_DESCENT_TIME_S,
            "terminal_waypoint_switch_s": self.match_end_time_s,
            "match_alignment_angle_rad": self.match_alignment_angle_rad,
            "window_rule": "first_full_nominal_tumble_period",
        }
