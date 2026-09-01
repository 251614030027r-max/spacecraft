"""Corrupted target-state estimate for the partial-observability probe.

The first three probes (terminal value, phase, model mismatch) all came back
negative because short-horizon MPC with per-step *truth* state feedback makes
every prediction-side error shrink with the (short) horizon. The one error that
does NOT shrink with the horizon is the state estimate itself: a wrongly
estimated target pose moves the body-fixed corridor/FOV frame at full magnitude
right now, so the controller flies a displaced corridor and can breach the real
one.

This estimator produces the target pose the controller *believes*, from the
truth pose, with three degradations that a per-step average cannot wash out and
that do not shrink with the horizon -- deliberately not zero-mean white noise:

* **attitude bias** -- a per-episode constant rotation offset (magnitude fixed,
  axis sampled), i.e. a persistent misestimate of the target's orientation;
* **delay** -- the estimate lags the truth by a fixed number of control steps;
* **low update rate** -- the estimate refreshes only every N steps and is held
  (zero-order hold) in between.

Truth is untouched: the environment still propagates the real target and judges
constraint violations on the real geometry. Only what the controller is told is
corrupted -- an output-feedback setup with a realistic non-cooperative sensor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dynamics.lie import so3_exp
from dynamics.types import SpacecraftState


class TargetStateEstimator:
    def __init__(
        self,
        *,
        bias_rad: float = 0.0,
        bias_seed: int | None = None,
        delay_steps: int = 0,
        update_every: int = 1,
        filter_mode: str = "hold",
        propagate_step: Callable[[SpacecraftState, float], SpacecraftState] | None = None,
        dt_s: float = 0.1,
    ) -> None:
        if bias_rad < 0.0:
            raise ValueError("bias_rad must be non-negative")
        if delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if update_every < 1:
            raise ValueError("update_every must be >= 1")
        if filter_mode not in {"hold", "propagate"}:
            raise ValueError("filter_mode must be 'hold' or 'propagate'")
        if filter_mode == "propagate" and propagate_step is None:
            raise ValueError("propagate filter needs a propagate_step callable")
        self.delay_steps = int(delay_steps)
        self.update_every = int(update_every)
        self.filter_mode = filter_mode
        self._propagate_step = propagate_step
        self.dt_s = float(dt_s)
        if bias_rad > 0.0:
            rng = np.random.default_rng(bias_seed)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            # A persistent inertial-frame attitude misestimate of fixed size:
            # the estimated body frame (and its body-fixed corridor) is rotated
            # by this from the truth for the whole episode.
            self._bias_rotation = so3_exp(axis * bias_rad)
        else:
            self._bias_rotation = np.eye(3)
        self._history: list[SpacecraftState] = []
        self._current: SpacecraftState | None = None
        self._fix_age_steps = 0
        self._fix_time_s = 0.0

    def reset(self) -> None:
        self._history = []
        self._current = None
        self._fix_age_steps = 0
        self._fix_time_s = 0.0

    def _biased(self, state: SpacecraftState) -> SpacecraftState:
        return SpacecraftState(
            rotation=self._bias_rotation @ state.rotation,
            position=state.position.copy(),
            omega=state.omega.copy(),
            velocity=state.velocity.copy(),
        )

    def estimate(
        self, truth_target: SpacecraftState, step: int, time_seconds: float = 0.0
    ) -> SpacecraftState:
        """Return the target pose the controller believes at this control step.

        A new sensor fix (the truth from ``delay_steps`` ago, plus the constant
        bias) is taken only on update steps; otherwise the last fix is held.
        Under ``filter_mode="hold"`` the raw fix is returned (zero-order hold).
        Under ``"propagate"`` -- an EKF-style predict step / dead-reckoning
        output-feedback baseline -- the fix is forward-propagated by its age with
        the target model, compensating the delay and hold; a persistent attitude
        bias is *not* removed by a predict-only filter (that needs a bias state),
        which is the expected "helps the lag, not the bias" behaviour.
        """

        self._history.append(truth_target.copy())
        delayed_index = max(0, len(self._history) - 1 - self.delay_steps)
        if self._current is None or step % self.update_every == 0:
            self._current = self._biased(self._history[delayed_index])
            self._fix_age_steps = self.delay_steps
            self._fix_time_s = time_seconds - self.delay_steps * self.dt_s
        elif self._current is not None:
            self._fix_age_steps += 1
        if self.filter_mode == "hold":
            return self._current
        assert self._propagate_step is not None
        estimate = self._current
        clock = self._fix_time_s
        for _ in range(self._fix_age_steps):
            estimate = self._propagate_step(estimate, clock)
            clock += self.dt_s
        return estimate


__all__ = ["TargetStateEstimator"]
