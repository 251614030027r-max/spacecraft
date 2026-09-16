"""Entry-phase favourability: a real window against a fixed inertial staging
direction, unlike the rotation-invariant (no-window) target-frame alignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from dynamics.types import SpacecraftState
from env.task import PrecaptureTaskConfig, precapture_entry_phase_favourability


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _target(rotation: np.ndarray) -> SpacecraftState:
    return SpacecraftState(rotation, np.zeros(3), np.zeros(3), np.zeros(3))


def test_favourable_when_port_normal_aligns_with_staging():
    cfg = PrecaptureTaskConfig()  # approach_axis = (-1, 0, 0)
    staging = np.array([-1.0, 0.0, 0.0])  # fixed inertial staging direction
    # target identity: port normal = approach_axis = (-1,0,0) -> aligned
    assert precapture_entry_phase_favourability(_target(np.eye(3)), staging, cfg) == pytest.approx(1.0)
    # target turned 180 deg: port normal = (1,0,0) -> anti-aligned
    assert precapture_entry_phase_favourability(_target(_rz(np.pi)), staging, cfg) == pytest.approx(-1.0)
    # target turned 90 deg: port normal = (0,-1,0) -> orthogonal
    assert precapture_entry_phase_favourability(_target(_rz(np.pi / 2)), staging, cfg) == pytest.approx(0.0, abs=1e-12)


def test_a_full_tumble_sweeps_a_real_window():
    cfg = PrecaptureTaskConfig()
    staging = np.array([-1.0, 0.0, 0.0])
    vals = [
        precapture_entry_phase_favourability(_target(_rz(a)), staging, cfg)
        for a in np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
    ]
    vals = np.array(vals)
    # a genuine window: the phase spends part of the tumble favourable and part
    # unfavourable -- not the constant value the corotating (no-window) case gives.
    assert vals.max() > 0.9 and vals.min() < -0.9
    assert np.ptp(vals) > 1.5


def test_zero_staging_direction_rejected():
    with pytest.raises(ValueError):
        precapture_entry_phase_favourability(_target(np.eye(3)), np.zeros(3))
