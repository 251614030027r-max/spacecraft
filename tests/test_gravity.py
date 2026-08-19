import numpy as np

from dynamics.constants import EARTH
from dynamics.gravity import (
    central_and_second_moment_force,
    second_moment_from_inertia,
)


def test_second_moment_uses_standard_rigid_body_relation() -> None:
    inertia = np.diag([2.0, 3.0, 4.0])
    expected = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    assert np.allclose(second_moment_from_inertia(inertia), expected)


def test_central_gravity_points_toward_earth() -> None:
    position = np.array([7.0e6, 0.0, 0.0])
    force = central_and_second_moment_force(
        np.eye(3),
        position,
        100.0,
        np.eye(3),
        include_second_moment=False,
    )
    assert force[0] < 0.0
    assert np.allclose(force[1:], 0.0)
    np.testing.assert_allclose(
        np.linalg.norm(force), EARTH.mu * 100.0 / np.linalg.norm(position) ** 2
    )
