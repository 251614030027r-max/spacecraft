"""Central gravity, gravity-gradient torque, second moment and J2 terms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import EARTH, EarthConstants
from .lie import hat3
from .types import GeneralizedForce, SpacecraftParameters, SpacecraftState


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class GravityOptions:
    include_central: bool = True
    include_second_moment: bool = True
    include_j2: bool = True


def second_moment_from_inertia(inertia: ArrayLike) -> FloatArray:
    """Convert rigid-body inertia to the standard second-moment matrix."""

    value = np.asarray(inertia, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("inertia must have shape=(3, 3)")
    return 0.5 * np.trace(value) * np.eye(3) - value


def _validate_orbit(
    rotation: ArrayLike, position: ArrayLike
) -> tuple[FloatArray, FloatArray, float]:
    rotation_value = np.asarray(rotation, dtype=np.float64)
    position_value = np.asarray(position, dtype=np.float64)
    if rotation_value.shape != (3, 3) or position_value.shape != (3,):
        raise ValueError("rotation and position must have shapes (3,3) and (3,)")
    radius = float(np.linalg.norm(position_value))
    if radius <= 1.0:
        raise ValueError("orbital radius is too small")
    return rotation_value, position_value, radius


def gravity_gradient_torque(
    rotation: ArrayLike,
    position: ArrayLike,
    inertia: ArrayLike,
    earth: EarthConstants = EARTH,
) -> FloatArray:
    rotation_value, position_value, radius = _validate_orbit(rotation, position)
    inertia_value = np.asarray(inertia, dtype=np.float64)
    position_body = rotation_value.T @ position_value
    return 3.0 * earth.mu / radius**5 * (
        hat3(position_body) @ inertia_value @ position_body
    )


def central_and_second_moment_force(
    rotation: ArrayLike,
    position: ArrayLike,
    mass: float,
    inertia: ArrayLike,
    earth: EarthConstants = EARTH,
    *,
    include_second_moment: bool = True,
) -> FloatArray:
    rotation_value, position_value, radius = _validate_orbit(rotation, position)
    position_body = rotation_value.T @ position_value
    force = -earth.mu * float(mass) / radius**3 * position_body
    if include_second_moment:
        moment = second_moment_from_inertia(inertia)
        quadratic = float(position_body @ moment @ position_body)
        force = (
            force
            - 3.0 * earth.mu / radius**5 * (moment @ position_body)
            + 7.5 * earth.mu * quadratic / radius**7 * position_body
        )
    return force


def j2_force_body(
    rotation: ArrayLike,
    position: ArrayLike,
    mass: float,
    earth: EarthConstants = EARTH,
) -> FloatArray:
    rotation_value, position_value, radius = _validate_orbit(rotation, position)
    x, y, z = position_value
    z_ratio = z * z / (radius * radius)
    inertial_shape = np.array(
        [
            x * (1.0 - 5.0 * z_ratio),
            y * (1.0 - 5.0 * z_ratio),
            z * (3.0 - 5.0 * z_ratio),
        ],
        dtype=np.float64,
    )
    coefficient = (
        -1.5
        * earth.mu
        * float(mass)
        * earth.j2
        * earth.equatorial_radius**2
        / radius**5
    )
    return coefficient * (rotation_value.T @ inertial_shape)


def gravity_wrench(
    state: SpacecraftState,
    parameters: SpacecraftParameters,
    options: GravityOptions = GravityOptions(),
    earth: EarthConstants = EARTH,
) -> GeneralizedForce:
    if not options.include_central:
        return GeneralizedForce.zeros()
    torque = gravity_gradient_torque(
        state.rotation, state.position, parameters.inertia, earth
    )
    force = central_and_second_moment_force(
        state.rotation,
        state.position,
        parameters.mass,
        parameters.inertia,
        earth,
        include_second_moment=options.include_second_moment,
    )
    if options.include_j2:
        force += j2_force_body(
            state.rotation, state.position, parameters.mass, earth
        )
    return GeneralizedForce(torque, force)
