"""Relative SE(3) pose and body-twist state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .lie import adjoint, inverse_transform, make_transform, se3_log, split_transform
from .types import SpacecraftState


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RelativeState:
    transform: FloatArray
    twist: FloatArray

    @property
    def rotation(self) -> FloatArray:
        return self.transform[:3, :3]

    @property
    def position(self) -> FloatArray:
        return self.transform[:3, 3]

    @property
    def omega(self) -> FloatArray:
        return self.twist[:3]

    @property
    def velocity(self) -> FloatArray:
        return self.twist[3:]

    @property
    def exponential_coordinates(self) -> FloatArray:
        return se3_log(self.transform, project=True)


def state_transform(state: SpacecraftState) -> FloatArray:
    return make_transform(state.rotation, state.position)


def relative_state(target: SpacecraftState, chaser: SpacecraftState) -> RelativeState:
    """Return the chaser state expressed relative to the target."""

    target_transform = state_transform(target)
    chaser_transform = state_transform(chaser)
    relative_transform = inverse_transform(target_transform) @ chaser_transform
    relative_twist = chaser.twist - adjoint(
        inverse_transform(relative_transform)
    ) @ target.twist
    return RelativeState(relative_transform, relative_twist)


def reconstruct_chaser_state(
    target: SpacecraftState, relative: RelativeState
) -> SpacecraftState:
    """Invert :func:`relative_state` without depending on a controller module."""

    chaser_transform = state_transform(target) @ relative.transform
    rotation, position = split_transform(chaser_transform)
    chaser_twist = relative.twist + adjoint(
        inverse_transform(relative.transform)
    ) @ target.twist
    return SpacecraftState(
        rotation, position, chaser_twist[:3], chaser_twist[3:]
    )
