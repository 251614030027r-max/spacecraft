"""Disturbance interfaces for the nominal baseline dynamics."""

from .types import GeneralizedForce


def zero_disturbance(_: float) -> GeneralizedForce:
    """Return the nominal no-disturbance wrench."""

    return GeneralizedForce.zeros()
