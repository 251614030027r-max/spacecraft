"""Physical constants and baseline limits in SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EarthConstants:
    mu: float
    equatorial_radius: float
    j2: float


EARTH = EarthConstants(
    mu=398600.44e9,
    equatorial_radius=6378.0e3,
    j2=1.08262668e-3,
)

CONTROL_DT_S = 0.1
MAX_EPISODE_TIME_S = 200.0
MAX_RELATIVE_DISTANCE_M = 30.0
MAX_FORCE_PER_AXIS_N = 5.0
MAX_TORQUE_PER_AXIS_NM = 0.6
