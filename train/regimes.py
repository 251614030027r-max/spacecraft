"""Task regimes for the final-regime screening (2026-10-06).

A regime changes only the target tumble rate and the initial-state
distribution (range, inertial speed). Everything else -- MPC, authority,
constraints, completion, time limit, reward, interface -- stays the V3e
mainline. The MPC predicts the target from its measured state, so a faster
tumble is a genuine input change for it, not a handicap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from env.se3_rendezvous_env import SE3RendezvousConfig

#: Tumble rate (deg/s) produced by ``phase2_target_tumble_scale = 0.20``,
#: measured on the mainline environment (0.041231 rad/s).
NOMINAL_TUMBLE_DEG_S = 2.362365508345389
NOMINAL_TUMBLE_SCALE = 0.20


@dataclass(frozen=True)
class Regime:
    name: str
    tumble_deg_s: float
    range_min_m: float
    range_max_m: float = 28.0
    speed_max_m_s: float = 0.10

    @property
    def tumble_scale(self) -> float:
        return NOMINAL_TUMBLE_SCALE * self.tumble_deg_s / NOMINAL_TUMBLE_DEG_S


def _grid(tumbles: tuple[float, ...], range_mins: tuple[float, ...], speed: float = 0.10) -> list[Regime]:
    out = []
    for w in tumbles:
        for r in range_mins:
            tag = f"w{w:.2f}_r{r:.0f}" + ("" if speed == 0.10 else f"_v{speed:.2f}")
            out.append(Regime(tag, w, r, speed_max_m_s=speed))
    return out


#: Stage 1 (Pure only): tumble 2.36 / 3.0 / 3.5 deg/s x initial range 15-28 / 18-28 / 20-28 m.
#: The first cell is the current mainline, kept as the same-block reference.
STAGE1 = _grid((NOMINAL_TUMBLE_DEG_S, 3.0, 3.5), (15.0, 18.0, 20.0))
#: Stage 3 fallback (only if no stage-1 cell passes): initial speed cap 0.15 m/s.
STAGE3 = _grid((3.0, 3.5), (18.0,), speed=0.15)
REGIMES = {r.name: r for r in STAGE1 + STAGE3}


def regime_environment(base: SE3RendezvousConfig, regime: Regime) -> SE3RendezvousConfig:
    return replace(
        base,
        phase2_target_tumble_scale=float(regime.tumble_scale),
        precapture_initial_range_min_m=float(regime.range_min_m),
        precapture_initial_range_max_m=float(regime.range_max_m),
        precapture_initial_inertial_speed_max_m_s=float(regime.speed_max_m_s),
    )


__all__ = ["NOMINAL_TUMBLE_DEG_S", "REGIMES", "STAGE1", "STAGE3", "Regime", "regime_environment"]
