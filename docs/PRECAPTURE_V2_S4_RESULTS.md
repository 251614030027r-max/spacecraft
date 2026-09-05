# Precapture v2 S4 hand-guidance rework manifest

Date: 2026-09-05
Lower layer: MPC h20, terminal weight 1000
Public upper-layer action: target-centred inertially oriented 3D waypoint, 2 s hold

## Frozen rule after S4-R

The rule receives only the current controller-visible relative state and target
pose/rate. It creates a nominal free-rigid-body forecast; it never reads
`env._target_trajectory`, future truth, or per-seed offline optimisation.

Two changes were made and are now frozen:

1. Window timing is the first interior local minimum of the angle between the
   predicted approach axis `R(t) a` and the chaser's initial inertial direction.
   The fixed 40 s match starts at that predicted time.
2. MPC plans against a fixed 45 degree FOV half-angle while the environment and
   all reported truth margins retain the real 50 degree half-angle.

The rule holds the initial radius during angular repositioning, then hands the
terminal waypoint to MPC. The external action remains 3D; the lower MPC previews
motion by differencing consecutive accepted waypoints.

## Development freeze block

Seeds 262100--262104 were run once before the reporting block. No parameter was
changed after seeing this block.

- completion: 2/5; zero-active-constraint episodes: 5/5;
- failures: distance 3 / illegal-entry episodes 1 / timeout 0 / corridor 0 / speed 0;
- successful completion time: mean 227.5 s, range 215.8--239.2 s;
- successful equivalent delta-v: mean 4.797 m/s, max 6.177 m/s;
- worst real-geometry FOV margin: +0.5580 rad;
- minimum normalized truth margin across episodes: +0.1499;
- serial single-process command time: mean 0.0663 s, p95 0.1160 s,
  max 0.3489 s;
- selected first-local-minimum times: 8.6, 26.7, 28.7, 34.2, and 127.6 s.

Artifact:
`logs/precapture_planning_v2/hand_guidance_s4r_dev_262100_5seeds.json`.

The 45 degree planning margin removed the prior FOV-edge failure. Three
action-driven outer-distance failures show that fixed-radius repositioning at
17--19 m is physically expensive; this is a development result, not a reason to
retune the frozen rule on the reporting seeds.

## Known conservatism of the fixed rule

These choices are intentionally left unresolved and must not be silently added
to the classical baseline:

1. Reposition radius is fixed at the initial radius; it does not decide to
   descend before repositioning. For the same angle, moving at 6 m instead of
   16 m reduces the idealised time and delta-v by about 39 percent under
   `T = theta sqrt(r/a)` and `delta-v = theta sqrt(a r)`, `a = 5/106`.
2. Only the first predicted local-minimum window is used. It does not compare
   first versus second window cost; the known development examples have mixed
   preference, so a fixed first-window rule necessarily chooses poorly in part
   of the distribution.
3. Match duration is fixed at 40 s and is not adapted to angle or radius.
4. Outer descent behaviour is fixed and is not adapted to the remaining time
   budget.
5. Entry timing and fuel are not explicitly traded against one another.

## Evidence boundary

The earlier 262000--262004 result at commit `b6c388d` used a fixed 152.4 s wait
and is superseded as the S4 baseline. Five-seed batches are development probes,
not the required 20-episode formal report.

## Frozen reporting block

Seeds 262000--262004 were run after the development block without changing any
rule or parameter.

- completion: 3/5; zero-active-constraint episodes: 5/5;
- failures: distance 2 / illegal-entry episodes 2 / timeout 0 / corridor 0 / speed 0;
- successful completion time: mean 252.667 s, range 224.9--298.2 s;
- successful equivalent delta-v: mean 6.079 m/s, max 6.677 m/s;
- worst real-geometry FOV margin: +0.5603 rad;
- minimum normalized truth margin across episodes: +0.1497;
- serial single-process command time: mean 0.0501 s, p95 0.0608 s,
  max 0.2813 s;
- first local-minimum times: 0.9, 14.2, 15.2, 113.8, and 117.5 s.

Artifact:
`logs/precapture_planning_v2/hand_guidance_s4r_report_262000_5seeds.json`.

Verdict: **FAIL** as a stable classical baseline. R-2 successfully removed the
FOV-edge violation, but R-1 combined with fixed-radius repositioning exposes
action-driven outer divergence in two reporting episodes. Per the frozen-block
rule, this result does not trigger another S4 tuning pass and does not block the
separate S5 interface test.
