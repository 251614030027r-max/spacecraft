# Precapture v2 S4 hand-guidance development result

Date: 2026-09-05
Seed block: 262000--262004
Episode limit: 300 s
Lower layer: MPC h20, terminal weight 1000, 2 s high-level hold

The frozen hand baseline uses the oracle's outer-approach--rate-match--cross
structure but makes its timing online from controller-visible state. It builds
a nominal free-rigid-body forecast from the current target estimate, waits one
nominal tumble period (152.4 s), matches direction rate over the preceding
40 s, then switches to the terminal waypoint. It never reads the environment's
cached future truth and has no per-seed parameters.

`external_local` was extended compatibly from a 3D inertial position waypoint
to optionally accept a 6D inertial position/velocity waypoint. The original 3D
path remains unchanged. This was required because holding only an inertial
position for 2 s made a moving body-fixed waypoint drift inside every MPC
horizon. The 6D form completed seed 262000 in 220.2 s; the otherwise identical
3D smoke timed out at 300 s.

Five-seed result:

- completed: 4/5; truth-zero-violation completed: 4/5;
- successful completion time: mean 218.325 s, range 212.4--220.5 s;
- successful equivalent delta-v: mean 2.857 m/s, max 3.678 m/s;
- serial MPC command time: mean 0.0621 s, p95 0.0715 s, max 0.6069 s;
- four successful episodes had no active-constraint violation;
- seed 262004 terminated at 73.7 s on an FOV margin of -0.003706 rad;
- the aggregate development acceptance rule passes at its 0.80 threshold, but
  this batch is not 5/5 safe and must not be reported as such;
- every seed logged one illegal entry event before the later legal completion
  in the four successful episodes; illegal entry does not latch or terminate.

Artifact:
`logs/precapture_planning_v2/hand_guidance_h020_tw1000_5seeds.json`.

Two diagnostic attempts to slow the outer descent and to select a lower-rate
later window did not remove seed 262004's FOV failure, so those variants were
not frozen and their scratch outputs are excluded from the commit. S5 may now
use the common 6D interface for offline waypoint/switch-time optimisation; S4
itself remains an honestly negative 4/5 development baseline.
