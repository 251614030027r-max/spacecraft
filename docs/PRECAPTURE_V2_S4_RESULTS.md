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

The public `external_local` action remains exactly the declared 3D inertial
position waypoint held for 2 s. The lower MPC now estimates inertial waypoint
velocity from two consecutive accepted 3D commands and previews that motion
inside its horizon. This was required because treating a moving body-fixed
waypoint as inertially stationary for 2 s made its target-frame reference drift.
With internal motion preview seed 262000 completed in 222.2 s; the otherwise
identical stationary-hold smoke timed out at 300 s.

Five-seed result:

- completed: 4/5; truth-zero-violation completed: 4/5;
- successful completion time: mean 219.900 s, range 216.4--222.2 s;
- successful equivalent delta-v: mean 3.089 m/s, max 3.920 m/s;
- serial MPC command time: mean 0.0611 s, p95 0.0689 s, max 0.3587 s;
- four successful episodes had no active-constraint violation;
- seed 262004 terminated at 73.6 s on an FOV margin of -0.000458 rad;
- the aggregate development acceptance rule passes at its 0.80 threshold, but
  this batch is not 5/5 safe and must not be reported as such;
- every seed logged one illegal entry event before the later legal completion
  in the four successful episodes; illegal entry does not latch or terminate.

Artifact:
`logs/precapture_planning_v2/hand_guidance_h020_tw1000_3d_5seeds.json`.

Two diagnostic attempts to slow the outer descent and to select a lower-rate
later window did not remove seed 262004's FOV failure, so those variants were
not frozen and their scratch outputs are excluded from the commit. S5 can use
the same 3D interface for offline waypoint/switch-time optimisation; S4 itself
remains an honestly negative 4/5 development baseline.
