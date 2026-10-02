# Why the 3D waypoint interface does not steer the precapture MPC

Measured 2026-09-05 in the upper-level sandbox, on `1fe5549` (**not** on
`e212b4a` -- see *What this does not cover*). One episode per row, `h20`,
`terminal_weight=1000`, `input_weight=0.01`, `analytic_local`, oracle
`CoastThenMatchPlan` delivered through the declared interface: one
target-centred, inertially oriented 3D waypoint held two seconds. Truth drives
the environment and every margin below; only what the controller is asked to
track changes. Runs were executed in parallel, so **no compute number is
reported from them** -- serial single-process timing is the only admissible
real-time evidence and none was taken here.

Reproduce with `experiments/diagnose_precapture_reference.py`.

## 1. The question this answers

The S5 stop asked whether the h20 lower layer can enforce attitude and
field-of-view at all, and whether a 3D translational waypoint can influence
that behaviour. The evaluation path never recorded the one quantity that
decides it: **how far the chaser actually is from the waypoint it was given.**
That is now recorded per step, split into radial and tangential parts.

## 2. The chaser does not fly the waypoint it is handed

`frozen` attitude reference, oracle waypoint, three seeds, two settings of the
position entry of `state_scales`:

| seed | position scale | end | r_end | lag median | of which tangential | lag max | FOV max | latched | illegal entries | violation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 262001 | 20 m (default) | 299.9 s cap | 3.41 m | **4.32 m** | 2.60 m | 7.28 m | 15.26 deg | no | 2 | none |
| 262001 | 10 m | 299.9 s cap | 3.02 m | 1.61 m | 1.32 m | 4.63 m | 16.00 deg | no | 2 | none |
| 262002 | 20 m (default) | 299.9 s cap | 6.34 m | **3.32 m** | 2.53 m | 9.93 m | 16.12 deg | no | 2 | none |
| 262002 | 10 m | 299.9 s cap | 2.99 m | 2.03 m | 1.53 m | 7.25 m | 34.08 deg | no | 2 | none |
| 262004 | 20 m (default) | 98.9 s | 2.79 m | **6.35 m** | 4.50 m | 7.91 m | **49.75 deg** | no | 1 | FOV, 1 step |
| 262004 | 10 m | 86.1 s | 5.83 m | 3.69 m | 2.78 m | 5.52 m | **49.99 deg** | no | 1 | FOV, 1 step |

Two things follow immediately.

**The interface is being diluted, not used.** At the default scale the chaser
sits a median 3.3-6.4 m from the waypoint it was just handed, on a task whose
whole working range is 17 m down to 3 m. Decomposed against the chaser's own
radial direction on 262001, the median lag is **-3.08 m radial** (the chaser is
*outside* its own waypoint, behind on the descent) and **2.60 m tangential**,
which at that range is **22.0 deg of co-rotation phase behind**. 60% of the
error is phase. An upper layer -- learned or oracle -- cannot steer a lower
layer that answers a waypoint with a 22 deg phase lag.

**Most of that lag is cost normalisation, not actuation.** The position entry
of `state_scales` is 20 m. A 4 m position error therefore normalises to 0.2,
while the angular-rate entry is 0.05 rad/s, so a 0.02 rad/s rate error
normalises to 0.4 -- the objective weighs a small rate error twice as heavily
as being 4 m off the waypoint. Halving the position scale to 10 m cuts the
median lag by 63% / 39% / 42% on the three seeds. **This is a diagnostic
sweep, not an adopted setting**: 10 m was not evaluated as a factor, it has no
three-seed formal row behind it, and on 262002 it raised peak FOV from 16.1 to
34.1 deg. It is reported because it identifies where the lag comes from.

## 3. The FOV failure is a close-range saturation, not a pointing bug

Seed 262004 is the seed that loses the camera. The last twelve seconds, scale
20 m:

| t | r | FOV | tangential lag | relative rate | torque | force |
|---:|---:|---:|---:|---:|---:|---:|
| 87.0 s | 4.62 m | 12.8 deg | 6.75 m | 0.029 rad/s | 0.02 N*m | 3.19 N |
| 91.8 s | 3.45 m | 18.2 deg | 6.39 m | 0.041 rad/s | 0.12 N*m | 4.17 N |
| 93.0 s | 3.23 m | 20.8 deg | 6.14 m | 0.041 rad/s | **1.04 N*m** | **8.63 N** |
| 95.4 s | 2.88 m | 30.2 deg | 5.89 m | 0.094 rad/s | 0.85 N*m | 5.90 N |
| 97.8 s | 2.72 m | 44.7 deg | 6.07 m | 0.098 rad/s | 0.84 N*m | **8.66 N** |
| 98.9 s | 2.79 m | **49.7 deg** | 6.20 m | 0.058 rad/s | 0.86 N*m | **8.66 N** |

The chaser arrives at 3 m still **6 m off in phase**. The port sits 1.5 m from
the target centre, so the sightline's angular rate grows as `1/r`: the same
tangential error that was 5 deg of pointing at 16 m is tens of degrees at 3 m.
The FOV angle goes 12.8 -> 49.7 deg in twelve seconds, and it accelerates.
Through the last six of them the force is at **8.66 N, the full three-axis
envelope (5*sqrt(3))**, and the torque touches **1.04 N*m, the full three-axis
torque envelope (0.6*sqrt(3))**.

So the answer to "why does a 40 deg planning cone not change the control" is
that by then **nothing can**: the actuator is saturated in both channels and
the required slew is larger than +-0.6 N*m per axis can deliver. The
constraint is not being ignored and the row is not inert -- the vehicle has run
out of authority. Tightening the planned cone cannot buy back a slew rate the
thrusters cannot produce. The lever that works is upstream: do not arrive at
3 m with 6 m of phase error.

This is an internal, action-dependent failure. Nothing was scripted into it.

## 4. The attitude reference was suspected, and is not the cause

`_apply_precapture_attitude_reference` and the `external_local` branch aim one
attitude from the current state, hold it across the horizon, and leave the
relative-rate rows at zero. Reference positions are written in the *target*
body frame, so a waypoint held fixed inertially sweeps through that frame at
the target's tumble rate (0.0412 rad/s, 2.36 deg/s), and the old code comment
justified the zero rate rows with "a constant desired relative attitude means
zero relative rate". That looked like a feed-forward omission, so two
alternatives were built and measured rather than argued:

- `swept` -- index 0 unchanged, later indices carried by the rotation the
  *reference* sightline undergoes, rate rows differenced from that sequence;
- `aimed` -- every index aimed from its own reference position, so the
  reference is one self-consistent trajectory.

Seed 262001, oracle waypoint, scale 20 m:

| mode | FOV max | lag median | attitude error median | mean torque | mean force |
|---|---:|---:|---:|---:|---:|
| **frozen** (default) | **15.26 deg** | 4.32 m | 6.40 deg | 0.0165 N*m | 1.49 N |
| swept | 40.85 deg | 3.06 m | 8.36 deg | 0.0353 N*m | 1.26 N |
| aimed | 49.29 deg | 6.52 m | 13.18 deg | 0.3454 N*m | 2.17 N |

**The existing mode is the best of the three, and the ordering is monotone in
how much of the reference position the attitude trusts.** The reason is the
lag measured in section 2: aiming the camera from the waypoint, when the
chaser is 4-7 m away from it, points the boresight at a place the port is not.
`aimed` costs 21x the torque of `frozen` and ends the episode 11 m out.

The hypothesis is therefore falsified, and the honest conclusion is that
`frozen` should stay. The two alternatives are kept as a validated option --
default `"frozen"`, so nothing already measured moves -- because if the lag is
ever closed, the feed-forward argument becomes live again and the experiment
should be one flag, not a rewrite.

## 4b. The interface itself was the defect, and it is now measured as lossless

Sections 2-4 all measure the *inertially oriented* waypoint contract. A
control the evaluation path never ran settles what that contract costs. Send
the MPC the one point it already flies well -- the task's own desired pose --
and change nothing but how it is delivered. Three seeds, `h20`, scale 20 m:

| how the same point is delivered | completed | completion time | FOV max |
|---|---:|---|---:|
| directly, as the `fixed` reference (no upper layer) | **3/3** | 121.8 / 115.8 / 126.5 s | 12.3 / 7.0 / 8.0 deg |
| through the 3D waypoint channel, **inertially oriented** | **0/3** | all hit the 299.9 s cap at r = 3.9 / 3.7 / 3.6 m | 32.7 / 10.2 / 32.4 deg |
| through the 3D waypoint channel, **target body frame** | **3/3** | 121.8 / 115.8 / 126.5 s | 12.3 / 7.0 / 8.0 deg |

The third row is not merely equal to the first, it is **bitwise identical**:
max `|dr|` over the full episode is `0.000e+00 m` on all three seeds. So the
channel is capable of carrying a reference losslessly, and under the declared
contract it was not.

The mechanism is visible in the endgame of the failing row. The desired pose
is body-fixed on a target turning at 0.0412 rad/s, so an upper layer that
wants the chaser there must re-issue a *rotating* inertial waypoint every
step. The channel holds each one for 2 s and maps it forward across the
horizon with `so3_exp(-index * dt * omega)`, which turns the stationary goal
into a circular reference. The optimiser then tracks a circle with a standing
lag: on seed 262001 the last 60 s sit at a **1.33 m mean offset** (radial
-1.05 m, tangential +0.95 m) with a 0.08 m sawtooth whose autocorrelation
peaks at exactly **20 steps = 2.0 s = the hold period**. The completion
tolerance is **0.25 m**. The chaser parks at 3.90 m and stays there for the
rest of the episode.

`external_reference_frame` now selects the contract. The default stays
`"inertial"`, so nothing already measured moves.

### What this does to the S5 stop

S5 stopped because a continuous oracle plan, delivered through this channel,
could not be made to work and neither could any waypoint family the search
found. Re-run with the frame corrected and nothing else changed:

| oracle plan through the channel | completed | completion time |
|---|---:|---|
| inertially oriented, scale 20 / 10 / 5 m | **0/9** | 300 s cap, one FOV loss, one escape to 29.95 m |
| target body frame, scale 20 m | **3/3** | 226.4 / 276.1 / 259.4 s |

**The S5 search was measuring the frame error, not the waypoint family.** No
conclusion drawn from it about what an upper layer can express survives.

### One thing this exposes that is not good news

With the channel transparent, the `fixed` row -- a myopic fixed-setpoint MPC
with no upper layer at all -- completes 3/3 in **116-127 s**, while the
scripted oracle plan through the same channel takes **226-276 s**. The oracle
is not an upper bound on this task; it is twice as slow as tracking the goal
directly. Any headroom for a learned upper layer has to be argued against the
116-127 s row, not against the oracle. Three seeds, no F1, truth state, so
this is a direction, not a row -- but it is the number the coupling design
now has to beat.

## 5. What changed in the code

| file | change |
|---|---|
| `controllers/mpc/config.py` | new `precapture_attitude_reference: str = "frozen"` over `{frozen, swept, aimed}`, and `external_reference_frame: str = "inertial"` over `{inertial, target}`; both default to the measured behaviour |
| `controllers/mpc/controller.py` | `_precapture_reference_rotations` / `_reference_angular_rates` build a per-index attitude sequence and the relative rate it implies; `frozen` reproduces the old path exactly. The `external_local` horizon map skips the inertial-to-target rotation when the frame is `"target"` |
| `tests/test_precapture_attitude_reference.py` | new -- pins both defaults, pins that `frozen` and `swept` are bitwise equal on a target-frame-stationary reference, pins that the swept rate rows carry the tumble rate with no parallax step in the first column, and pins that the target-frame channel reproduces the direct reference while the inertial one turns a stationary goal into a moving one |
| `experiments/diagnose_precapture_reference.py` | new -- the per-step lag / attitude / saturation recorder used above |

Regression evidence for "nothing already measured moves": a full `fixed`
reference h20 episode on seed 262001, 1219 steps, run before and after the
change -- **max |dFOV| = 0.000e+00 deg, max |dtorque| = 0.000e+00 N*m**, same
step count, same end state. `python -B -m pytest -q` is **181 passed**.

## 6. What this does not cover

- **This is `1fe5549`, not `e212b4a`.** F1's MPC entry preview is absent here,
  which is exactly the mechanism that should stop the illegal entries seen in
  every row of section 2. The lag and saturation findings are upstream of the
  entry plane and are not expected to move, but the entry and latch columns
  should be re-read on `e212b4a` before anything is concluded from them.
- **The reported FOV failure was not reproduced on its own seed.** On this
  commit seed 262001 does not lose the camera (15.3 deg peak); 262004 does.
  The 30.6 s / -0.001269 rad event belongs to code this sandbox does not have.
- One episode per cell, three seeds. Enough to locate a mechanism, not enough
  to report a row. Nothing here is a formal result.
- No compute number is taken from these runs.
