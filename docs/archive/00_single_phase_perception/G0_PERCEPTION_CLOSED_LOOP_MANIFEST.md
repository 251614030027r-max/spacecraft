# G0 perception closed-loop manifest

Status: **PASS after the one authorized model correction**
Pre-registration date: 2026-09-02
Parent: A1 `phase2_perception_environment_config()` at commit
`be101cab32da422410818c4a8c5c37a10253cf3e`

## Frozen factor and comparison

G0 keeps the A1 task, geometry, reward, camera, EKF, 29D schema, continuous
actuators, MPC configuration and corridor-guidance reference unchanged.  The
only evaluated-controller factor is its state source:

- `estimate`: `env.observed_relative`, with the target state reconstructed from
  that estimate and the known chaser state;
- `oracle`: `env.relative` and the corresponding truth target state.

Truth remains available only to environment propagation, reward, termination,
real-geometry constraint judging and evaluation metrics.  No training is
authorized.  No camera, EKF covariance/noise, reward, schema, geometry, window,
obstacle or actuator change is authorized.

The controller is the existing fair Pure MPC row: horizon 50, one outer
iteration, exact linearization refreshed every 10 control steps, CLARABEL,
fixed-quadratic terminal cost, and `corridor_guidance` reference at speed
fraction 0.6.

## Fixed evaluation design

Run both sources on three disjoint 20-episode blocks with base seeds `262000`,
`20288000`, and `20290000`.  Within each block, the two sources share episode
seeds and every configuration except state source.

Fixed outputs are:

1. truth-geometry zero-violation completion rate;
2. position, attitude, translational-velocity and angular-velocity estimation
   error p50/p95 versus time;
3. 12D NEES p50/p95 versus time using the full EKF covariance;
4. visible-feature count and longest continuous interval without an accepted
   measurement;
5. controller-command compute p95/max and fraction above the 0.1 s period;
6. real-geometry constraint margins and representative trajectories.

## Decision rule frozen before execution

G0 passes only if all of the following hold:

- aggregate estimate zero-violation completion rate is no more than 0.05 below
  oracle;
- aggregate estimate raw completion rate is no more than 0.05 below oracle;
- in each 20-episode block, estimate has at most one fewer zero-violation
  completion than oracle;
- estimate introduces no systematic violation: no violation type occurs in at
  least two more estimate episodes than oracle over the aggregate 60 paired
  episodes.

These tolerances are locked and will not be changed after results are read.
Compute is reported, not used to silently change the controller.

If G0 fails at a continuously visible operating point and the failure is
localized to a biased propagation/coordinate model, exactly one time-boxed
model/coordinate correction is allowed, followed by a full rerun.  Process
noise tuning is forbidden.  If failure remains, stop before A2 and report for
upper review.  Weak observability with large but unbiased uncertainty is a
physical result, not a bug to tune away.

## Executed correction and audit boundary

The frozen A1 local mean predictor failed the first complete paired diagnostic
block even though all five features were measured at every step. Oracle
completed 20/20 without a truth-geometry violation; estimate completed only
3/20, with 16 total-speed violation episodes. Estimate pooled position error
was 0.995/4.398 m p50/p95, velocity error was 0.194/0.443 m/s, and 12D NEES was
1646/11679. This localized the failure to biased/inconsistent propagation, not
loss of visibility.

Exactly one correction was then applied: reconstruct the target estimate from
the estimated relative state and known chaser state, and propagate the EKF mean
with the same absolute rigid-body, gravity, J2 and integration model family as
the environment. The A1 local Jacobian and every covariance/noise value were
left unchanged. No second correction or result-driven tuning was performed.
The failed diagnostic is retained locally under
`logs/g0_perception_closed_loop/diagnostic_pre_correction/`; formal corrected
logs are local-only and excluded from Git.

## Formal corrected result

All six files were complete and configuration-audited: one unique environment
hash, one unique MPC hash, nominal controller model, horizon 50, exact
linearization, corridor guidance and the frozen 29D perception schema. Every
block contained its declared 20 unique consecutive seeds.

| source | completed | zero-violation completed | position p50/p95 (m) | attitude p50/p95 (rad) | velocity p50/p95 (m/s) | angular velocity p50/p95 (rad/s) | NEES p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | 60/60 | 60/60 | 0.0430 / 0.4412 | 0.00922 / 0.04487 | 0.00365 / 0.07765 | 0.000864 / 0.00832 | 6.27 / 70.77 |
| estimate | 60/60 | 60/60 | 0.0432 / 0.4531 | 0.00924 / 0.04662 | 0.00365 / 0.07769 | 0.000865 / 0.00833 | 6.26 / 72.09 |

Each paired block was 20/20 estimate versus 20/20 oracle for zero-violation
completion. No corridor, FOV, total-speed or closing-speed violation occurred.
Worst estimate truth margins remained positive: axial 1.704 m, lateral
1.181 m, FOV 0.324 rad, total speed 0.0587 m/s and closing speed 0.0423 m/s.
Median completion time was 89.4 s estimate versus 88.3 s oracle; median force
impulse was 208.1 versus 164.8 N s.

Five of five features were visible and accepted at every sample; the longest
no-measurement streak was 0 s. The high NEES tail is concentrated in the
initial convergence transient (time-step p95 peak about 610 at 0.1 s) and falls
toward the 12D reference by about 30 s. This remains a covariance-consistency
limitation and is not hidden or tuned away, but it did not cause a formal safety
or completion gap.

The three seed blocks were run concurrently at the user's request. Therefore
the measured controller compute tail is a loaded-host result, not a clean
single-process real-time benchmark: oracle p95/max 316/1604 ms with 89.1% over
0.1 s; estimate 310/1431 ms with 80.0% over 0.1 s. A serial compute profile is
required before making an intrinsic real-time claim; no safety result depends
on this caveat.

All four locked acceptance checks pass. G0 therefore freezes the corrected A1
perception chain as safe enough for the A2 starting point. It does not prove
perfect estimator calibration, active-perception benefit, or a learning gap.
