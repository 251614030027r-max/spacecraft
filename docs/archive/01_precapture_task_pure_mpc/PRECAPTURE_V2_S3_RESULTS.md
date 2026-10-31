# Precapture v2 S3 development results

Date: 2026-09-05
Seed block: 262000--262004
Episode limit: 300 s
Terminal weight: 1000 (frozen)

These are five-seed development results on the M1--M3 geometry. They are not
the required 20-episode formal report. They predate the predictive entry-plane
activation added after Q1 and are retained only as pre-F1 comparison data; they
are not the current Pure-MPC baseline.

| Controller | Completed | Zero-truth-violation completion | Timeout | Illegal entry events | Mean / p95 command time |
|---|---:|---:|---:|---:|---:|
| Pure MPC h20 | 4/5 | 4/5 | 1/5 | 2 | 0.0516 / 0.0590 s |
| Pure MPC h50 | 1/5 | 1/5 | 4/5 | 4 | 0.1269 / 0.1412 s |

Both batches had zero active-constraint violations. Horizon 50 exceeded the
0.1 s serial control budget at both mean and p95 and did not improve the
development completion rate. This is a measured result, not a claim that every
long-horizon formulation must underperform.

Artifacts:

- `logs/precapture_planning_v2/pure_mpc_h020_tw1000_5seeds.json`
- `logs/precapture_planning_v2/pure_mpc_h050_tw1000_5seeds.json`
- `logs/precapture_planning_v2/curve_a_divergence_seed99.json`

The curve-A artifact is a separate, constructed mechanism illustration required
by M3: at 16 m, the initial target-frame angular rate is 0.08 rad/s and the
applied wrench is zero. It reaches 30.083 m at 19.9 s and terminates as
`distance_failure`, not `outer_speed_failure`. Because the zero action is imposed
as an initial-condition diagnostic rather than produced by a controller, this
artifact is not an action-driven failure curve and must not be labeled as a
Pure-MPC rollout or used as the controller-performance denominator. A genuine
action-driven curve A remains unmeasured in S3.

The proposed 3 x 3 timing scan is **NOT DONE** as a reportable experiment. A
discarded diagnostic draft selected the same downstream window at all nine
points, so it did not identify a timing-cost surface. It is not committed
evidence and supports no comparative or optimality claim.
