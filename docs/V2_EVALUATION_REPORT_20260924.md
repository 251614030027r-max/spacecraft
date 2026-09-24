# V2 formal evaluation report - 2026-09-24

## Executive result

The V2 channel stayed active, but nominal capability did not preserve the Pure MPC baseline and seed robustness failed.

Pure MPC completed **36/48**. V2-262410 completed **0/48**; V2-262411 completed **24/48**; V2-262412 completed **14/48**.

## Nominal paired outcomes

| model | retained | rescued | destroyed | both failed |
|---|---:|---:|---:|---:|
| 262410 | 0 | 0 | 36 | 12 |
| 262411 | 19 | 5 | 17 | 7 |
| 262412 | 10 | 4 | 26 | 8 |

No completed episode met the strict A criterion. Nearly all completed episodes remained in the 'other' bucket because the last-20 reference-step median stayed near the 0.40 m limiter; only one episode was B. Thus completion generally did not come from a learned low-step stop, and saturation remains the dominant interface signature.

Rescues exist for two seeds, so the learned layer can change outcomes; destruction is larger, so an architecture-level arbiter is necessary rather than optional.

## Q2 task-state stopping

| model | A | B | other | median rho gap (m) | rho-at-max fraction | median c |
|---|---:|---:|---:|---:|---:|---:|
| 262410 | 0 | 0 | 0 | None | None | None |
| 262411 | 0 | 1 | 23 | 0.07286684145983813 | 0.125 | 0.9977716399589553 |
| 262412 | 0 | 0 | 14 | 0.24930313501271506 | 0.35714285714285715 | 0.99903205037117 |

Across the 9 rescued model-episode pairs, 4 avoided illegal crossings entirely; 2 came from baseline cases whose first illegal crossing was beyond 10 m. Rescues therefore are not uniformly explained by merely avoiding the far-field plane sweep, but several still contain illegal crossings and cannot be described as clean entry-mode changes.

K3 shows retained V2 completions were always slower in the two non-dead seeds (minimum ratios above 1, medians above 2). Together with reference-step medians near 0.40 m, this is consistent with the limiter/active-reference interface slowing capture; it is not by itself a causal isolation of the limiter.

## Q5 channel activity and saturation

| model | changed fraction | step median m | step p95 m | min episode mean m | backward fraction |
|---|---:|---:|---:|---:|---:|
| 262410 | 1.000000 | 0.398535 | 0.399836 | 0.357050 | 0.548231 |
| 262411 | 1.000000 | 0.398453 | 0.399849 | 0.247424 | 0.459160 |
| 262412 | 1.000000 | 0.398443 | 0.399815 | 0.303113 | 0.492303 |

## Upper-review K1/K2/K3

K1: 5/13 illegal crossing events (38.5%) occurred beyond 10 m. Full rows are in REVIEW_K1_K3.md.

- 262410: rescued=0; retained timing ratio median=None, minimum=None (n=0).
- 262411: rescued=5; retained timing ratio median=2.0882838283828384, minimum=1.2956152758132957 (n=19).
- 262412: rescued=4; retained timing ratio median=2.352679781238864, minimum=1.4672686230248306 (n=10).

## Architecture floor and compute

The reject-all floor matched Pure MPC per control step for 12/12 episodes; max |delta wrench| = 0.0. The 262410 parallel/serial replay also matched 4/4 exactly.

Exclusive h35 profile (300 steps): mean 70.78 ms, p95 77.33 ms (0.773x the 100 ms period), max 384.80 ms; 2 steps exceeded budget. The maximum includes cold start.
The measured p95 headroom is 22.7% on this machine, not the older approximate 6% figure.

## Design handoff

1. Decide whether the Pure MPC anchor is a selectable point in the policy action space or an architecture-external bypass.
2. V2.5 bargaining/arbitration must fit the measured exclusive p95 compute margin; do not assume a second MPC solve is affordable.
3. Larger fresh seed blocks, a hand-written rule comparator, and EKF rows are deferred final-stage work, not additions to this diagnostic round.

## Evidence boundary

Nominal results support: the coupling acts, can rescue some baseline failures, and can also destroy many baseline successes. They do not support baseline preservation or seed-robust superiority.

Cross-condition/generalization evaluation was explicitly paused by the user and is excluded from this package and every conclusion. Partial tumble=0.10 files are not evidence and are not packaged.