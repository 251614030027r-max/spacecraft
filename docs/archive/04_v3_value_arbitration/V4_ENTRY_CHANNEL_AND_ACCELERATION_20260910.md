# V4 entry-channel diagnosis and training acceleration (2026-09-10)

## Gate result

The three V3 25k checkpoints were evaluated deterministically and serially on
seeds 262000--262019 (20 episodes each).  No training ran concurrently.  The
raw records are in `logs/hybrid/entry_channel_diagnosis_20260910/`.

| checkpoint seed | completed | crossings | illegal | radial violation | speed violation | closing violation | retry episodes | legal retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262100 | 0/20 | 21 | 15 | 100.0% | 26.7% | 33.3% | 2/20 | 1/6 |
| 262101 | 0/20 | 19 | 17 | 100.0% | 52.9% | 41.2% | 3/20 | 1/4 |
| 262102 | 0/20 | 18 | 18 | 100.0% | 50.0% | 16.7% | 3/20 | 0/5 |
| **aggregate** | **0/60** | **58** | **50** | **100.0%** | **44.0%** | **30.0%** | **8/60** | **2/15** |

This meets the preregistered radial-cause threshold (>60%) by a wide margin:
every illegal crossing violated the 3.151 m radial entry bound.  Retry episodes
were 13.3%, just above the <10% task-budget criterion, so the result does not
establish that retry is impossible within 300 s.  However, retry legality was
2/15 (13.3%), essentially no better than all-crossing legality (8/58, 13.8%).
The data therefore do not support merely extending V3 training.

## V4 single factor

V4 changes only the scheduling policy observation.  It retains the canonical
24D full-state core and appends:

- the first two inertial columns of the target rotation matrix (continuous 6D
  absolute-attitude/port-phase representation); and
- normalized remaining episode time in [0, 1].

The resulting policy observation is 31D.  The task definition, 300 s budget,
reward, action map and gains, 2 s decision period, MPC horizon/configuration,
truth dynamics, target phase sampling, and actuator path are unchanged.  The
legacy/default wrapper remains 24D so V3 checkpoints remain reproducible.

This remains a full-state result (`perception=None`); it is not evidence for a
local-vision/EKF SAC--MPC chain.

## Engineering acceleration

Training alone disables unused MPC post-solve diagnostics and uses exact
on-demand target RK45 propagation instead of eagerly integrating 300 s at each
reset.  The on-demand path calls the same `propagate_rk45` with identical
parameters, gravity, settings, step and time stamps.  `target_nfev` consequently
changes from the eager cache's synthetic zero to the real per-step integration
count and is not comparable across the switch.

On the same 20-decision single-process timing probe, the old switches measured
1.635 s/decision and the training switches measured 1.067 s/decision (1.53x).
Reset fell from 9.177 s to 0.003 s and mean MPC command time from 50.4 ms to
41.6 ms.  Three accelerated processes delivered 1.670 decisions/s aggregate,
1.96x the single-process aggregate throughput (0.850 decisions/s).  Concurrent
per-step latencies are intentionally not reported as real-time measurements.
The timing records are in `logs/hybrid/v4_performance_20260910/`.
