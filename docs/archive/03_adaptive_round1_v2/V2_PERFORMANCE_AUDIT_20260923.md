# V2 formal-training performance audit (2026-09-23)

## Scope and provenance

- Profiler commit: `9f52a5ddb9936618ef68c6af3edcee23ee5a4f80`
- Configuration: `task_state_v2`, adaptive task, horizon 35, execution feedback,
  monotone commit, CLARABEL, 20 control steps per high-level decision.
- No formal training was started. The steady-update runs set
  `learning_starts=0` only to measure the production SAC update cost.
- Local platform: Ryzen 5 5600H. The cloud figure available before this audit is
  the supplied AutoDL Xeon 8352V 1000-step rollout-only benchmark.
- Table entries are steady means after discarding the first complete high-level
  decision. Times are seconds per high-level decision.

## Single-process breakdown

| Platform | Mode | s/decision | MPC command | `problem.solve()` wall | `solver_stats.solve_time` | wrapper approx. | pre-solve setup | constraint lin. | model lin. | truth RK45 | drift RK45 | SAC update |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Local 5600H | rollout-only | 1.656 | 1.486 | 0.560 | 0.466 | 0.095 | 0.597 | 0.393 | 0.045 | 0.132 | 0.013 | — |
| Local 5600H | steady-update | 1.691 | 1.516 | 0.574 | 0.474 | 0.101 | 0.607 | 0.398 | 0.046 | 0.135 | 0.014 | 0.023 |
| Cloud 8352V | rollout-only, prior 1000-step run | 2.964 | — | — | — | — | — | — | — | — | — | — |
| Cloud 8352V | steady-update | not run | — | — | — | — | — | — | — | — | — | — |

The local steady-update ratios are:

- Gate A: `solver_stats / problem.solve = 0.825`, so Gate A **passes**.
- Gate B: `wrapper / problem.solve = 0.175`, so Gate B **fails**.
- Gate C: `(MPC - problem.solve) / MPC = 0.621`, so Gate C **passes**.

The largest measured non-solver block is pre-solve setup (0.607 s), principally
constraint linearization (0.398 s), followed by Parameter/Python-loop overhead
(approximately 0.163 s) and model linearization (0.046 s). This identifies the
only justified implementation-optimization target; it does not authorize a
mathematical or solver change. Persistent Clarabel is not justified because
Gate B fails.

## Three-process throughput

| Platform/configuration | Aggregate decisions/s | Equivalent 60k x 3 wall | CPU/RAM peak |
|---|---:|---:|---:|
| Local, defaults | 1.454 | 34.38 h | not instrumented |
| Local, `OMP/OPENBLAS/MKL_NUM_THREADS=1` | 1.600 | 31.26 h | not instrumented |
| Local, thread-limit repeat (different profiler seed) | 1.605 | 31.16 h | not instrumented |
| Cloud 8352V | not run | single-process lower bound is 49.4 h/seed | not recorded |

The thread limit reproduced: two runs gave 1.600 and 1.605 decisions/s versus
1.454 at defaults. Use it for any later three-process run. A one-core affinity
check improved single-process training wall time from 53.06 s to 51.35 s
(3.2%), below the 10% continuation threshold, so affinity work stops here.

## Evidence files

Generated outside the Git repository to avoid adding benchmark output to
history:

- `C:\Users\35884\Documents\Spacecraft\v2_profile_local_p1.json`
- `C:\Users\35884\Documents\Spacecraft\v2_profile_local_p2.json`
- `C:\Users\35884\Documents\Spacecraft\v2_profile_local_p3.json`
- `C:\Users\35884\Documents\Spacecraft\v2_profile_local_p3_threads1.json`
- `C:\Users\35884\Documents\Spacecraft\v2_profile_local_p3_threads1_repeat.json`
- `C:\Users\35884\Documents\Spacecraft\v2_profile_local_p4_affinity.json`

## Decision

1. The main bottleneck is the serial MPC path, not SAC; steady SAC update cost is only 0.023 s/decision.
2. Gates A and C pass; Gate B fails. Do not start persistent-Clarabel work.
3. Verified local thread limiting reduces the three-seed projection from 34.38 h to about 31.2 h.
4. CPU affinity provides less than 10% and is rejected.
5. Local and the current cloud instance both miss the 20-24 h gate; do not start formal 3 x 60k on either yet.
6. Stop the current AutoDL instance for formal training: its measured single-process rollout is 79% slower than local (2.964 versus 1.656 s/decision).
7. If another optimization round is authorized, target duplicated/fine-grained constraint-linearization and Parameter-loop work locally, with paired control/trajectory regression and before/after throughput proof.
