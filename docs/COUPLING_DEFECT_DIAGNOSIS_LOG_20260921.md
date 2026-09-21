# Coupling defect diagnosis log

Date: 2026-09-21 JST. Branch:
`claude/sac-mpc-coupling-design-ns7g6i`.

## Superseded third nominal row

The 48-seed gated evaluation of `adp_rf_262412` was **actively stopped** after
26/48 episodes. This was not a model failure and no completed episode was
discarded. The algebraic absorbing-state mechanism had already been identified,
and the preregistered D1 instrumented roll-out supersedes the remaining nominal
episodes with a stronger causal check. The partial artifact is retained at:

`eval/adp/proposed_nominal_adp_rf_262412.json.partial`

- covered evaluation seeds: 262000--262025;
- SHA-256: `53ab60de5d02f474158f56329057386a900ef207202dbe0713fa883cc43d2d7a`;
- stopped before diagnostic implementation; generating commit: `32fe20ef218295811fb3b9369401b996fe6ec982`.

## Diagnostic implementation

- C1 stochastic policy plus latch/effective-reference instrumentation:
  `d424012` and deterministic stochastic seed follow-up `939916a`.
- C2 scripted staged-residual arm and explicit episode-seed list: `bb5f08c`.
- C3 replay support for adaptive task and baseline residual: `26bddff`;
  one-episode mainline smoke succeeded with model 262410 / evaluation seed
  262000 (`eval/adp/c3_replay_smoke_262410.json`).
- T-A/T-B/T-C strict expected failures: `a963f2f`; result was
  `2 passed, 3 xfailed`.

The implementation commits through `939916a` were pushed before diagnostic
roll-outs. D0 is a post-hoc estimate from each final policy's stochastic
distribution; it must not be described as a measurement captured during
training.

## D0 -- post-hoc training-reachability estimate

All three final policies were rolled out stochastically for 12 episodes on
seeds 262000--262011. This samples the final policies' stochastic
distributions; it is **not** a trace retained from training.

| model | latch Q1 / median / Q3 | latch=0 | null | effective reference changed | latch cause |
| --- | --- | ---: | ---: | ---: | --- |
| 262410 | 0 / 0 / 0 | 12/12 (100%) | 0 | 0/973 (0%) | policy 12/12 |
| 262411 | 0 / 0 / 0 | 12/12 (100%) | 0 | 0/973 (0%) | policy 12/12 |
| 262412 | 0 / 0 / 0 | 12/12 (100%) | 0 | 0/973 (0%) | policy 12/12 |

The episode decision count is 973 for each model because every policy locks to
the same Pure-MPC reference on decision zero and therefore reproduces the same
8 completions and 4 failures. This satisfies the preregistered D0 threshold
(`median <= 2`) by the strongest possible margin and supports R2. Per the
protocol, R2 remains an indicated root cause rather than a final unique verdict
until the required cross-reading is complete.

Artifacts (generated from pushed commit `2dea3dc`, whose code includes C1--C3
and the stochastic seeding fix):

- `eval/adp/d0_stochastic_262410.json` -- SHA-256
  `423952da57beb21345e1f815d43c1109c287939bd3180f89b30985dfb1e766e0`;
- `eval/adp/d0_stochastic_262411.json` -- SHA-256
  `66c2629d3f3cc426b8c180e14ae46a230245b0601ef1308019989f03bbb2bb08`;
- `eval/adp/d0_stochastic_262412.json` -- SHA-256
  `c55e56dc50f80ca05ec047b8afb3b0524232aabb821d21707a4d73e275e801c8`.
