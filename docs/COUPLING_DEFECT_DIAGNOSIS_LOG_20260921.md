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

