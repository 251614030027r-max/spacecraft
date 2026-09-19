# D0: hybrid MPC infeasibility diagnosis

This is the no-training gate requested in
`HANDOFF_UPPER_RULING_20260908.md` sections 6, 7, and 11.  All evaluations
used the same seeds 262000--262019, horizon 20, the `radial_local` action
parametrisation, the historical zero-wrench infeasibility fallback, and the
default constraint slack limit.  They ran serially in one process.  The three
models are the preserved `interrupted_model.zip` artifacts from the stopped
v2 training runs; the fourth row samples the same action space uniformly.

## Added episode record fields

- `qp_infeasible_steps_total`: number of 0.1 s control steps whose QP status
  starts with `infeasible`.
- `first_infeasible_time_s`: pre-command environment time of the first such
  step, or `null` if none occurred.
- `max_consecutive_zero_wrench_steps`: longest run of control steps for which
  the actual six-dimensional wrench command was exactly zero.

The QP status and actual wrench are deliberately measured separately.  In
these four runs they select the same set of episodes, but the record does not
assume that every possible solver error is an infeasibility or that every zero
wrench must be a fallback.

## Results

| control | completed | episodes with infeasible QP | infeasible control steps | episodes with zero-wrench streak >=20 | maximum zero-wrench streak | earliest first infeasible | failures with no infeasible step |
|---|---:|---:|---:|---:|---:|---:|---:|
| stopped model 262100 | 0/20 | 8/20 | 2,016 | 7/20 | 402 | 0.8 s | 12/20 |
| stopped model 262101 | 0/20 | 6/20 | 1,477 | 6/20 | 391 | 1.0 s | 14/20 |
| stopped model 262102 | 0/20 | 7/20 | 2,164 | 6/20 | 562 | 0.9 s | 13/20 |
| **three stopped models** | **0/60** | **21/60** | **5,657** | **19/60** | **562** | **0.8 s** | **39/60** |
| random action | 0/20 | 15/20 | 8,509 | 15/20 | 1,834 | 0.8 s | 5/20 |

Per episode, the stopped models average 94.3 infeasible steps against 425.5
for random actions.  Their episode incidence is 35% against 75%.  The learned
policies therefore avoid many references that make the h20 QP infeasible;
the action-space change and partial learning are not numerically inert.

However, all 39 model episodes with no infeasible control step also fail, and
only 19/60 model episodes contain a zero-wrench streak of at least one full
2 s decision.  QP infeasibility plus zero fallback is consequently a real and
sometimes catastrophic failure channel, but it is not a sufficient
explanation for the stopped models' 0/60 completion result.  D0 does not
separate inadequate exploration from reward credit assignment.  It does rule
out the claim that the observed absence of completion is solely downstream of
MPC infeasibility, and it supports proceeding to the already ruled single
factor D1 macro-potential correction rather than reopening fallback or slack
experiments.

These are full-state results (`precapture_planning_full_state_v1_24d`,
`perception=null`).  They are not local-vision or EKF evidence.

## Evidence

- `logs/hybrid/d0_20260908/model_262100_20ep.json`
- `logs/hybrid/d0_20260908/model_262101_20ep.json`
- `logs/hybrid/d0_20260908/model_262102_20ep.json`
- `logs/hybrid/d0_20260908/random_20ep.json`

No training was run for D0, and none of the four evaluations changed the
models, replay buffers, MPC fallback, or slack limit.
