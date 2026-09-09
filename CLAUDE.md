# Working notes for this repository

High-fidelity 6-DoF SE(3) pre-capture control for a tumbling non-cooperative
target. Chaser 106 kg, target 225 kg free-tumbling, 500 km / 45 deg circular
orbit, RK45 truth with central gravity, second moments, gravity gradient and
J2, 0.1 s control period, 200 s episode cap, +-5 N / +-0.6 N*m per axis.

## Current decision -- 2026-09-04 (A3 P1 negative; B gate closed)

A3 removes the nominal-inertia information advantage with one fixed prediction
model (`mismatch=0.20`, seed `260903`) shared by MPC and EKF while truth remains
nominal. It also adds an endpoint-only online reference and a deployment path
with analytic local linearization and no post-solve diagnostic rollout. The
prepared P1 rows are deployable online MPC, offline estimate/truth efficiency
bounds and an exact h50 non-real-time control.

Formal P1 is negative. Deployable online analytic-local h20 completed 60/60
with zero truth-geometry violations, median time 65.05 s and mean force impulse
139.87 N s. The offline episode-plan estimate row managed 59/60, 81.10 s and
184.77 N s; online was faster and cheaper in every block. Exact online h50 was
also 60/60 but essentially identical in quality (64.85 s, 139.52 N s) while
serial p95 rose from 40.25 to 207.11 ms. H10/h15/h20 sensitivity is flat. The
offline plan is not an efficiency upper bound and must be labelled only as a
reference. No long-horizon planning gap exists under current S1-v2, so B
training remains unauthorized. A single higher-Lambda or longer-range factor
requires upper approval before another probe. See
`docs/A3_DEPLOYABLE_PLANNING_P1_RESULTS.md`.

## Prior decision -- 2026-09-03 (A2 complete; learning gate closed)

G0 accepted the A1 perception chain after the physics-consistent EKF mean
correction. A2 is now implemented as
`perception_guidance_free_environment_config()`, derived from A1 with one
environment flag. Camera/EKF settings, 29D schema, local estimated-state input,
S1-v2 distribution, continuous actuators, truth dynamics and constraints are
frozen. The existing A1 29D core already carries estimated relative velocity
itself, so no observation schema change was needed. A2 changes terminal reward
velocity shaping from corridor guidance to zero terminal velocity and forbids
corridor guidance in its three realistic controller rows: fixed-terminal h10,
receding endpoint-plan h50, and explicit episode-plan plus h20 tracking. The
matched planning/tracking oracle uses truth only as a feasibility control; the
A1 h50 guided estimated-state row remains the control.

Implementation acceptance is **152 passed**. Formal A2 evaluation is complete:
the explicit episode plan plus h20 MPC completed 60/60 with zero truth-geometry
violations, matching oracle completion/time and beating A1 guidance on time and
force impulse. All successful trajectories kept 5/5 features visible with no
measurement gaps and matched estimate/oracle covariance, so the present task
has no active-perception gap. Strict compute p95 fails (52.17 ms mean, 163.02 ms
p95) because the exact refresh tail is about 123 ms. No training is authorized;
upper-level review must choose a separate compute stage or a physically
motivated sensing-hard factor. See `docs/A2_GUIDANCE_FREE_RESULTS.md`.

## Prior decision -- 2026-09-02 (A1 perception foundation complete)

The research direction is now locked to action-dependent local perception. A1
adds an independent five-feature pinhole-camera and 12D relative-state EKF chain
through `phase2_perception_environment_config()`. The camera uses the chaser +x
boresight, 1024x1024 image, 430 px focal length, 50 deg half-FOV, 30 m range and
1 px noise. The EKF state/error order is `[dtheta, dp, domega, dv]`; its mean
uses `LocalRelativePredictionModel` with the executed wrench, while transition
and image Jacobians use centred differences in the same local coordinates.

The A1 task is still the frozen `single_phase` S1-v2 task with the existing
corridor guidance, reward, truth dynamics, termination and constraint checks.
Only the observation source changes: the 29D `phase2_perception_v1_29d` schema
contains an estimated 24D core, four bounded log covariance summaries and
visible-feature fraction. Target angular velocity is reconstructed from known
chaser navigation plus the estimated relative state; truth remains confined to
dynamics, reward, termination, geometry and evaluation diagnostics.
`perception=None` takes the original path without drawing extra randomness, and
the compatibility test pins the initial observation and one-step transition
bitwise.

Acceptance is **149 passed**. The fixed seed 260902 perception smoke ran 300
steps with finite 29D observations and covariance, five visible features and a
measurement update on all 301 frames including reset; it reached the 30 s smoke
cap without early termination. This is a wiring check, not a formal estimation
or control result. No training was run. A2 (removing artificial guidance as a
separate task) is not implemented in A1; do not combine it with camera, EKF,
reward, geometry or S1-v2 changes. See
`docs/A1_PERCEPTION_FOUNDATION_MANIFEST.md`.

## Prior decision -- 2026-09-01 (superseded by the A1 direction above)

Four probes for a defensible SAC-MPC gap are now closed: learned terminal
value, sampled target phase, target-model mismatch, and target-state
observation error. After correcting an observation-error bug, all four are
negative. The bug rotated the target's inertial velocity together with an
attitude bias, injecting about 6.6 m/s of false relative velocity from the
7.6 km/s orbital velocity. The correct estimator rotates only the attitude and
keeps inertial position, velocity, and angular velocity unchanged.

With the correction, attitude bias is harmless through at least 5 deg in the
diagnostic frontier; delay and low update rate are compensated by a classical
model-prediction step. Sampled target phase and +-30% inertia mismatch do not
break short-horizon MPC either. The learned convex terminal cost is actively
harmful because its radial gradient cuts across the curved corridor, while a
fixed terminal cost at horizon 10 already preserves the long-horizon quality.

The warm `single_phase` probes remain historical negative evidence, but the
active task has moved to the full-state precapture-planning benchmark. The
current hybrid uses a 2 s `radial_local` waypoint decision over an h20 MPC.
D0 measured that infeasibility plus zero fallback is real but does not explain
all failures, and D1 moved only the potential term to the SAC decision boundary.
The current authority is `HANDOFF_UPPER_RULING_20260908.md`; measured D0 evidence
is in `docs/D0_HYBRID_INFEASIBILITY_DIAGNOSIS_20260909.md`. Historical pivots
are archived under `docs/handoffs/archive_202609/` and are not current execution
authority.

## Historical research line

The original target was a three-way comparison: Pure SAC, Pure MPC and an
architecture-uniform hierarchical SAC-MPC hybrid. If a hybrid is ever
reconsidered, it must still use one architecture across the whole mission;
switching architecture mid-episode would confound the method with a phase
switch. The four negative probes mean this is no longer an active build plan.

**The common benchmark is the `single_phase` task**, cleared for that
use by `eval/validate_single_phase_semantics.py` (20/20 scripted completion, all
five margins positive, one control law for the whole mission). The Waypoint was
diagnostic scaffolding, not a task element: it postponed the terminal
constraints until 8 m, which understates the problem (see *Co-rotation*). It
keeps its place as the instrument that localised the co-rotation failure, and
`full_mission` remains a valid secondary task -- Pure SAC reaching the Waypoint
there is what certifies the baseline implementation is competent. Phase-I is a
SAC *training* curriculum, not a headline result. Training may be staged;
evaluation may not.

**The entropy temperature was the driver of critic overestimation, and
pinning it fixed the critic.** `auto_0.005` never converged: alpha rose
monotonically in every run ever measured, and calibration error was monotone
in alpha across two observation schemas and two update-to-data ratios.
Raising `gradient_steps` to 4 only made alpha diverge faster (0.0041 -> 0.109
by 125k, tripping the abort threshold, calibration error 45.7), so that factor was
reverted. The likely cause is that `target_entropy = -6.0` is unreachable for
a 6-dimensional tanh-squashed Gaussian whose scale already saturates, so the
tuner raises alpha without bound and the entropy bonus, entering every
bootstrap over a 333-step effective horizon, inflates the soft targets.

| alpha | calibration error | run |
|---|---|---|
| 0.005 fixed | **-0.46** | fixed-alpha 100k |
| 0.005 fixed | **+0.94** | fixed-alpha 200k |
| 0.0037 | 4.9 | v3 200k |
| 0.0146 | 18.4 | v3 final |
| 0.0149 | 21.5 | v2 400k |
| 0.0203 | 32.2 | v2 final |
| 0.0605 | 45.7 | UTD 4 at 100k |

**With `ent_coef = 0.005` fixed and training stopped at 200k, Pure SAC
acquires the Waypoint.** Three training seeds, each evaluated deterministically on
the independent 20290000 block:

| seed | Waypoint 100k | error 100k | Waypoint 200k | error 200k | hard return 200k |
|---|---|---|---|---|---|
| 260815 | 6/20 | -0.46 | 1/20 | +0.94 | -4.20 |
| 260816 | 7/20 | -2.51 | **17/20** | -0.60 | -0.85 |
| 260817 | 11/20 | -3.50 | **20/20** | -3.56 | **+3.01** |

Seed 260817 matches the scripted Waypoint-PD reference (20/20) and comes within
0.65 of its discounted return (+3.01 against +3.66). The periodic evaluations
on the disjoint 20288000 block agree (0.8 and 1.0 at 200k), so this is not an
artefact of the evaluation seeds. Every calibration error is now negative --
the critic *under*-estimates, which is the benign direction.

The spread is the honest headline: 1/20, 17/20, 20/20. Pure SAC solves this
task on most seeds and fails on some, which is a characterised weak baseline,
not a solved problem. Report the distribution, never the best seed.

**Read survival together with the Waypoint rate, never alone.** In
`phase1_pretrain` a Waypoint success terminates the episode, so once the Waypoint rate
is high a *short* episode is a fast success. Seed 260817 survives 26.6 s at
200k because it reaches the Waypoint in about that time, close to Waypoint-PD's 24.1 s.
Survival alone only separates dying from not dying.

**Replay eviction is not the cause of the late decay.** Doubling
`buffer_size` to 600k, so nothing is evicted at all within a 500k run, changed
nothing: calibration error at 500k went 23.661 -> 23.536 and `critic_q1` 3.181
-> 2.801. The two runs are bitwise identical through 300k (same seed, and the
buffers only differ once more than 300k transitions exist), which is the
control this comparison needed. The factor is reverted.

The decay begins before eviction could matter -- calibration error is +0.94 at
200k and 14.159 at 300k, while the 100k-200k data would not leave a 300k
buffer until 400k. The behaviour collapses first and the critic follows: the
true soft return falls from -0.4 to -13.3 between 200k and 300k, and only
afterwards does `actor_action_std` slide monotonically (0.67 at 300k -> 0.37
at 500k) with saturation reaching 0.34-0.64. The remaining explanation is the
actor/critic feedback loop: with alpha pinned at 0.005 the entropy term is
negligible against a Q that has grown to order 1, so the actor turns greedy,
narrows the data distribution, and the critic extrapolates.

Current state: **200k is the training length; 500k is past the cliff.** The
three-seed replication above is the Pure SAC Phase-I result. The late decay is
a documented limitation of longer training, not something to keep chasing.

## Where this stands

Pure SAC on `single_phase` is **closed**: a real weak baseline whose honest
three-seed result on the aligned 262000 block is completion 3/1/0 and
no-violation 20/10/0 -- it mostly fails to complete, and on some seeds is not
even feasible (260862 violates on every episode). Do not keep tuning it; the
distribution is the finding. See *Pure SAC on `single_phase`: three seeds*.

**Pure MPC's original evidence was terminal-only and does not transfer.**
`constrained_mpc_nominal_config()` fixes the reference at the desired pose and
its evidence (`logs/phase2_mpc_v2_clarabel_5seeds.json`, 5/5, zero QP fallback)
was measured in `terminal_phase_environment_config()`: a 2-10 m shell, 100 s, the
*old* S1 task, no Waypoint. None of that transfers to the benchmark. What does
transfer is the cost, and the honest composition of it is the 1.7x below, not the
3.2x this row first reported. Pure MPC now runs on the benchmark itself -- see
the next paragraph -- so this is history, kept because the terminal-only numbers
are still the record for that shell.

The hybrid does not exist. Four subsequent probes found no defensible gap for
it, so implementation and training are paused; see the current-decision block
at the top of this file.

`eval.evaluate_policy.evaluate_model` needs nothing but an object with
`.predict(obs, deterministic) -> (action, state)`, which a test pins. That is
the shared measurement path for all three methods; do not fork it.

**Pure MPC now runs on the benchmark, one command, with the fair reference.**
`experiments/evaluate_mpc.py --task single_phase` puts the Pure MPC row on the
same task the SAC and scripted rows use (10-14 m, constrained from step one) and
**defaults to the corridor-guidance reference for that task** -- a fixed setpoint
is a myopic regulator on a 95 s approach, so the resolver picks the shared
guidance path unless `--reference-source` overrides it. `terminal` still defaults
to `fixed`, so the terminal-only evidence is reproducible. Every episode emits
the shared `main_table` block; `eval/main_table.py` assembles the three rows and
prints an alignment line so a reader can confirm they share task, seed block and
episode count before trusting the comparison.

The 20-episode baseline, seed block 262000, one seed block, aligned 20/20 with
the scripted row (measured on the user's workstation, not the sandbox):

| method | completed | time s | force N*s | worst margin | compute mean | compute p95 | budget mean/p95 |
|---|---|---|---|---|---|---|---|
| Scripted | 20/20 | 95.25 | 168.1 | +0.047 | 0.09 ms | -- | 0.00x |
| Pure MPC (corridor) | 20/20 | 89.31 | 160.0 | +0.044 | 99.2 ms | 206 ms | **0.99x / 2.06x** |

Both complete every episode, so completion rate does not separate them. On the
three quality columns **Pure MPC is slightly ahead** -- faster (89.3 vs 95.3 s),
less impulse (160 vs 168 N*s), higher return (4.35 vs 4.12) -- because the
optimiser flies the shared corridor reference closer to the edge. The one column
that separates them is compute, and the honest reading of it is not the mean:

- **The mean, 0.99x, is an artefact of the machine.** Compute is wall-clock, so
  the absolute ms is not comparable across machines (the sandbox measured
  176 ms / 1.76x for the same run). Never report Pure MPC as "within budget" off
  the mean.
- **The p95 is 2.06x and the max is 7.7x (773 ms).** That is the once-a-second
  exact-linearisation refresh step. Real-time control is bounded by the worst
  case, not the mean, so Pure MPC cannot guarantee a step inside the 100 ms
  period on this platform -- the refresh step always overruns.
- **The structural part is machine-independent:** the QP's 0.51x, and the
  refresh lump, which exists *because* the target tumbles fast enough that the
  linearisation goes stale within a horizon (`Lambda > 1`). That is the compute
  claim that survives review; state it with p95/max and the QP fraction, and
  note that a spacecraft processor is one to two orders slower than this
  workstation, so 89 ms on the ground is not 89 ms in flight.

Zero QP fallback, zero predicted-safe truth violations, all five margins
positive over the twenty episodes. The SAC row comes from a trained checkpoint
through the same `main_table` path.

**Settle the main-table metrics before running MPC, not after.** Completion
rate alone does not separate successful controllers because the scripted and
Pure MPC rows already reach 20/20. The quality columns are completion time,
force impulse, worst constraint margin, and per-step compute. **All four are
now on the shared path, in conventions
`eval/metrics.py` pins**, so a row is never assembled from a private
convention. That module is imported by the learned rows
(`eval/evaluate_policy.py`), the Pure MPC row (`experiments/evaluate_mpc.py`)
and the scripted reference row (`eval/validate_single_phase_semantics.py`); every
one of them now emits a `main_table` block, and `digest_run` prints it. Three
conventions it fixes, each because the alternative was producing a number that
could not be compared:

- **Compute is the controller, not the simulator.** `controller_time_s` covers
  only the call that produces an action, `environment_step_time_s` covers the
  RK45 truth propagation. The old `full_control_cycle_runtime_s` timed them
  together and so over-reported the controller by about 8.6x -- measured on
  CPU, `predict` is 1.09 ms against `env.step` at 8.32 ms -- while the MPC side
  has always timed `controller.command` alone. It is retained as the sum so
  historical evaluation files still parse; **it is not the compute column.**
  `per_step_compute_s.controller_mean_over_budget` is the headline: > 1 means
  the controller cannot run at the 0.1 s control period whatever the simulator
  costs.
- **Completion time and discounted return are recorded per episode**, as
  `steps`, `survival_s` and `discounted_return` -- the names the scripted
  validators already used, on one `gamma` shared from `train.configs.PURE_SAC`.
  For a completed episode `survival_s` *is* the completion time; for one that
  hit the 200 s cap it is not, which is why `completion_time_s` is `None`
  rather than zero when nothing completed. A zero there reads as "instant".
- **Effort and time are read over completed episodes.** Force impulse is a time
  integral, so an episode that dies at 10 s scores a smaller one; ranking rows
  by the raw mean rewards dying early. `completed_only` is the table figure,
  `all_episodes` sits beside it. The worst margin is the opposite: it spans
  *every* episode, because grazing a boundary on the episodes a method loses
  does not earn it a clean margin column.

Three tests in `tests/test_train_eval_config.py` pin this, including that the
two timers stay separate; re-merging them is the failure mode that cost this
round.

The later compute probes tested whether MPC call rate created such a gap.
Short-horizon local and sparse-refresh MPC kept completion and safety across
three sampled-phase blocks, so no learned call-rate layer is currently
justified.

### Compute was half artefact. The honest number is 1.7x, not 3.2x

The 0.322 s per step was measured against a constraint linearisation that
finite-differenced the margins ten times per horizon index, fifty indices per
control step. Profiled, that was **74% of `MPCController.command` against the
QP's 9%**, and two thirds of it was inside `env.task.compute_task_metrics`: a
margin reads three of its twenty fields yet paid for an `se3_log`, an `so3_log`
and the SVD in `project_to_so3` on every one of those five hundred calls.

Both halves are fixed in `controllers/mpc/constraints.py`. The margin functions
compute only what a margin is a function of -- values bitwise unchanged, pinned
against the task module by a test -- and the Jacobian is now analytic. Every
margin is a function of the relative position `p`, the target-frame velocity
`R v` and the chaser-frame line of sight `R^T (p_port - p)`, each of which
differentiates in closed form against the exponential coordinates; only
`d(J_l(phi) rho)/d(phi)` needed deriving. A test holds it to central
differences over random states (agreement 1.2e-9, which is the difference's own
floor).

Re-measured with `python -B -m experiments.profile_mpc_step`, horizon 50,
CLARABEL, exact linearisation refreshed every 10 control steps:

| | mean | p95 | budget |
|---|---|---|---|
| full command, all steps | **175.5 ms** | 405.4 ms | **1.75x** |
| full command, no refresh due | 149.7 ms | 161.9 ms | 1.50x |
| full command, refresh step | 407.4 ms | 424.6 ms | 4.07x |
| of which QP solve | 51.4 ms | 54.5 ms | 0.51x |
| of which constraint linearisation | 35.5 ms | 40.1 ms | 0.35x |

So: **compute is still over budget, so it is not purely an artefact -- but 3.2x
was, and the composition is now entirely different.** The constraint term went
from ~295 ms to 35.5 ms; what remains is the QP, the fifty-step nominal
rollout, and the refresh lump. Report 1.7x with this composition, not 3.2x, and
state the implementation (Python, CVXPY, CLARABEL, one core) beside it. The
p95 is a real property, not noise: it is the refresh step, once a second.

For the original horizon-50 exact configuration, the QP's own 0.51x is the
formulation cost that survives profiling. Later horizon-10 probes reduced the
ordinary-step QP cost and showed that exact-refresh tails, not an absent learned
layer, are the remaining issue. Local MPC puts p95 inside budget without losing
safety on the measured phase blocks; max/cold-start tails remain reportable
limitations.

## What each reference is for

`References/` is part of the method, but each paper has one job:

| paper | use it for | do not |
|---|---|---|
| `哈工大.pdf` | the SE(3) modelling this work builds on | claim modelling as a contribution against it |
| `北航.pdf` | constraint handling (approach cone, FOV, saturation); its 0.0173 rad/s target and absent speed cap are what put our regime outside it | |
| `南航.pdf` | its Limitation 1 -- an LTI prediction model assuming a *moderate* tumbling rate -- is this work's motivation | claim the layered architecture as novel: 南航 is **already** upper-layer-proposes / lower-layer-filters-for-safety, with deadlock detection on top. What is ours is that the upper layer is *learned* and that the regime is `Lambda > 1`. Concede the pattern, claim the two differences |
| `北航编队.pdf` | the RL-supplies-a-schedule-to-MPC interface pattern | its impulsive model, which is not comparable |
| `上海交大.pdf` | the three-way comparison table format (hybrid / pure MPC / pure RL, one row for per-step solve time) | **its method: adapting MPC cost weights online is explicitly ruled out** |
| `引入死区迟滞...pdf` | nothing -- it is the user's own prior paper | cite it |

## How the work is run

Long training happens on the user's machine, not here. This session does code
review, `pytest`, short smoke runs (<= 5k steps) and diagnostic probes. Training
is currently stopped. Repository changes are made on an explicit branch and
committed once per auditable stage. Run the suite as `python -B -m pytest -q`;
the 2026-09-02 repository state is **149 passed**, not the stale 146/147 recorded
in the superseded handoff. The checked-in `.venv` launcher points to a missing
Python 3.12.6; use the verified fallback in `docs/REPRODUCIBILITY.md` rather
than rebuilding or modifying code merely for that launcher symptom.

## Co-rotation: why this task is not translational rendezvous

The Waypoint and the desired pose are **body-fixed on a target tumbling at 0.0412
rad/s**, and `total_speed_m_s` is the relative speed in that rotating frame. An
inertially frozen chaser therefore appears to move at `omega * r`:

| r | apparent speed | margin to the 0.35 m/s limit | co-rotation force |
|---|---|---|---|
| 12 m | 0.494 m/s | **negative** | 2.16 N |
| 10 m | 0.412 m/s | **negative** | 1.80 N |
| 8 m (Waypoint) | 0.330 m/s | +0.020 | 1.44 N |
| 3 m (desired) | 0.124 m/s | +0.226 | 0.54 N |

Write `Lambda = omega * r / v_max`. **`Lambda > 1` means station-keeping is
inadmissible and the chaser must co-rotate continuously.** The task runs from
`Lambda = 1.41` at 12 m to 0.35 at 3 m, crossing 1 near the Waypoint. A zero-thrust
chaser breaches the speed limit after 19.2 s on average (14.6-23.6 over 12
seeds), always on `total_speed`, against a ~95 s mission.

So the total-speed limit is not a manoeuvring cap, it is a **sustained-thrust
condition whose required thrust scales with range**. This is the physical
motivation for the benchmark, and it is the regime the reference papers do not
occupy: `References/北航.pdf` has the same corridor and FOV geometry but a
target at 0.0173 rad/s and *no* speed constraint (`Lambda` undefined; 0.74 if
ours were applied at their 15 m anchor), and `References/南航.pdf` names
"tumbling rate is moderate relative to the prediction horizon" as the standing
assumption of its LTI prediction model, and lists successive re-linearisation
as future work.

**Do not claim we already implement that.** An earlier version of this file
did, and it is false as configured. Under `constrained_mpc_nominal_config()`
(`linearization_source="exact"`, `exact_linearization_refresh_steps=10`),
`controller.command` assigns the one cached exact Jacobian to *every* index of
the 50-step horizon and refreshes it only every 10 control steps. That is a
single LTI model held constant across a 5 s window and rebuilt once a second --
slower than the per-sampling-period relinearisation 南航 actually performs, and
squarely inside the failure condition its Limitation 1 describes (the target
turns 0.206 rad = 11.8 deg within one horizon). The `"local"` path does
re-linearise along the horizon every 5 indices, but against
`LocalRelativePredictionModel`, not the RK45 truth. **Re-linearising along the
horizon against the truth is not implemented on either path.**

**That cost is now measured, and it settles the question.** One exact
linearisation point -- 36 central-difference pairs of RK45 truth propagation --
costs **258 ms**, 2.6x the control period, which is why the refresh step's p95
is 407 ms against 150 ms on ordinary steps (`experiments/profile_mpc_step.py`
prints both). Re-linearising against the truth at every one of the 50 horizon
indices is 50 of those: **about 13 s per control step, 130x the budget.** It is
not implementable at this tumbling rate and horizon, so do not build it.

State the weaker, true claim instead: an exact one-step Jacobian, refreshed
every 10 control steps and held across the horizon. And note what the 13 s
means, because it is the sharpest single number this work has for why the
regime is hard -- the successive re-linearisation 南航 lists as future work is
not merely unimplemented, it is two orders of magnitude outside a real-time
budget once the target tumbles at 0.0412 rad/s. That is an argument for the
hybrid, not an embarrassment: the learned layer is what buys the right to call
the optimiser less often.

## Trusted entry points

| Purpose | Entry |
|---|---|
| Environment | `env.phase2_env.phase2_environment_config("single_phase" \| "phase1_pretrain" \| "full_mission")` |
| Task | `env.phase2_env.phase2_s1v2_mission_config()` |
| SAC config | `train.configs.PURE_SAC` (single instance, pinned by tests) |
| Train | `python -B -m train.train` |
| Evaluate | `python -B -m eval.evaluate_policy` |
| Phase-I reachability | `python -B -m eval.validate_phase1_semantics` |
| single-phase reachability | `python -B -m eval.validate_single_phase_semantics` |
| Full-mission reachability | `python -B -m eval.validate_phase2_semantics` |
| Why episodes ended | `python -B -m eval.inspect_failures --run logs/<run>` |
| Critic calibration | `python -B -m eval.diagnose_value_calibration` |
| One-screen run digest | `python -B -m eval.digest_run --run logs/<run>` |
| Main-table conventions | `eval.metrics.main_table_metrics` -- the one definition of the four differentiating columns; every row imports it |
| MPC per-step cost | `python -B -m experiments.profile_mpc_step` |
| Pure MPC row (fair) | `python -B -m experiments.evaluate_mpc --task single_phase --episodes 20 --seed 262000 --output OUT.json` -- single_phase defaults to the corridor reference |
| Three-way main table | `python -B -m eval.main_table --row "LABEL=eval.json" ...` -- reads each path's `main_table` block |

## Rules

- **Strictly one interpretable factor per experiment**, recorded in the
  manifest and in its own commit. Never mix a task-parameter change with a
  hyperparameter change.
- **S1-v2 task parameters are frozen.** Changing them invalidates the
  comparability of eleven prior single-factor rounds. Change one only on
  measured evidence of a definitional defect, never on judgement alone.
- **Do not judge short experiments by Waypoint rate** -- it lags and its variance
  is large. Use critic calibration error and survival time.
- **Every training run starts from zero.** Fresh actor, critic and replay, no
  initialising from a checkpoint, no resuming a mid-run checkpoint, and no
  staged hand-off from `phase1_pretrain` into `full_mission`. This is a
  standing decision, not a temporary diagnostic measure: it is what keeps the
  data behind every reported number clean and each run independently
  reproducible from its manifest alone.
- The waypoint that blocked Phase-II entry is cleared -- Phase-I now works and the
  full mission is reachable -- so `full_mission` runs are in scope. What is
  still out of scope is initialising them from a Phase-I model.
- **`n=1` training seeds are not conclusive.** Independent re-evaluation on a
  fresh seed block flipped 8 of 9 comparable published numbers. Any conclusion
  reported upward needs >= 3 training seeds.
- Report results through `eval/digest_run.py`, not by shipping log files.
- `References/` holds the papers this work is measured against and the failure
  modes it must avoid. Read them before designing a new comparison; they are
  part of the method, not clutter.
- `models/` is deliberately untracked. Checkpoints are reproducible from a
  manifest plus its seed, so they stay on the machine that trained them.

## Reference numbers (measured, seed block 262000 unless noted)

| Policy | Waypoint | Survival | Discounted return |
|---|---|---|---|
| Zero action | 0/20 | 53.3 s | -8.71 |
| Uniform random | 0/20 | 53.5 s | -- |
| Station-keeping (2.50 N) | 0/20 | 200 s, never dies | -1.51 |
| Waypoint-PD cruise 0.15 | 20/20 | 24.1 s | +3.66 |
| Scripted full mission | 20/20 waypoint, 20/20 completion | 95.2 s | +5.21 |

Single-phase `single_phase` task (seed block 262000, its own reward scale -- these
numbers are **not** comparable with the two-phase rows above, which carry the
+5 Waypoint event):

| Policy | Completion | Survival | Discounted return |
|---|---|---|---|
| Zero action | 0/20, terminal violation | 16.9 s | -9.08 |
| Hover (co-rotate, never close) | 0/6, never dies | 200 s | +3.03 |
| Scripted single law | **20/20** | 95.3 s | **+4.13** |

Scripted worst margins over 20 seeds: total speed +0.140, closing speed +0.047,
FOV +0.325, corridor lateral +1.19, axial +1.71; peak total speed 0.210 against
the 0.35 limit. Thrust mean 1.76 N, p95 6.29 N, peak 8.66 N (= the full
three-axis authority), saturated on 1.2% of steps. **The binding resource early
in the task is actuation, not constraint margin** -- if the task ever has to be
degraded, move the initial range or the terminal tolerances, never
`total_speed_limit_m_s`, which is what puts the task in the `Lambda > 1` regime
that makes it worth posing.

**Attribute a calibration error before acting on it.** `calibration_error` is
`Q - soft`, and the soft return carries `alpha * entropy` accumulated *per
step*, so it grows with survival: a policy living 110 s collects about +5 of
entropy bonus, one dying at 16 s collects a few tenths. `digest_run` and the
diagnostic CLI print `hard`, `soft` and `entropy_contribution` side by side for
exactly this reason. A critic that tracks the hard return but lags the soft one
is failing to anticipate the bonus for surviving longer, which is what chasing
an improving policy from below looks like; that is a different finding from a
critic that misjudges task quality.

Reachable discounted return spans only about 16, so a calibration error of
several units means the critic carries no information about policy quality.
That was true of every `auto_0.005` model (4.9 to 45.7); it is no longer true
at fixed alpha, where the error is under one unit through 200k.

Target body rate 0.0412 rad/s puts the body-fixed Waypoint on a 0.33 m/s circle,
so co-rotation needs a *sustained* 1.44 N at the Waypoint and 2.16 N at 12 m, plus
about 1.31 N of Coriolis. Phase-I is therefore attitude-orbit coupled tracking,
not translational rendezvous. This motivates the benchmark but, after the four
negative probes, does not by itself justify a hybrid.

## The single_phase reward, and the one thing it does not fix

`single_phase` uses **one shaping form for the whole mission**: the bounded,
discount-consistent potential that Phase-I always used, with its velocity term
measured against `active_desired_velocity`, which now returns a corridor-aware
guidance law whenever the terminal constraints are live instead of zero. The
unbounded potential and its per-step state penalty are gone. Observation schema
`phase2_mission_v4_phase_guidance_error_24d` records the change: the velocity
channel is a tracking error against the *active* law in every phase, where v3
degraded it into a raw speed once the constraints went live.

This is a guidance reference, not a control law -- the chaser still has to find
the thrust, including the sustained co-rotation. It is the same relationship
Phase-I always had with `phase1_desired_velocity`.

**What it does not fix is the horizon.** Potential shaping is policy-invariant
by construction, so the choice between completing and hovering rests entirely on
the discounted completion bonus, and at `gamma = 0.997` the effective horizon is
333 steps against a 953-step mission -- 2.9 horizons, `gamma^953 = 0.057`, so the
+20 is worth **+1.14** at reset. Hovering therefore captures 73% of the scripted
return risk-free (+3.03 against +4.13).

Do not patch this pre-emptively. Pure SAC's measured failure is that it cannot
hold co-rotation for more than ~35 s, so it cannot yet reach the hover policy at
all; the +1.11 gap is the last increment, not the first obstacle. Run the
baseline first. If a seed plateaus on `time_failure` with no violations and near
200 s survival, it has reached hover and the horizon is then the binding factor
-- raising `gamma` or `final_success_reward` becomes the next single factor.

## Pure SAC on `single_phase`: three seeds on the aligned block

The three `..._fix_seed26086{0,1,2}` checkpoints (400k, after the guidance law
got its axial restoring term), re-evaluated deterministically on the **same
262000 block, 20 episodes** the scripted and Pure MPC rows use, through
`evaluate_policy --assume-canonical` (their pre-rename manifests no longer load,
but the 24d observation, action and network are unchanged):

| seed | completion | no-violation | note |
|---|---|---|---|
| 260860 | **3/20** | 20/20 | best; clean but rarely completes |
| 260861 | 1/20 | 10/20 | violates on half its episodes |
| 260862 | 0/20 | **0/20** | violates on *every* episode |

Two things this settles:

- **The block flips the number.** 260860 was recorded at 7/20 on its old
  evaluation block; on the aligned 262000 block it is 3/20. This is exactly the
  "independent re-evaluation flipped 8 of 9 published numbers" rule -- the
  aligned three-seed distribution is the honest figure, single-seed/single-block
  is not.
- **Pure SAC does not even guarantee feasibility.** 260860 never violates a
  constraint, 260862 violates on all twenty episodes. Same algorithm, same
  training, only the seed changed, and the behaviour goes from a clean co-rotating
  hover to hitting a boundary every episode. Constraint satisfaction is a seed
  lottery, not a property. Read the three-way table's Pure SAC row with this in
  mind: whichever seed fills it, its worst-margin cell describes that seed alone,
  and 260862's is negative.

So the honest Pure SAC headline on `single_phase` is completion 3/1/0 over three
seeds and no-violation 20/10/0 -- a real weak baseline that mostly fails to
complete and, on some seeds, is not even safe. It establishes the value of a
constrained MPC safety layer: Pure MPC holds every constraint on all 20 episodes
with zero predicted-safe truth violations, which SAC cannot promise. The probes
did not establish that a learned upper layer adds anything to that controller.

**This is a real weak baseline, not a straw man**, and Pure SAC is closed here.
Note what it does *not* beat: the scripted controller is 20/20 on the same
guidance law with a hand-tuned PD. Completion rate therefore does not by itself
distinguish successful classical controllers. **The differentiating metrics are
completion time, force impulse, constraint margin and per-step compute.**

## Pure SAC on `single_phase`, before the overshoot fix (three seeds, 260850-260852, 400k)

Completion is 0/20 on every seed at every checkpoint, but the reason moved
twice, and each move needed a different metric to see:

| checkpoint | closest approach (260850) | worst corridor_axial | what ended it |
|---|---|---|---|
| 100k | 9.38 m | +4.63 | FOV 16/20, total speed 4/20 |
| 250k | 3.44 m | +4.36 | **no violation 19/20**, ran to the 200 s cap |
| 350k | 0.61 m | +1.74 | FOV 11/20, corridor 4/20 |
| 400k | **0.07 m** | **+0.29** | corridor lateral 13/20 |

**Do not judge this task by `constraint_success`.** It rewards not trying: the
250k checkpoint scores 19/20 clean by hovering 3.44 m short. Closest approach
is the metric that tracks progress, and by it 400k is the best checkpoint, not
250k.

At 400k the chaser reaches 0.07 m from the desired pose -- inside the 0.25 m
completion tolerance -- and then keeps going. `corridor_axial` is the distance
ahead of the port and the corridor radius is that distance times
`tan(35 deg)`, so at the 0.29 m it reaches the cone is only 0.21 m wide and
squeezes it out. It overshoots the desired pose by 1.2 m, straight at the port.
Every other margin is grazed but barely negative (FOV -0.024, total speed
-0.002, closing -0.001): a policy flying on every boundary at once.

The critic is not the problem. Q rises monotonically to 2.0-2.3 on all three
seeds, alpha stays pinned, no abort fires, and at 250k the calibration errors
are -2.89 / +2.78 / +4.83 with entropy contributions of 3.96 / 1.94 / 2.96 --
the good seed under-estimates a long-surviving policy, which is the benign
direction and mostly the entropy bonus it has not yet learned to expect.

## Pure SAC on `full_mission` (three seeds, 260830-260832)

Training length is **150k**; the run collapses by 200k. Judged as the rules
require, on calibration error rather than Waypoint rate:

Measured under observation schema v3, before the guidance-law rebuild, so these
numbers are a record of the two-phase baseline rather than a live comparison.

| seed | Waypoint @150k | calibration @150k | Waypoint @200k | calibration @200k |
|---|---|---|---|---|
| 260830 | **20/20** | **0.002** | 1/20 | 6.636 |
| 260831 | 18/20 | 2.165 | 1/20 | 9.524 |
| 260832 | 0/20 | -- | 0/20 | -- |

0.002 is the best calibration ever measured here. The collapse is behavioural,
not critical: Q barely moves (+0.458 -> +0.580) while the true soft return falls
(+0.456 -> -6.056). Same signature as the Phase-I late decay, arriving earlier.
alpha stayed pinned at 0.005 throughout and no abort threshold fired.

**Completion is 0/20 on every seed at every checkpoint.** Pure SAC clears the
Waypoint and then loses co-rotation: corridor survival is ~35 s against the 71 s the
terminal leg needs, and of 61 Waypoint arrivals, 41 died on `total_speed`, 15 on
FOV, 4 on the corridor and 1 on closing speed. **None died on the transition
step**, so the Waypoint acceptance set is admissible and its tolerances need no
change. The scripted controller holds the same leg with 0.66 N sustained.

## Traps

1. `env.task.Phase2MissionConfig` defaults to the **old** S1 task. S1-v2 only
   exists via `phase2_s1v2_mission_config()`. Always pass the config explicitly.
2. `eval/evaluate_policy.py::main` compares a manifest against the current
   canonical config and raises on any difference, so pre-v3 schema models
   cannot be loaded through the CLI. Call the library functions with the
   manifest's own config instead (`diagnose_value_calibration.py` shows how).
3. `train/train.py` writes `translational_observation_frame`,
   `translational_velocity_observation` and `mission_task_version` as string
   literals. Fix those before ever changing the observation schema. The v3 -> v4
   bump left all three still accurate, but `tests/test_train_eval_config.py`
   pins the schema name and is the tripwire that catches a silent change.
4. `logs/phase2_mission_s1_semanticfix_validation/` predates the current reward
   implementation and cannot support any current claim. Its replacement is
   `logs/phase2_mission_s1v2_validation/`.
5. `terminal_constraint_failure` is evaluated on the Waypoint transition step
   itself, so an inadmissible arrival ends the episode immediately. Phase-I's
   0.30 m/s Waypoint speed tolerance does not by itself guarantee admissibility
   against the 0.28 m/s closing-speed limit that applies there.
6. The repository has no `conftest.py` and is not installed as a package, so
   run the suite as `python -B -m pytest -q` from the repository root. A bare
   `pytest` can resolve to a different interpreter and fails collection on all
   thirteen files at once.
7. A `single_phase` manifest still carries fourteen Waypoint and Phase-I fields
   (`waypoint_position_target_m`, `waypoint_reward`, `premature_entry_distance_m`,
   `phase1_cruise_speed_m_s`, ...) because `Phase2MissionConfig` is shared with
   the two-phase modes. **None of them do anything in `single_phase`** -- every one
   is keyed off mission phase 0, which that mode never enters. Do not read task
   semantics off them. For the same reason observation dimension 24, the mission
   phase flag, is constant 1 there.
8. Reward weights and the guidance constants are constructor defaults, not
   config fields, so `asdict(config)` misses them. `train.py` records them under
   the manifest's `reward_settings` -- read the numbers there, not from the
   environment block.
9. The nominal benchmark intentionally keeps one deterministic target-tumble
   realisation, and that remains a paper limitation. A separate, single-factor
   `single_phase_phase_sampled` probe has now sampled initial attitude and
   angular-velocity direction at fixed magnitude. Across three 20-episode seed
   blocks, scripted, exact-r10, local and refresh100 MPC all completed 20/20
   with positive truth margins. Thus phase sampling has been tested and did not
   expose a hybrid gap. Do not fold it into frozen S1-v2 or claim that the
   nominal benchmark itself now generalises across tumble; the probe is a
   secondary task. See `docs/PROBE_RESULTS.md`.
10. `constrained_mpc_nominal_config()` fixes `reference_state` at the desired
    pose with a 50-step (5 s) horizon -- a myopic regulator with no path plan on
    a ~95 s task from 10-14 m, while the scripted 20/20 comes from a
    corridor-aware guidance law. **`corridor_tracking_mpc_config()` is the fair
    Pure MPC configuration**: it tracks the same `env.task.corridor_guidance_velocity`
    the reward, the observation and the scripted controller use, rolled forward
    from the current position into a per-stage reference trajectory, so the
    comparison is on how each method flies a shared reference rather than on
    whether it has one. The fixed-setpoint config is kept as the record for the
    terminal-only evidence measured before the trajectory existed; the default
    stays `"fixed"` so that evidence is bitwise reproducible, and
    `experiments/evaluate_mpc.py --reference-source corridor_guidance` selects
    the fair row. The guidance law now lives on the task, not in the reward, for
    exactly this reason -- a fourth private copy is how the reward and the
    scripted validator drifted apart once (see *The single_phase reward*).
