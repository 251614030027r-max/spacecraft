# Working notes for this repository

High-fidelity 6-DoF SE(3) **pre-capture** control for a fast-tumbling
non-cooperative target. Chaser 106 kg, target 225 kg, 500 km / 45 deg circular
orbit, RK45 truth with central gravity, second moments, gravity gradient and
J2, 0.1 s control period, +-5 N / +-0.6 N*m per axis.

**The active task is `precapture_planning`, and the active work is the
SAC-MPC coupling.** Everything about `single_phase`, the Waypoint, the A1/A2/A3
perception line and the 24D mission schemas is history; it is kept under
*Historical research line* because the lessons transfer, not because it is live.

---

## Current state -- 2026-09-16

Two load-bearing assumptions have now been **measured false**, and the research
question has moved accordingly. This is the live direction; the 2026-09-12 block
below is prior context.

- **T12 (learned scheduling vs fixed MPC, full truth).** `arrival_condition` is
  learnable (`radial_local` is not: 0/48 x3), but three arrival seeds average
  **26/48** against Pure MPC's **32/48**, slower and more fuel. Pre-registered
  branch 2: the interface makes coupling learnable, but a learned layer that
  unconditionally rewrites the reference does not beat the fixed setpoint. See
  `docs/T12_S10_*`.
- **Non-cooperative probe 2 (Pure MPC on EKF estimate vs truth, same 48 seeds).**
  **31/48 vs 32/48** (delta 1 -> pre-registered "basically level, stop"). The
  chaser goes blind ~40-60% of the tumble period (observability windows are real,
  measured over a full rotation), but the EKF coasts through them well enough
  that estimation does **not** degrade the fixed MPC. Non-cooperative observation
  is a real operational condition, not a source of a completion gap. See
  `docs/PROBE2_NONCOOP_RESULT_20260916.md`.

The direction is no longer "find a factor that makes MPC fail" or "prove RL
beats MPC". It is the **coupling mechanism itself**, motivated by the one positive
T12 finding: the learned policy *rescues* states Pure MPC fails (262401: +6) but
*destroys* states it succeeds (-4). The question is whether a **Bidirectional
Baseline-Anchored Task-Level SAC-MPC** turns that measured complementarity into a
stable net gain without breaking the strong baseline:

- **Baseline-anchored task residual.** SAC outputs a low-dim residual on the
  nominal arrival action (fixed setpoint); **residual 0 recovers Pure MPC
  bitwise**. It learns *whether/how much to deviate*, not the whole task.
- **Critic-advantage deployment gate.** The SAC critic (trained on full-episode
  return, so it carries the long horizon the short MPC cannot) judges whether
  deviating is worth it: `A_task = Q(s,a_cand) - Q(s,a_nom)`. This resolves the
  circularity of asking a short-horizon MPC to arbitrate long-horizon value.
  Caveat to watch: `Q(s,a_nom)` is off the policy's own action distribution, so
  the advantage can be noisy (this repo's critic-calibration history); the
  residual anchor + MPC feasibility gate keep it safe when the advantage is wrong.
- **MPC proposal certificate (bottom-up).** For the proposed task intent the MPC
  returns a compact (2-4 dim) predicted feasibility/safety certificate, fed into
  the next decision's observation -- a two-way negotiation, not raw telemetry
  (T7 showed telemetry-in-observation does not help).
- **Fallback.** Deviate only when the critic advantage clears a margin AND the
  MPC certificate says locally feasible; else nominal.

**Honest ceiling:** the per-episode baseline-preserving upper bound from T12 is
~**38/48 (+6)**. This is a mechanism paper with a modest, real gain, not a
blowout; the arbitration pattern itself is known (南航/AC4MPC) -- concede it,
claim the combination + task-level opportunity + `Lambda>1` non-cooperative
regime. Gate: does the full method, 3 seeds, beat Pure MPC on completion while
retaining most of its successes at acceptable fuel? See
`docs/BIDIRECTIONAL_SACMPC_EXECUTION.md`.

## Research execution discipline (adopted 2026-09-16)

The bottleneck was never rigor; it was **dev-phase over-gating**. Separate three
things and only the first two apply now:

> development validation  !=  paper formal validation  !=  flight certification

- **No probe proliferation.** Once a method version's design logic holds, run
  `implement -> smoke -> formal train (>=3 seeds) -> formal evaluate`, not
  `probe -> probe -> oracle -> grid -> seed archaeology -> maybe implement`.
  Performance questions are answered in formal training/eval, not in a stack of
  pre-training probes.
- **Smoke is a correctness check, not a direction gate.** Smoke verifies: runs,
  shapes/interfaces, no NaN, solver sane, fallback fires, residual 0 recovers
  nominal. Its success *rate* or a few-seed score never decides the direction.
- **Training noise is expected.** One seed failing does not kill a method; one
  seed passing does not prove it. Conclusions need >=3 training seeds judged as a
  distribution.
- **Do not switch the research question on a single probe/seed/metric.** After a
  version fails, first check implementation, convergence, and whether the
  coupling did what it should -- only consistent multi-seed formal negatives
  reopen the direction.
- **Method-first, then luxury.** Before a positive main result, do not sink time
  into large real-time profiling, hardware, perception stress grids, many
  ablations, or robustness sweeps.
- **Every new module must name its main claim** ("delete it -- does the paper
  still stand?"). If not, it is not added.

**Unchanged non-negotiables (scientific integrity, never relaxed):** truth
RK45 geometry is the only arbiter of safety/violation (never observation or QP
"solved"); Pure MPC stays frozen (no Q/R, horizon, corridor, terminal, or
tolerance edits except a named code bug or a genuine input-semantics change) and
is re-run once clean at the end for the final number; **no manufactured gap** (no
artificial obstacles, unmotivated noise, shortened horizon, throttled thrust,
deliberately wrong model, or rule change that only hurts nominal); reward never
encodes the answer (wait/enter timing, per-seed success times); no cherry-picking
seeds or checkpoints; comparisons use a pre-fixed evaluation seed block and the
final model; **any number entering a decision or the paper must point at an
artifact** (JSON/CSV/manifest/commit/script) or be marked exploratory.

## Prior state -- 2026-09-12

### The task

| | |
|---|---|
| Target tumble | 0.041231 rad/s (2.362 deg/s), period 152.4 s |
| Start | 15-20 m, 300 s cap, 30 m distance failure |
| Entry plane | 6 m from target centre (`entry_port_axial_distance_m = 4.5`), disc radius 3.151 m |
| Legal entry | radial < 3.151 m **and** target-frame speed <= 0.35 m/s **and** closing speed <= 0.20 m/s |
| Completion | position < 0.25 m, attitude < 10 deg, speed < 0.05 m/s, omega < 0.02 rad/s, held 1 s, terminal region latched |

An illegal crossing is **counted only**: it does not latch, does not end the
episode and carries no reward penalty. Re-entry stays available.

`Lambda = omega * r / v_max > 1` over most of the range, so station-keeping is
inadmissible and the chaser must co-rotate continuously. That is the physical
reason this is not translational rendezvous, and the regime the reference
papers do not occupy.

### The main table

| row | status |
|---|---|
| Pure SAC | closed, weak baseline (historical, on the old task) |
| **Pure MPC** | **done -- h35: 9/12, 88.5 s, 167.2 N s, p95 0.77x** (`docs/PURE_MPC_ROW_VERIFIED.md`) |
| **SAC-MPC coupled** | trained, **evaluation in flight** |

### The coupling interface (T6, frozen)

`env/hybrid_env.py`, `waypoint_parametrization="arrival_condition"`. The upper
layer emits a **2D** action every 2 s (20 control steps); the interface to the
MPC is the unchanged 3D waypoint through `reference_source="external_local"`.

- `a[0]` blends from an **inertially frozen hold** to the **desired pose**.
  Direction by slerp, radius by lerp -- not a chord between the two points.
- `a[1]` scales the hold radius about the radius it was frozen at.
- Optional 3D **execution feedback** appended to the observation: fallback
  fraction, peak slack read only from steps that solved, mean actuator usage.

**`a = (+1, *)` is the fixed-setpoint Pure MPC controller, bitwise.** Twelve
seeds, 300 control steps each, `max |delta wrench| = 0`. A test pins the
mapping; the equivalence is the capability floor and it is measured, not
asserted. See `docs/T6_COUPLING_INTERFACE.md`.

**Waiting must be inertial.** The approach axis is body-fixed, so holding a
fixed *target-frame* point co-rotates with it and the entry geometry never
changes -- there is no window to wait for, only fuel to spend. This has been
got wrong once and is now pinned by a test.

### What T7 established

Training four runs from zero, 60,000 decisions each:

| | last 100 episodes | first success | timeout fraction, Q1 -> Q4 |
|---|---|---|---|
| V2 / 262200 (no feedback) | **75%** | 5,942 | 24% -> **4%** |
| V3 / 262300 (feedback) | 55% | 13,062 | 28% -> **7%** |
| V2 / 262201 | **0** | -- | 31% -> **69%** |
| V3 / 262301 | **0** | -- | 39% -> **67%** |

1. **The interface is what made the coupling learnable, as a single factor.**
   The manifests differ in the parametrisation and nothing else -- same 31D
   observation, horizon, decision period, discount, learning rate, batch size,
   entropy coefficient, tau, learning-starts and network. At the common budget
   of 25,697 decisions the 4D `radial_local` batch completed **0 of 1308**
   episodes over three seeds; the 2D one reached **55 of 285** on its best
   seed. The earlier observation factor (target phase + remaining time)
   changed nothing on the old parametrisation, so information was not the
   binding constraint and expression was.
   See `docs/INTERFACE_SINGLE_FACTOR.md`.
2. **Two seeds learned to run the clock out instead of finishing.** Their
   return improves monotonically while their timeout fraction climbs, their
   illegal-entry rate matches the learning seeds, and their reference radius
   parks at ~7 m instead of closing to ~4.7 m. This is the hover pathology
   this repository already documented, in a new action space. The split is
   settled inside the first fifth of training, which points at exploration.
3. **The feedback ablation does not support the reverse channel.** V2 leads on
   first success, on every quartile and on final return. One learning seed per
   arm is below the three-seed rule, so this is *unsupported with the point
   estimate against it*, not a measured cost.

See `docs/T7_RESULTS_AND_VERDICT.md`. Next round is `docs/T8_EXECUTION_ORDER.md`.

### 262006: solvable offline, and closed as a case-level limitation

The offline feasibility certificate completes it in 217.5 s with **zero truth
violations** and 0.693 rad of field-of-view margin, so it is not at a
reachability boundary. But four families of upper-level position decision
(commit time, hold radius, approach rate, lateral offset) fail across ~90
scanned cells, all dying on field of view.

That suggested a pointing failure a position channel could not address, and
**the probe falsified it**: all three attitude reference modes fail, and
`aimed` fails *sooner* (34.9 -> 10.7 s at h20, 29.7 -> 8.5 s at h35). So the
attitude reference is **struck from the candidate factor queue**, and 262006 is
written as a case-level limitation with no proposed fix: *under the four
families of position decision and the three attitude modes tested it was not
rescued, while a zero-violation admissible path exists.*

Two things that probe did establish. On 262005 `aimed` does remove field of
view as the failure mode (96.0 s to the 299.9 s cap at h20), but trades it for
a timeout and for being pushed out to 29.96 m against the 30 m limit --
pointing authority is borrowed from translation, which is expensive where
co-rotation already needs the full three-axis authority. And `frozen` and
`swept` are numerically identical under a fixed setpoint in all four matched
pairs, because that reference sightline does not sweep; the repository's "best
of three" reads as best of two distinct behaviours.
See `docs/ATTITUDE_REFERENCE_PROBE.md`.

---

## Rules

- **Strictly one interpretable factor per experiment**, in its own commit with
  a manifest.
- **Task parameters are frozen**: geometry, constraints, time limit, reward.
- **>= 3 training seeds** before any conclusion is reported upward. n=2 with
  one dead seed is a noise floor of about +-37 points; nothing is measurable
  on it.
- **Every training run starts from zero.** No resuming, no checkpoint warm
  start, no staged hand-off.
- **Compute is serial single process only.** Parallel timings are never a
  real-time claim. Training may be parallel; evaluation may not.
- **Violations are judged on RK45 truth and real geometry**, never on the
  MPC's predicted margins.
- **Do not tune against the outcome**: no swapping seeds, no picking
  checkpoints, no dropping failed seeds. Report the distribution.
- **Any number that enters a decision or the paper must point at an artifact
  in the repository.** If it cannot, mark it unreproduced and do not cite it.
  This rule exists because a set of figures circulated for days before anyone
  noticed the scripts that produced them had never existed.

---

## Traps

1. **The raw per-key margins in `info` are not gated.** Corridor and terminal
   margins exist at every step whether or not the chaser is in the terminal
   region, so their running minimum is hugely negative on episodes that never
   violated anything, and the keys mix metres, radians and m/s. The comparable
   column is `minimum_truth_normalized_margin`, from
   `normalized_precapture_truth_margins`, which gates inactive rows positive
   and divides each active row by its own limit. Both the horizon rows and
   `experiments/evaluate_hybrid_policy.py` emit it.
2. **`predicted_minimum_margin` is not free and is not a truth margin.** It
   needs a full horizon rollout and only exists under
   `runtime_diagnostics=True`, which training disables. It is the optimiser's
   own forecast, not a geometry judgement.
3. **`maximum_slack` can be stale on a fallback step.** The solver variable may
   still hold the previous successful solve. Read it only from steps that
   solved, or pair it with the fallback flag.
4. **The wrapper is transparent only when `horizon_steps ==
   external_reference_hold_steps`.** Both are 20 in the coupling. Changing the
   horizon alone silently changes how the reference sequence is built.
5. **`target_entropy` is inert while `ent_coef` is a fixed number.** It is
   -3.0, sized for an older higher-dimensional action; the action is now 2D. It
   matters only if automatic temperature is ever re-enabled, and the history of
   that is in *Historical research line*.
6. **Loading a checkpoint against the wrong observation used to be silent.**
   The V3 runs are 34D (phase/time + feedback), V2 is 31D. The evaluation and
   replay paths now check and refuse; match the run's manifest rather than
   guessing flags.
7. **The `max` column of any compute measurement is step zero.** At every
   horizon the worst single step is the first one; the runner-up is 69-129 ms.
   Report it as a cold start a flight system pays once, or the reader will read
   it as a recurring tail.
8. **What separates h35 from h50 is not p95.** 0.77x against 1.14x reads as a
   near miss; the honest discriminator is steps over the control period, 2/300
   against 269/300.
9. The repository has no `conftest.py` and is not installed as a package, so
   run the suite as `python -B -m pytest -q` from the repository root.

---

## Trusted entry points

| Purpose | Entry |
|---|---|
| Task config | `env.task.PrecaptureTaskConfig` |
| Environment | `env.phase2_env.precapture_planning_environment_config()` |
| Coupling env | `env.hybrid_env.PrecaptureHybridEnv` |
| Lower layer | `controllers.mpc.config.precapture_mpc_config()` |
| Train the coupling | `python -B -m train.train_hybrid --parametrization arrival_condition` |
| Evaluate a coupled policy | `python -B -m experiments.evaluate_hybrid_policy` |
| Per-decision diagnosis | `python -B -m experiments.replay_hybrid_policy` |
| Scripted upper layers | `python -B -m experiments.evaluate_hybrid_scripted` |
| Serial per-step cost | `python -B -m experiments.profile_precapture_mpc --no-diagnostics` |
| Offline feasibility certificate | `python -B -m experiments.evaluate_precapture_oracle` |
| Main-table conventions | `eval.metrics.main_table_metrics` |
| Assemble the table | `python -B -m eval.main_table --row "LABEL=eval.json" ...` |

## Where the evidence lives

| Document | What it holds |
|---|---|
| `docs/PURE_MPC_ROW_VERIFIED.md` | the four horizon rows and the compute table, recomputed from artifacts |
| `docs/T6_COUPLING_INTERFACE.md` | the frozen interface, the bitwise floor, the disconnected feasible set |
| `docs/T7_RESULTS_AND_VERDICT.md` | the four training runs and what they do and do not establish |
| `docs/T8_EXECUTION_ORDER.md` | the round in flight |
| `docs/P0_HORIZON_PERSISTENT_FAILURES.md` | the horizon sweep |
| `docs/D0_HYBRID_INFEASIBILITY_DIAGNOSIS_20260909.md` | lower-layer infeasibility |
| `docs/INTERFACE_SINGLE_FACTOR.md` | the headline claim verified as a single factor |
| `docs/ATTITUDE_REFERENCE_PROBE.md` | the pointing hypothesis, falsified |

**Void, do not cite**: the return audit (+14.11 / -24.0) and the action-space
audit (a0 = -0.76 to -0.81). Their scripts have never existed in this
repository and the figures have never been reproduced.

---

## What each reference is for

| paper | use it for | do not |
|---|---|---|
| `哈工大.pdf` | the SE(3) modelling this builds on | claim modelling as a contribution |
| `北航.pdf` | constraint handling (cone, FOV, saturation); its 0.0173 rad/s target and absent speed cap are what put our regime outside it | |
| `南航.pdf` | its Limitation 1 (an LTI prediction model assuming *moderate* tumbling) is the motivation | claim the layered architecture as novel -- 南航 is already upper-proposes / lower-filters with deadlock detection. Ours is that the upper layer is *learned* and the regime is `Lambda > 1` |
| `北航编队.pdf` | the RL-supplies-a-schedule-to-MPC interface pattern | its impulsive model |
| `上海交大.pdf` | the three-way comparison table format | **its method: adapting MPC cost weights online is ruled out** |
| `AC4MPC` | critic as terminal cost + parallel double solve for a "no worse than" bound | both halves are closed to us -- learned terminal cost measured harmful, and a double solve is 1.5x the budget. **The infeasibility is itself a publishable contrast** |
| `ecc26_rollout.pdf` | *learning supplies a good nominal, optimisation improves in its neighbourhood* | |
| `引入死区迟滞...pdf` | nothing -- the user's own prior paper | cite it |

---

## How the work is run

Long training happens on the user's machine (`D:\py\DRL2`), not in the
sandbox. This session does code review, `pytest`, short smoke runs and
diagnostic probes. Repository changes go on an explicit branch, one auditable
stage per commit. The sandbox has no numeric stack by default; build a venv
and install `numpy scipy cvxpy gymnasium stable-baselines3 pytest`.

---

## Historical research line

The original target was a three-way comparison on the `single_phase` task with
a Waypoint and a 24D mission schema. That line is closed. Three lessons from it
still bind:

**The entropy temperature drove critic overestimation.** `auto` never
converged -- alpha rose monotonically in every run measured, and calibration
error was monotone in alpha across two observation schemas. `ent_coef` has
been a fixed 0.005 ever since.

**Hovering is a real attractor.** On the old task a policy that co-rotated and
never closed captured 73% of the scripted return risk-free, because the
discounted completion bonus was worth little at reset. The pre-registered
reading was: *a seed that plateaus on `time_failure` with no violations has
reached hover, and the horizon or the completion bonus is then the binding
factor.* The T7 dead seeds are this, in the new action space.

**Independent re-evaluation flips published numbers.** A fresh seed block once
flipped 8 of 9 comparable figures. Single-seed, single-block results are not
conclusions.

Pure SAC on the old task, three seeds on the aligned block: completion 3/1/0
and no-violation 20/10/0 -- it mostly fails to complete and on some seeds is
not even feasible. That is a characterised weak baseline, and it is what
establishes the value of a constrained MPC layer underneath.
