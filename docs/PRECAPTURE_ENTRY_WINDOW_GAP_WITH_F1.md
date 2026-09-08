# N-A: the 12-seed sweep re-run with F1, and what it does to the gap

Measured 2026-09-08 in the upper-level sandbox on the merged tree
(`fabdfe4`, which contains F1's `_predicted_terminal_active`). Twelve seeds,
`precapture_planning`, fixed-setpoint reference, `analytic_local`, truth state,
`perception=None`, one episode per cell, horizon 20. This is the re-run
`docs/PRECAPTURE_ENTRY_WINDOW_GAP.md` section 4 asked for.

Wall-clock is not reported: the twelve runs were executed four at a time, so
nothing here may be read as a real-time measurement.

## 1. The completion count did not move. Everything under it did

| | pre-F1 (`1fe5549`) | with F1 (`fabdfe4`) |
|---|---|---|
| h20 completion | 8/12 | **8/12** |
| failure mechanism | illegal entry-plane crossing, never re-latched; chaser then parks on the desired pose to the 300 s cap | **QP infeasibility cascade; the controller falls back to zero wrench and the chaser drifts until a constraint breaks** |
| failing seeds | not recorded per seed in the pre-F1 document, and its raw sweep files are not in this tree, so the two sets are **not** compared here | 262001, 262005, 262006, 262011 |

Taking the count alone would have concluded "F1 changed nothing". The count is
the least informative column in the table.

**Three seeds do have matched pre-F1 traces in this tree**
(`logs/precapture_planning_v2/upper_diagnosis/upper_control_2620NN_fixed_ps20.json`,
recorded on `1fe5549` at a 1 s trace stride, so their solver statuses are a
1 Hz sample rather than every step):

| seed | pre-F1 | with F1 |
|---|---|---|
| 262002 | completed 115.8 s, 117/117 sampled steps `optimal` | completed **115.8 s**, 0/1159 infeasible |
| 262004 | completed 126.5 s, 128/128 sampled steps `optimal` | completed **126.5 s**, 0/1266 infeasible |
| 262001 | **completed 121.8 s**, 123/123 sampled steps `optimal` | **fails at 38.6 s**, 379/387 infeasible |

So on this evidence F1 is a no-op where the problem was already feasible -- two
completion times reproduce to the recorded 0.1 s -- and on 262001 it converts a
121.8 s completion into a controller that is infeasible from 0.8 s onward.
**F1 is not free.** It buys the entry-plane preview by adding constraint rows,
and on some seeds those rows make the problem unsolvable from the start.

## 2. The failure is now the solver, and the separation is clean

| seed | completed | end t (s) | end r (m) | QP steps `infeasible` | first infeasible | ended by |
|---|---|---:|---:|---|---:|---|
| 262000 | yes | 113.5 | 3.23 | **0 / 1136** | -- | completion |
| 262002 | yes | 115.8 | 3.22 | **0 / 1159** | -- | completion |
| 262003 | yes | 154.5 | 3.17 | **0 / 1546** | -- | completion |
| 262004 | yes | 126.5 | 3.22 | **0 / 1266** | -- | completion |
| 262007 | yes | 134.0 | 3.22 | **0 / 1341** | -- | completion |
| 262008 | yes | 125.5 | 3.23 | **0 / 1256** | -- | completion |
| 262009 | yes | 142.1 | 3.23 | **0 / 1422** | -- | completion |
| 262010 | yes | 115.1 | 3.22 | **0 / 1152** | -- | completion |
| 262001 | no | 38.6 | 14.68 | **379 / 387** | 0.8 s, 16.75 m | FOV violation |
| 262006 | no | 34.9 | 17.22 | **283 / 350** | 6.7 s, 17.38 m | FOV violation |
| 262005 | no | 96.0 | 10.61 | **172 / 961** | 31.1 s, 11.98 m | FOV violation |
| 262011 | no | 103.2 | 30.00 | **760 / 1033** | 27.3 s, 14.37 m | 30 m distance failure |

**At h20, every episode that completes solves the QP on every step, and every
episode that fails goes infeasible and never fully recovers.** There is no
overlap in this column. The statuses are CVXPY's own: `optimal` or
`infeasible`, nothing else appears.

**That clean separation does not survive h50, and the counter-example is the
useful part** -- see section 3b. Seed 262001 at h50 is infeasible for 927 of
its 1579 steps, in one unbroken run from t = 0, and **completes anyway** at
157.8 s. So the infeasible *count* is not the discriminator, and any sentence
built on "infeasible therefore doomed" is wrong.

## 3. Why an infeasible QP is fatal here specifically

`MPCController.command` catches the solver failure and commands **zero wrench**
(`controllers/mpc/controller.py`, the `except` that sets `used_fallback`). In an
ordinary regulation problem that is a defensible fallback: do nothing for one
step and try again.

This task is not that problem. It runs at `Lambda > 1` -- at 15 m the body-fixed
goal is on a 0.62 m/s circle -- so a zero-thrust chaser is not holding station,
it is being left behind at `omega * r` while the sightline rate grows. The
traces show exactly that: force and torque both identically 0.0 for hundreds of
consecutive steps, the FOV angle climbing monotonically to the 50 deg limit
(49.92, 49.92, 49.83 deg on the three FOV failures), or the range walking out to
the 30 m boundary on 262011. **The fallback that makes an infeasible step
survivable in a station-keeping regime is the fallback that kills the episode in
a co-rotating one.**

262001 is the sharpest case: infeasible from step 8 -- 0.8 s, before the chaser
has moved -- and infeasible for 379 of its 387 steps. It never had a controller.

## 3b. Retraction, and the sharper mechanism it forced

Section 2's separation was written from the h20 column alone and stated too
broadly. **Seed 262001 at h50 completes in 157.8 s with 927 of 1579 steps
infeasible, in one unbroken run starting at t = 0.** Ninety-three seconds of
continuous zero wrench, and the episode still finishes. The infeasible count is
therefore not the discriminator, and neither is the longest run.

The same seed at the two horizons is the cleanest instrument in this sweep,
because the environment, the seed and the initial state are identical and only
the horizon differs:

| | first 8 steps | afterwards | outcome |
|---|---|---|---|
| **h20** | `optimal`, **force 8.660 N and torque 1.0388 N*m -- both the full three-axis envelope** -- driving the relative angular rate from 0.04497 down to 0.01907 rad/s | infeasible; zero wrench freezes that half-finished attitude rate | FOV 5.1 deg -> **49.9 deg**, violation at 38.6 s |
| **h50** | infeasible from step 0; the controller never acts | zero wrench; the relative rate stays at 0.04497 rad/s, i.e. the chaser stays inertially fixed | FOV 6.8 deg at 30 s, 24.8 deg at 90 s, feasible again by 92.7 s, **completes** |

**The mechanism sentence that stood here has been falsified by the experiment it
motivated.** It read: what kills the episode is not coasting, it is being
abandoned part-way through a manoeuvre. That reading explained the contrast in
the table above, and it was wrong -- see section 3e, where completing the
manoeuvre is measured and the episode dies sooner. The h20/h50 contrast is real
and the table stands; the causal claim drawn from it does not.

It did make the fallback a well-posed single factor, which is what it was worth.
Both of the replacements it suggested -- keep flying the shifted previous
solution, and relax the constraint softening -- have now been run, and both are
negative. Sections 3e and 3f.

One diagnostic fact to carry forward separately: at h20 the very first control
steps sit at **both** full envelopes at once, `8.660 N = 5*sqrt(3)` and
`1.0388 N*m = 0.6*sqrt(3)`. That is transient, not the sustained saturation the
task forbids, but it happens at step 0 and belongs in the per-episode
saturation diagnostics.

## 3c. The h50 column, and the trade it exposes

The same twelve seeds at horizon 50, identical in every other setting.

| horizon | completed | mean completion time | failing seeds |
|---|---:|---:|---|
| h20 (2 s) | 8/12 | 128.4 s | 262001, 262005, 262006, 262011 |
| **h50 (5 s)** | **10/12** | **77.3 s** | 262005, 262006 |
| h50 pre-F1 (`1fe5549`, for contrast) | 1/12 | -- | -- |

**The pre-F1 horizon ordering is inverted, exactly as that document said it
would be.** `docs/PRECAPTURE_ENTRY_WINDOW_GAP.md` section 3 refused to read
h50 < h20 as a horizon result and predicted that F1 would restore the expected
ordering. It does: 1/12 becomes 10/12.

On the eight seeds **both** horizons complete, so the comparison is paired:

| seed | h20 time | h20 impulse | h50 time | h50 impulse |
|---|---:|---:|---:|---:|
| 262000 | 113.5 s | 202.0 N s | 57.9 s | 182.7 N s |
| 262002 | 115.8 s | 132.2 N s | 62.0 s | 182.8 N s |
| 262003 | 154.5 s | 318.9 N s | 82.1 s | 216.8 N s |
| 262004 | 126.5 s | 299.3 N s | 63.8 s | 272.8 N s |
| 262007 | 134.0 s | 198.1 N s | 70.3 s | 165.6 N s |
| 262008 | 125.5 s | 222.4 N s | 60.9 s | 180.1 N s |
| 262009 | 142.1 s | 110.2 N s | 69.7 s | 157.6 N s |
| 262010 | 115.1 s | 182.3 N s | 59.9 s | 163.9 N s |
| **mean** | **128.4 s** | **208.2 N s** | **65.8 s** | **190.3 N s** |

**h50 dominates h20 on all three columns at once** -- two more completions,
half the time, and 8.6% less force impulse on the paired seeds. It is not
buying speed with fuel. Fuel is the final criterion for this work, so this is
the ranking that counts.

Saturation, over completed episodes, against the full three-axis envelopes
(8.660 N, 1.0392 N*m), counting a step as saturated at 95% of the envelope:

| horizon | median force-saturated steps | median torque-saturated | longest continuous force saturation | median mean force |
|---|---:|---:|---:|---:|
| h20 | 1.0% | 0.5% | 2.2 s | 1.68 N |
| h50 | 3.1% | 0.9% | 5.8 s (262004) | 2.94 N |

Neither is sustained saturation -- the worst case is 5.8 s inside a 63.8 s
episode -- so both satisfy the standing requirement that thrust and torque must
not sit on the envelope. h50 works the actuators harder per second and still
finishes with less total impulse, because it finishes so much sooner.

### The two h50 failures are two different failures

- **262006** is the abandonment mechanism of section 3b: 253 of 285 steps
  infeasible in one run, FOV violation at 28.4 s.
- **262005 is not.** It solves 247 of its 281 steps, with a longest infeasible
  run of **4 steps**. It fails by flying in too hard: force pinned at 8.660 N
  from 16.7 m down to 9.5 m in 28 s, torque at 1.0392 N*m, and the FOV angle
  goes 8.0 deg at 20 s to 49.5 deg at 28 s. This is the close-range actuator
  saturation already recorded for 262004 pre-F1 -- the sightline rate grows as
  1/r and the torque envelope runs out. **Tightening the cone cannot buy back
  authority the thrusters do not have**, and a longer horizon does not either.

So h50's remaining gap is one fallback failure and one actuation limit, not one
phenomenon.

## 3d. The better controller is the one that cannot be run

Measured with `experiments/profile_precapture_mpc.py`, **serial, single process,
nothing else on the machine**, 300 control steps, seed 262000, timing
`MPCController.command` alone -- the RK45 truth propagation (6.95 ms per step)
is not in the controller budget.

| horizon | mean | p95 | max | steps over the 100 ms period |
|---|---:|---:|---:|---:|
| h20 | 84.3 ms (**0.84x**) | 93.9 ms (**0.94x**) | 348.9 ms | **7 / 300** |
| h50 | 202.4 ms (**2.02x**) | 222.0 ms (**2.22x**) | 944.4 ms | **300 / 300** |

Both maxima are step 0 -- the first solve pays CVXPY's problem construction --
so read them as a cold start, not a steady-state property. Excluding it, h20's
worst step is 120.1 ms (1.20x) and seven of three hundred steps overrun; h50
overruns on **every step measured**, and its steady-state worst is 248.2 ms.

Absolute milliseconds are a property of this machine and do not transfer; the
ratio to the control period is the reportable figure, and it must be stated with
the implementation beside it (Python, CVXPY, CLARABEL, one core) and with the
standing caveat that a spacecraft processor is one to two orders slower than
this one.

**This is the gap, stated in one line.** On the same twelve seeds, the same
task, the same reference and the same truth:

| | completion | mean time (paired) | mean impulse (paired) | real time |
|---|---:|---:|---:|---|
| h20 | 8/12 | 128.4 s | 208.2 N s | **fits**, 0.94x at p95 |
| h50 | **10/12** | **65.8 s** | **190.3 N s** | **does not**, 2.22x at p95, over on every step |

**The horizon that wins every quality column is the horizon that cannot be run
at the control period, and the horizon that fits real time is the worse
controller in all three.** That is the trade a learned upper layer exists to
break: the target for the coupled row is h50-like outcomes at h20 cost, and both
ends of that target are now measured on this task rather than assumed from the
`single_phase` compute rounds.

It is also the concrete form of `References/南航.pdf`'s Limitation 1 in this
regime. Its LTI prediction model assumes the tumble is moderate relative to the
horizon; here the horizon that is long enough to plan the approach is the one
whose per-step cost has already left the budget.

## 3e. D2: keep flying the last feasible plan -- negative

Single factor: `infeasible_fallback`, `zero` (the default, and the historical
behaviour) against `shift`, which commands the previous solution advanced one
step. Twelve seeds, h20, fixed setpoint, everything else identical.

**Completion is 8/12 either way**, and the eight episodes that never see an
infeasible step reproduce bitwise -- completion time to the recorded 0.1 s,
final range to 1e-12. That is the self-check that the factor touched nothing
else, and it passed.

| seed | `zero` | `shift` |
|---|---|---|
| 262001 | 38.6 s, FOV | **27.3 s, FOV** |
| 262005 | 96.0 s, FOV | **241.2 s, FOV** |
| 262006 | 34.9 s, FOV | 38.4 s, FOV |
| 262011 | 103.2 s, 30 m | **81.4 s, FOV** |

None is rescued, and they move in both directions -- one lives two and a half
times longer, two die sooner. Continuing the plan changes the trajectory
substantially and changes no outcome.

**The trace says the fallback worked and the hypothesis was still wrong.** On
262001 the shifted plan carries the relative angular rate 0.01907 -> 0.00011
rad/s by 4.0 s: the de-spin manoeuvre the zero fallback abandoned is completed,
the chaser ends up properly co-rotating -- and it then dies 11 s sooner than
when it was abandoned. Finishing the attitude plan does not help because
attitude was not the binding problem. At `Lambda > 1` a chaser with no force is
handed to `omega * r` whatever its attitude is doing; the plan only changes how
long it takes to leave the cone.

An uncomfortable corollary worth stating: **doing nothing was accidentally
protective.** Zero wrench survived 38.6 s on 262001; following the optimiser's
own last advice survived 27.3 s.

## 3f. D3: the infeasibility is the slack cap, and removing it makes things worse

Reading the constraint assembly answers why "infeasible" appears at all. Every
precapture constraint is **soft**: it enters as
`J x + s >= b` with `s >= 0` priced at `constraint_slack_weight = 1e4`. But the
slack is itself bounded, `s <= constraint_slack_limit`, default **2.0**. That
cap turns the soft constraint back into a hard one -- a linearised margin worse
than -2.0 in normalised units is infeasible no matter how heavily it is priced.

Single factor, `constraint_slack_limit` 2.0 against 20.0, fallback left at its
default `zero`:

| seed | default 2.0 | cap 20.0 | slack actually used |
|---|---|---|---:|
| 262000 (control, completes) | 113.5 s, 0/1136 infeasible | 113.5 s, 0/1136 | **0.000** |
| 262001 | 38.6 s, **379/387** infeasible | **7.8 s**, **0/79** | 4.602 |
| 262005 | 96.0 s, **172/961** infeasible | **48.4 s**, **0/485** | 2.201 |
| 262006 | 34.9 s, **283/350** infeasible | **14.2 s**, 5/143 | 19.012 |

**Both halves of this are decisive and they point the same way.**

The cap *is* the mechanism: raise it and the infeasibility disappears
outright on two seeds and nearly on the third, and the slack the optimiser
then spends -- 4.6, 2.2, 19.0 -- is exactly the amount the 2.0 cap was
refusing. 262005 needed only 2.201, barely over the cap, and that alone
accounted for all 172 of its infeasible steps.

And it does not matter: **all three failing seeds fail sooner.** Removing the
refusal does not produce a feasible trajectory, it produces a solution that
knowingly violates -- and the truth then really does violate, faster, because
the optimiser is now driving toward the violation instead of declining to move.

The control is the other half of the argument: 262000 spends **exactly zero**
slack over 1136 steps, so the constraint set is comfortably satisfiable on the
seeds that work. (Its numbers are not bitwise identical -- final range and FOV
differ in the fourth significant figure -- because changing a variable's upper
bound changes CLARABEL's numerical path, and 1136 closed-loop steps amplify
that. The outcome and the completion time to 0.1 s are unchanged.)

## 3g. What D2 and D3 together establish

Two implementation-level rescues have now been measured and both are negative:
the solver's fallback (4/4 failures unrescued) and the softening cap (3/3
failures made worse). Neither is a tuning question left open.

So the honest statement about the four h20 failures is stronger than "the MPC
gives up badly":

> On these seeds the fixed setpoint is a reference for which the constrained
> problem has no satisfying trajectory inside the linearised horizon. The
> refusal to solve, the choice of fallback, and the amount of softening are all
> downstream of that. **The only remaining thing that can change is the
> reference itself** -- and choosing the reference is precisely what an upper
> layer does.

Section 3 of `docs/PRECAPTURE_ENTRY_WINDOW_GAP.md` reached the same conclusion
from the other side: the lower layer has no representation of backing out and
coming round again. D2 and D3 close the escape routes that would have let that
be an implementation defect instead of a structural one.

## 4. What this does and does not do to the paper's gap

**It does not shrink the gap; it renames it.** Pre-F1 the four failures were
decision failures at the entry plane. With F1 they are feasibility failures of
the constrained optimiser at 12-17 m, well before the entry plane -- three of
the four go infeasible outside 11 m, and 262011's illegal crossing happens after
it has already lost the controller, not before.

That is a **stronger** motivation for a learned upper layer, not a weaker one,
and it is the motivation the reference papers already frame: `南航`'s pattern is
upper-layer-proposes / lower-layer-filters-for-safety, and the standing question
for any such architecture is what the upper layer should propose so that the
lower layer's problem stays solvable. Here the fixed setpoint is a reference for
which the constrained problem is infeasible from 12-17 m on four of twelve
seeds. Choosing a reference the MPC can actually solve is precisely a decision,
it is precisely what the `radial_local` action space parametrises, and unlike
the entry-time story it does not depend on a latch rule.

**Three things it forbids saying.**

1. **The pre-F1 "all 15 failures are one failure" sentence is now history, not a
   current claim.** It described `1fe5549`. On the merged tree the failure
   signature is different and the document that carries it
   (`docs/PRECAPTURE_ENTRY_WINDOW_GAP.md`) must be read as a record of that
   commit.
2. **h20 vs h50 still cannot be read as a horizon result** until the h50 column
   of this sweep is in, and even then the infeasibility rate is a property of
   the constraint set at a given horizon, not of foresight.
3. **The zero-wrench fallback is a candidate defect, and it has not been
   changed.** Whether a better fallback (hold the previous shifted solution,
   or re-solve with the terminal rows relaxed) rescues these four seeds is an
   open, cheap, single-factor question. Until it is measured, "the MPC cannot
   solve it" and "the MPC gives up badly when it cannot solve it" are two
   different claims and only the second is established.

## 5. Reproduce

```
python -B -m experiments.diagnose_precapture_reference \
  --seed 2620NN --horizon 20 --guidance fixed \
  --output logs/upper_na_f1/na_f1_h20_2620NN.json
```

Twelve JSONs are in `logs/upper_na_f1/`; each carries the per-step `status`,
wrench norms and FOV angle the table above is read from.
