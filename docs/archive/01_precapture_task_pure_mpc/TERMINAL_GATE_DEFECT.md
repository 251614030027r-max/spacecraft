# The MPC enforces the approach corridor 16 m from the target

Measured 2026-09-12 in the upper-level sandbox, fixed setpoint only, no learned
policy and no training. Reproduce with
`python -B -m experiments.diagnose_terminal_gate`.

## 1. What was being investigated

`docs/PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §4 left one question open:

> The zero-wrench fallback is a candidate defect, and it has not been changed.
> Whether a better fallback (hold the previous shifted solution, or **re-solve
> with the terminal rows relaxed**) rescues these four seeds is an open, cheap,
> single-factor question.

Its §3e and §3f closed the first half. `infeasible_fallback="shift"` rescues
0 of 4 failures; raising `constraint_slack_limit` from 2.0 to 20.0 makes all 3
tested failures die sooner. From those two negatives §3g concluded:

> On these seeds the fixed setpoint is a reference for which the constrained
> problem has no satisfying trajectory inside the linearised horizon. [...]
> **The only remaining thing that can change is the reference itself** -- and
> choosing the reference is precisely what an upper layer does.

That sentence is the motivation the whole coupling line rests on. It is sound
only if the constraint set is the right one. **The terminal rows were never
audited, and they are not the right ones.**

## 2. The defect

`controllers/mpc/constraints._predicted_terminal_active` decides when the eleven
terminal rows -- corridor axial, eight corridor lateral facets, terminal total
speed, closing speed -- enter the QP:

```python
return bool(terminal_latched or port_axial_distance < task.entry_port_axial_distance_m)
```

With the task's frozen numbers (`port = (-1.5, 0, 0)`, `axis = (-1, 0, 0)`,
`entry_port_axial_distance_m = 4.5`) the second clause is `p_x > -6`. It is a
**half-space, with no bound on range**: every state on that side of the plane
has the approach corridor imposed on it, at 6 m or at 26 m.

The truth side does not agree. `normalized_precapture_truth_margins` gates the
same rows on `terminal_latched` alone, and `SE3RendezvousEnv` scores a corridor
violation only after `_terminal_region_entered` latches on a *legal entry-disc
crossing*. `_predicted_terminal_active` is used at
`controllers/mpc/constraints.py:308` and `:507` and nowhere else -- it is
purely the optimiser's own view.

**So the optimiser refuses to solve because of a region the task never judges.**

## 3. Measured, three ways

### 3a. Static, no trajectory involved

Walking one coordinate across the gate plane at a fixed 16.4 m range, target
tumble 0.041231 rad/s, nothing latched (`scratch: gate_unit3`):

| `p_x` | range | `port_axial` | MPC gate | worst MPC corridor row | slack needed (cap **2.0**) | truth corridor row |
|---:|---:|---:|---|---:|---:|---:|
| -6.84 | 16.76 m | 5.340 | off | inactive | 0.000 | inactive |
| -6.10 | 16.47 m | 4.600 | off | inactive | 0.000 | inactive |
| **-5.99** | **16.43 m** | **4.490** | **ON** | **-4.130** | **4.130** | inactive |
| -5.90 | 16.39 m | 4.400 | ON | -4.149 | 4.149 | inactive |
| -5.50 | 16.25 m | 4.000 | ON | -4.236 | 4.236 | inactive |

A step of 11 cm turns a satisfied constraint set into one that needs twice the
available slack, 16.4 m from the target, while the truth margins stay inactive
on both sides. The chaser is ~15 m off the approach axis and is being asked to
be inside a cone whose radius there is about 2.9 m.

### 3b. In the QP, on the seed the project calls a feasibility failure

Seed 262001, h20, fixed setpoint. The baseline reproduces the recorded row of
`docs/PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §2 exactly -- 387 steps,
**379 infeasible**, first at **0.8 s**, final FOV margin -0.0007 rad.

Re-running with the slack cap lifted so the same QP always solves, and reading
the slack it then spends:

| t | status (cap 2.0) | slack demanded |
|---:|---|---:|
| 0.0 - 0.7 s | optimal | 0.0001 |
| **0.8 s** | **infeasible** | **4.033** |
| 1.0 s | infeasible | 4.044 |
| 1.2 s | infeasible | 4.223 |

The demanding row is **row 8 -- a corridor lateral facet**. At t = 0.8 s the
chaser is at target-frame `(-6.84, -10.91, 10.72)`, 16.75 m out, with
`port_axial = 5.34`, so the gate is *off at the current state*: it is the h20
rollout, drifting at 0.45 m/s of axial closure, that reaches across the plane
inside the 2 s horizon. That is why infeasibility begins at 0.8 s and not at 0.

`infeasible_fallback="zero"` then commands zero wrench. At `Lambda > 1` that is
not station keeping, and the chaser drifts to a 49.96 deg FOV violation at
38.7 s having never had a controller.

### 3c. Closed loop, 12 seeds, two horizons

Fixed setpoint, seeds 262000-262011, `--terminal-gate proximity` (§4) against
the repository default. The h35 default column reproduces
`docs/PURE_MPC_ROW_VERIFIED.md` exactly -- 9/12, 88.5 s, 167.2 N s, failing
`{262005, 262006, 262011}` -- which is the independent check that this harness
is the same measurement.

| | completed | mean t | mean force impulse | **QP infeasible steps** | illegal entries | failing seeds |
|---|---:|---:|---:|---:|---:|---|
| h20 default | 8/12 | 128.5 s | 208.2 N s | **1 594** | 9 | 262001, 262005, 262006, 262011 |
| h20 proximity | 8/12 | 124.4 s | 205.7 N s | **0** | 5 | 262003, 262005, 262006, 262011 |
| h35 default | 9/12 | 88.5 s | 167.2 N s | **2 161** | 4 | 262005, 262006, 262011 |
| h35 proximity | 8/12 | 78.6 s | 173.3 N s | **0** | 5 | 262001, 262005, 262006, 262011 |

**Paired on the episodes both arms complete, the force impulse is identical**
-- 192.4 N s against 192.4 N s at h20 (7 seeds), 173.3 against 173.3 at h35
(8 seeds). The factor changed nothing where the QP was already solving, which
is the self-check that it touched only what it was meant to.

Two readings, and the second matters more than the first.

**The defect is removed.** Not one of the 24 proximity-gated episodes ever
loses the controller: 1 594 and 2 161 infeasible steps become 0 and 0. On
262001 at h20 that converts a 38.7 s drift into a 121.9 s completion, and
`PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §1 records the pre-F1 tree completing
that same seed in 121.8 s.

**And it does not move the completion count**: 8/12 either way at h20, and
9/12 to 8/12 at h35. `{262005, 262006, 262011}` fail under all four
configurations. **So infeasibility was never what was costing the
completions.** The hard seeds are hard with a controller that never gives up.

## 3d. What the failures actually are, once the controller stops giving up

Every timeout under the corrected gate ends in the same state:

| | end t | force impulse | **final range** | FOV margin | illegal entries | **latched** |
|---|---:|---:|---:|---:|---:|---|
| h20 prox 262003 | 300.0 s | 287.3 N s | **3.00 m** | 0.872 rad | 1 | **no** |
| h20 prox 262005 | 300.0 s | 930.6 N s | **3.00 m** | 0.854 rad | 1 | **no** |
| h20 prox 262011 | 300.0 s | 501.1 N s | **3.00 m** | 0.871 rad | 1 | **no** |
| h35 prox 262001 | 300.0 s | 351.1 N s | **3.00 m** | 0.873 rad | 2 | **no** |
| h35 prox 262005 | 300.0 s | 636.9 N s | **3.00 m** | 0.872 rad | 1 | **no** |
| h35 prox 262006 | 300.0 s | 584.5 N s | **3.00 m** | 0.873 rad | 1 | **no** |
| h35 prox 262011 | 300.0 s | 378.1 N s | **3.00 m** | 0.873 rad | 1 | **no** |

3.00 m is `||desired_position||`. **The chaser flies to the goal, crosses the
entry plane illegally on the way, and then sits on the goal with healthy
pointing until the clock runs out**, because the latch only arms on an
outside-to-inside crossing (`SE3RendezvousEnv`, `evaluate_terminal_entry_crossing`,
guarded by `not self._terminal_region_entered`). Once inside, re-arming
requires flying back out past the entry plane and coming in again, and a fixed
setpoint at the desired pose has no reason to ever do that.

This is verbatim the pre-F1 failure mechanism that
`PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §1 recorded and that F1 was
introduced to remove: *"illegal entry-plane crossing, never re-latched; chaser
then parks on the desired pose to the 300 s cap."* F1 removed it by imposing
the corridor early enough to shape the approach -- and paid for that with a
hard infeasibility at 16 m. Both halves of that trade are now measured.

## 4. Single factor: bound the gate by range

Treatment, and nothing else changes:

```python
terminal_latched or (port_axial_distance < entry_port_axial_distance_m
                     and ||p|| <= ||port_position|| + entry_port_axial_distance_m)
```

`1.5 + 4.5 = 6.0 m` is where the entry plane sits relative to the target
centre. Both terms are frozen task parameters, so the treatment adds **no
tunable constant**. Reference, horizon, solver, slack cap and fallback are
untouched.

**It is a correct diagnosis and it is not proposed as the final gate.** It
removes the defect cleanly, but it gives the optimiser no corridor preview
outside 6 m, so the illegal crossings F1 was suppressing come back (h35
illegal entries 4 -> 5, and 262001 at h35 flips from a 167.5 s completion to a
300 s park). Choosing the region the corridor should be enforced over -- and
whether entry legality belongs as a region constraint at all rather than as a
constraint on the crossing -- is a lower-layer design decision that is now
well posed and is not settled here.

## 5. What this does to the standing claims

**Retracted.**

1. `PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §3g -- *"the fixed setpoint is a
   reference for which the constrained problem has no satisfying trajectory
   inside the linearised horizon"* -- is **false as stated**. With the gate
   bounded by range the QP solves on every one of 24 episodes' steps and the
   same seeds still fail. The infeasibility was the gate, not the reference.
   §3e and §3f closed the two escape routes they tested; the constraint set
   itself was never audited, and §4 of that document had listed exactly that
   audit as still open.
2. Any sentence explaining a failure *by* infeasibility -- including the T8
   reading that the learned upper layer's contribution is that it **preserves
   lower-layer feasibility**. Its headline evidence was 262005: the fixed
   setpoint takes 172 infeasible steps from t = 31.1 s, V2/262200 takes 0.
   Under the corrected gate the fixed setpoint takes **0** infeasible steps on
   262005 and still fails.

**Survives, and is stronger for being restated.**

262005 under the corrected h20 gate: the fixed setpoint runs the full 300 s,
spends 930.6 N s, parks at 3.00 m and never latches, with **1 illegal
crossing**. V2/262200 completes it at 282.8 s with **0 illegal crossings**.
The learned layer's advantage is real and it is not about feasibility -- it is
that it **timed its entry so the one crossing was legal**. The fixed setpoint
structurally cannot do that: aimed at the desired pose it flies in when the
geometry says go, and once inside it cannot re-arm the latch without backing
out, which `PRECAPTURE_ENTRY_WINDOW_GAP.md` §3 already observed the lower
layer has no representation of.

That is what `a_commit` parametrises in the T6 interface -- hold outside,
commit when the crossing will be legal -- so the interface was built for the
right decision variable even while the motivating diagnosis was wrong.

**Unresolved, and expensive.**

Every training run in T7 and T8 was made against the defective lower layer.
Those runs stay valid measurements of *that* system; they are not measurements
of the intended one, and the policies were free to shape themselves around a
controller that abandons the chaser on a geometric technicality. Nothing from
them can be reported as a coupled row until the gate is settled and the runs
are redone from zero.

## 6. Reproduce

```
python -B -m experiments.diagnose_terminal_gate --horizon 20 \
    --seeds 262000 262001 262002 262003 262004 262005 \
            262006 262007 262008 262009 262010 262011 \
    --terminal-gate proximity --output logs/terminal_gate/gate_h20.json
```

Drop `--terminal-gate proximity` for the repository default. The static table
of §3a is `_predicted_terminal_active` and
`normalized_precapture_constraint_margins` evaluated directly, with no
environment involved.
