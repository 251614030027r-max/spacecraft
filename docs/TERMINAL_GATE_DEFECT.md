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

### 3c. Closed loop, three seeds so far

Fixed setpoint, h20, `--terminal-gate proximity` (§4) against the repository
default. The two seeds that already worked reproduce, which is the self-check
that the factor touched nothing else:

| seed | default gate | proximity gate |
|---|---|---|
| 262000 | completes 113.6 s, 202.0 N s, 0 infeasible | completes **113.6 s, 202.0 N s**, 0 infeasible |
| 262002 | completes 115.8 s, 132.2 N s, 0 infeasible | completes **115.9 s, 132.2 N s**, 0 infeasible |
| 262001 | **fails 38.7 s**, 379/387 infeasible from 0.8 s | **completes 121.9 s**, 298.8 N s, **0/1219 infeasible** |

`docs/PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §1 records 262001 completing in
**121.8 s** on the pre-F1 tree. The proximity gate returns that completion while
keeping F1's corridor preview inside the entry sphere.

**The full 12-seed sweeps (h20 proximity, h35 default, h35 proximity) are in
flight and are not reported here.** Until they land nothing in §5 is settled,
and no main-table row may be restated.

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

## 5. What is at stake, and what is not yet established

The defect in §2 and §3a is established: it is a static property of the
constraint assembly, it needs no trajectory, and it is independent of any
sweep outcome. The optimiser imposes the approach corridor on states the task
does not judge, and the resulting linearised violation is about twice the
slack cap, so the QP reports `infeasible` rather than paying a penalty.

What is **not** yet established is how much of the project's standing evidence
that accounts for. The claims that would have to be re-derived if the sweeps
confirm §3c are, in order of how much rests on them:

1. `PRECAPTURE_ENTRY_WINDOW_GAP_WITH_F1.md` §3g -- *"the fixed setpoint is a
   reference for which the constrained problem has no satisfying trajectory
   [...] the only remaining thing that can change is the reference itself"*.
   This is the motivation the coupling line is built on.
2. The horizon-invariant failure set `{262005, 262006}` and the four-seed h20
   failure set `{262001, 262005, 262006, 262011}`.
3. `T6_COUPLING_INTERFACE.md`'s disconnected feasible set, which was scanned on
   those seeds.
4. The T8 reading that a learned upper layer *preserves lower-layer
   feasibility* -- its strongest single piece of evidence is 262005, which the
   fixed setpoint loses to 172 infeasible steps from t = 31.1 s and the learned
   V2/262200 completes with 0.
5. The h20/h50 compute trade, *"the horizon that wins every quality column is
   the horizon that cannot be run at the control period"*. If the gate is what
   was costing h20 its completions, that gap narrows.

None of those is retracted here. They are listed so that the sweep is read as
the test of them that it is.

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
