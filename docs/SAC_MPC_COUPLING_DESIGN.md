# The SAC-MPC coupling: what it is, what it closes, and what it must not do

Built 2026-09-07 on `1fe5549`. Entry points:
`env/hybrid_env.py`, `train/hybrid_configs.py`, `train/train_hybrid.py`,
`experiments/evaluate_hybrid_scripted.py`.

## 1. The gap it closes

Twelve seeds, fixed-setpoint MPC, no upper layer: 8/12 completed at horizon 20
and 1/12 at horizon 50. **All fifteen failures were one failure** -- an illegal
crossing of the 6 m entry plane that never re-latched -- with **zero**
truth-geometry violations of any kind, after which the chaser flew to the
desired pose anyway and parked on it to a median 0.000 m. Seed 262001 shows
the recovery is available; the regulator simply has no term for "that crossing
was the last one that counted".

That is a *scheduling* decision sitting on top of a tracking problem, which is
the interface pattern `References/北航编队.pdf` uses, and the thing
`References/南航.pdf` lists as future work is one layer below it. See
`docs/PRECAPTURE_ENTRY_WINDOW_GAP.md`.

## 2. The interface

One 3D waypoint in the **target body frame**, held for 2 s (20 control steps).
This is the interface that was already declared, with the frame corrected --
under the older inertially oriented reading the channel could not carry even
the task's own desired pose (0/3 against 3/3 direct), because a body-fixed
goal has to be re-issued as a rotating inertial point, which the horizon map
turns into a circular reference tracked with a 1.33 m standing lag against a
0.25 m completion tolerance. In the target frame the channel is bitwise
transparent. See `docs/PRECAPTURE_REFERENCE_TRACKING_DIAGNOSIS.md`.

The waypoint is **absolute, not a displacement**. A displacement
parametrisation biases the policy toward moving, and "hold and let the target
turn" is exactly one of the two answers the task poses.

**Waiting means inertially still.** The approach axis is body-fixed, so a
chaser holding a fixed *target-frame* point co-rotates with it and the entry
geometry never changes -- there is no window to wait for, only fuel to spend.
Holding position is therefore a *sequence* of target-frame waypoints, one per
decision, tracing the frozen inertial point. The 2 s channel carries this
because the policy re-issues the waypoint every decision. This is not a
limitation of the interface, it is what the interface is for, and a scripted
policy that got it wrong is recorded in section 5 as the check that caught it.

## 3. Why the coarse step is the mechanism, not a convenience

A Pure SAC step is one 0.1 s control step, so a 300 s episode is about 950 of
them and at `gamma = 0.997` the completion bonus is worth `0.997**950 = 0.057`
at reset. That is the documented reason hovering captures most of the scripted
return without ever finishing.

A coupled step is one 2 s decision, so the same episode is about 150 steps. At
`gamma = 0.99` the bonus is worth `0.99**150 = 0.22` -- **four times** what
Pure SAC ever sees -- with an effective horizon of 100 decisions (200 s) that
covers the episode. The credit-assignment path from an entry decision to the
completion it enables is six times shorter. That is a structural consequence
of the temporal abstraction, and it is the reason the coupling can represent a
decision made 100 s before it pays.

## 4. Three things held deliberately fixed

- **One architecture for the whole mission.** The MPC is the only thing that
  touches the actuators, from 17 m to contact. No phase switch, no hand-off,
  so no result can be confounded with one.
- **The reward is the environment's own**, summed over the decision period.
  Nothing rewards entering legally, waiting, or approaching a window. If the
  policy learns to time the entry it is because completing pays and being
  locked out does not.
- **The observation is unchanged** -- the same 24D schema Pure SAC uses, which
  already carries relative attitude, relative rate and target angular
  velocity. Nothing about the window is precomputed and handed over. Time
  remaining is *not* in the schema, which is a real limitation and a candidate
  single factor later; it is not being smuggled in now.

## 5. The feasibility gate, and the error it caught

`experiments/evaluate_hybrid_scripted.py` runs hand-written waypoint policies
through the identical wrapper. Neither is a reportable baseline; their job is
to bound what the action space can express.

`desired_pose` -- one constant waypoint at the desired pose -- reproduces the
no-upper-layer result seed for seed, **8/12**, with completion times matching
the direct row to 0.1 s (one decision boundary). So the wrapper adds nothing
and loses nothing.

The first version of `hold_then_enter` evaluated the entry alignment in the
target body frame and commanded the chaser's current target-frame point while
waiting. Both are wrong for the same reason -- in that frame a co-rotating
chaser is stationary and the window never opens -- and it scored 3/12 with the
commit criterion never firing on eight seeds. It is recorded here because the
failure is instructive: **"wait" is the action most easily got wrong, and it is
half the decision the paper claims.** The corrected rule holds the frozen
inertial point and tests the alignment inertially.

## 5b. Is a solution even in there? Sweeping the one decision variable

The gate that matters is not whether a hand rule is good, it is whether the
action space **contains** a solution on the seeds the lower layer loses. So on
exactly those four seeds, the entry time was swept directly -- hold the frozen
inertial point until `t`, then command the desired pose. One scalar, delivered
through the same 3D channel.

| seed | commit at 0 | 20 | 40 | 60 | 80 | 100 | 120 | 140 s | result |
|---|---|---|---|---|---|---|---|---|---|
| 262003 | -- | **yes** | yes | yes | yes | yes | yes | yes | completes at 175.3 s |
| 262005 | -- | -- | -- | -- | -- | **yes** | yes | yes | completes at 257.1 s |
| 262006 | -- | -- | -- | -- | -- | -- | -- | -- | no commit time works |
| 262011 | -- | -- | -- | **yes** | -- | -- | -- | -- | completes at 224.0 s |

**Three of the four are solved by choosing the entry time alone.** The ceiling
through this interface is therefore **11/12** against the lower layer's 8/12,
and that difference is the headroom the learned layer has to earn.

Seed 262011 is the one to keep in front of the reader: **only** the 60 s commit
works, and 40 s and 80 s both fail. The window is narrow, it is not where a
greedy controller would look, and a fixed rule steps straight over it -- which
is why the hand rule in section 5 scores 4/12 while a per-seed choice of the
same variable scores 11/12. The decision is real, it is sharp, and it has to be
learned rather than written down.

Seed 262006 is not solved by entry time alone on this grid. It is recorded as
a limit, not smoothed over: the sweep is coarse (20 s), the hold radius is
whatever the episode started at, and both are free parameters a learned policy
has that this sweep does not.

## 6. Running it

```
python -B -m train.train_hybrid --steps 60000 --seed 262100 \
    --run-name sac_mpc_hybrid_262100
```

Cost: one decision step is 20 MPC solves plus 20 RK45 truth steps, measured at
**0.90 s** in this sandbox, so 60k decisions is about 15 hours serial per seed.
Training may be vectorised -- that touches no red line, because **real-time
compute is only ever reported from serial single-process evaluation**, never
from training. Evaluation stays serial.

Every run starts from zero: fresh actor, critic and replay, no checkpoint
input, reproducible from the manifest plus its seed.

## 6b. Reading the result

``experiments/evaluate_hybrid_policy.py`` produces the row, through the shared
``eval.metrics.main_table_metrics`` path, so the coupled row is assembled by
the same code as the Pure MPC and scripted rows. It also runs the two controls
without a model: ``--control desired_pose`` is the fixed-setpoint lower layer
delivered through the wrapper -- the row the policy has to beat -- and
``--control random`` is the floor.

Compute is charged the way a flight computer would pay it: the MPC every
0.1 s control step, the policy once per 2 s decision, charged to the first
control step of that decision. Measured serially in this sandbox on the
``desired_pose`` control, the h20 coupled controller's p95 is **0.47x** the
control period. Absolute milliseconds are not comparable across machines and a
spacecraft processor is one to two orders slower, so state the ratio with the
platform beside it -- but at horizon 20 this configuration is inside budget,
which horizon 50 was not.

**Run the evaluation serially, single process.** A parallel run's timings are
not a real-time claim.

## 7. Acceptance

Three training seeds, evaluated on the same 12-seed block as section 1,
through the shared `eval/metrics.py` path, reported as a distribution and
never as the best seed. The coupled row has to beat **8/12 at 116-142 s**, not
the scripted oracle -- the oracle takes 226-276 s through the same channel and
is not an upper bound on anything here. Section 5b puts the ceiling at
**11/12**, so the honest way to read a trained row is against both numbers:
8/12 is what it must beat, 11/12 is what choosing the entry time can buy, and
anything above 11/12 means the policy found something the entry-time sweep did
not.

The claim the table has to support is narrow and should stay narrow: the
learned layer times the entry, and completion rate is where that shows up.
