# The gap is the entry window, and every failure is the same failure

Measured 2026-09-07 in the upper-level sandbox on `1fe5549` (**no F1** -- see
*What F1 changes*). 24 episodes: `precapture_planning`, fixed-setpoint
reference (no upper layer), `terminal_weight=1000`, `input_weight=0.01`,
`analytic_local`, truth state, seeds 262000-262011, horizons 20 and 50. Runs
were parallel, so **no compute number is taken from them**.

## 1. The result

| horizon | completed | failed |
|---|---:|---:|
| h20 (2 s) | **8/12** | 4 |
| h50 (5 s) | **1/12** | 11 |

**All 15 failures are the same failure**, and it is not a violation:

- 15/15 crossed the 6 m entry plane **illegally** and never latched
- **0/15** had any truth-geometry violation -- no keepout, no corridor, no
  speed, no FOV, nothing
- after failing to latch, the chaser flew to the desired pose anyway and
  **parked on it**: median distance to the goal when the 300 s clock ran out is
  **0.000 m** (max 0.005 m), attitude error 0.0-0.1 deg, relative rate
  0.0001-0.0014 rad/s

So the vehicle is sitting exactly where the task wants it, holding a pose the
completion test would accept on every other criterion, and it gets nothing --
because it came through the door wrong once, 200 s earlier, and a fixed-setpoint
regulator has no notion of backing out and coming round again.

Seed 262001 at h20 shows the recovery is available: it entered illegally,
retreated and re-entered legally, and completed in 121.8 s. So this is a
**decision** deficiency, not an impossibility.

## 2. Why this is the gap the method needs

The entry plane is a one-shot legal crossing: cross it too fast or too far
off-axis and the episode is quietly unwinnable while every instantaneous
constraint stays satisfied. Nothing in a receding-horizon regulator's cost
represents "this crossing was the last one that counted". The decision the
task actually poses -- *take this window, or wait and come round with the
tumble* -- has no representation in the lower layer at all.

That is what an upper layer is for, and it is a schedule decision, not a
tracking one: the same interface pattern as `References/北航编队.pdf`.

## 3. The horizon result is measuring the absence of F1, not the horizon

h50 being **worse** than h20 (1/12 against 8/12) is not a claim about
horizons. Without F1 the MPC cannot see the entry disc's closing-speed limit
before it latches, so a longer horizon simply drives harder at the goal and
busts the gate more reliably. The lower window's own S3' on `e212b4a`, with
F1, reports the opposite ordering (h50 3/3, h20 2/3), which is consistent:
F1 restores the anticipation, and then the longer horizon helps.

**So the h20-vs-h50 question cannot be settled here.** It needs `e212b4a`.
What survives without F1 is section 1's failure signature, which is about the
gate itself and is the same under both horizons.

## 4. What to do with it

1. Re-run this 12-seed sweep on `e212b4a`, both horizons. The number that
   matters is how many failures remain *with* F1 and whether they are still
   entry-gate failures. F1 gives the MPC 2 s (h20) or 5 s (h50) of preview;
   the window decision needs more lead time than that, so failures are
   expected to persist.
2. The learned upper layer's job statement follows directly: **choose the
   entry time**, expressed through the 3D waypoint channel, which
   `docs/PRECAPTURE_REFERENCE_TRACKING_DIAGNOSIS.md` now shows is lossless in
   the target body frame.
3. The comparison table writes itself: Pure MPC busts the gate on a fixed
   fraction of seeds and cannot recover; the coupled method times the entry.
   Completion rate separates them, which is what the four main-table metrics
   were chosen for.

## 5. Reproduce

```
python -B -m experiments.diagnose_precapture_reference \
  --seed 262003 --horizon 20 --guidance fixed --output OUT.json
```

Read `completed`, `terminal_region_active`, `illegal_terminal_entry_count`,
`violation_steps`. Evidence for all 24 episodes is under
`logs/precapture_planning_v2/upper_diagnosis/`.

## 6. Limits

One episode per cell, 12 seeds, one configuration, no F1, truth state, no
estimator. This locates a mechanism and sizes it; it is not a reportable row
and no training conclusion rests on it.
