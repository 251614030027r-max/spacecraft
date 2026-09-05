# Precapture v2 close-out: Q1--Q3 and evidence corrections

Date: 2026-09-05
Scope: post-S4 static review and extraction from already completed runs. No new
experiment is introduced by this document.

## Q1: Pure MPC h50 illegal-entry events

Source:
`logs/precapture_planning_v2/pure_mpc_h050_tw1000_5seeds_entry_events.json`.
The frozen h50 block completed 1/5 episodes; the other 4/5 reached the 300 s
limit. All five episodes had zero active-constraint violations. Each timeout
episode recorded exactly one illegal outside-to-inside entry-plane crossing;
the completed episode recorded none.

At the entry plane the disc radius is 3.151 m, the target-frame total-speed
limit is 0.35 m/s, and the closing-speed limit is 0.20 m/s. Negative axial
velocity denotes motion in the closing direction; the positive closing-speed
column is its reported magnitude.

| Seed | Outcome | Event time (s) | Off-axis angle (deg) | Axial velocity (m/s) | Closing speed (m/s) | Total speed (m/s) | Radial distance (m) | Illegal cause |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 262000 | timeout | 34.1 | 7.979 | -0.275840 | 0.275840 | 0.289901 | 0.630781 | closing speed |
| 262001 | timeout | 2.9 | 73.705 | -0.378841 | 0.378841 | 0.580924 | 15.393609 | outside disc, closing speed, total speed |
| 262002 | timeout | 36.8 | 10.686 | -0.265737 | 0.265737 | 0.289519 | 0.849106 | closing speed |
| 262003 | completed at 82.2 s | -- | -- | -- | -- | -- | -- | no illegal event |
| 262004 | timeout | 40.5 | 13.365 | -0.258763 | 0.258763 | 0.302886 | 1.069184 | closing speed |

This rejects the hypothesis of repeated counted impacts: the four failures each
crossed illegally once, failed to latch the terminal region, and then remained
without a later legal outside-to-inside crossing until timeout. It does not show
an active safety-constraint violation; illegal entry is a diagnostic event and
the outer-region constraints stay active.

## Q2: why exactly one is not double counting

`evaluate_terminal_entry_crossing` declares a crossing only when the previous
port-axial coordinate is strictly outside the 4.5 m plane and the current one is
inside or on it (`previous_offset > 0` and `current_offset <= 0`). A trajectory
that remains inside cannot produce another event. Legality is then evaluated at
the interpolated crossing using disc radius, total speed, and closing speed.

`tests/test_precapture_task.py::test_entry_disc_crossing_semantics` explicitly
checks that an already-inside continuation is not a crossing, and
`test_environment_latches_only_on_legal_entry_disc_crossing` checks that the
terminal state latches only on a legal crossing. The S4-R development/report
blocks also record illegal-entry episodes in only 1/5 and 2/5 cases, rather than
one event in every seed. Together, the implementation, unit tests, and observed
variation rule out a plane-step double-counting explanation.

## Q3: M1 compute interpretation

Commit `513b715` changed terminal activation in the truth-margin, MPC-margin,
and analytic-linearization paths from
`terminal_latched or range_m <= terminal_activation_range_m` to
`bool(terminal_latched)`. It did not reduce the fixed constraint dimension:
`normalized_precapture_constraint_margins` and its linearization remain
`corridor_facets + 7` rows (15 rows for the default eight facets). Before a
legal latch, terminal rows stay inactive at the fixed inactive margin while the
outer rows remain active; after latch, terminal rows activate.

Therefore the observed command-time difference between completed batches must
not be attributed to "fewer constraint rows." Both reported batches were
serial, single-process measurements, but wall-clock milliseconds can vary with
executed code paths, solver behaviour, warm starts, episode composition, and
machine load. The defensible statement is only the measured batch result: h20
was within the 0.1 s budget in its five-seed block, whereas h50 exceeded it at
mean and p95. No causal compute attribution has been established.

## Evidence corrections and stopping state

- The proposed S1 3 x 3 scan is **NOT DONE** as reportable evidence. A discarded
  diagnostic selected the same downstream window at all points and measured no
  timing-cost surface.
- `curve_a_divergence_seed99.json` is a zero-action constructed mechanism
  illustration, not a controller-generated failure, not a Pure-MPC rollout,
  and not a performance denominator.
- S4-R is frozen and closed at 3/5 on reporting seeds, with 5/5 zero active
  constraint violations. It fails as a stable classical baseline; no further
  S4 tuning is authorized by this close-out.
- Superseding status (2026-09-06): S5 was executed after F1--F3 and failed its
  interface-improvement gate; S6 training was therefore not started. See
  `docs/PRECAPTURE_V2_FASTTRACK_RESULTS.md`. No 20-episode formal evaluation
  was run.
