# Reward units fix: the safety proximity warning was not time-integrated

*2026-09-19. Branch `claude/sac-mpc-coupling-design-ns7g6i`.*

## Symptom (from the artifacts)

Three adaptive-task training seeds (262410/411/412), ~30k/60k decisions each,
`train.monitor.csv` uploaded to this session. Per-seed return split by
`completed`:

| seed (file) | completion | COMPLETED return (min/med/max) | NOT-COMPLETED return (min/med/max) |
|---|---|---|---|
| 581bf761 | 302/418 = 0.72 | -87.1 / **-78.1** / -34.6 | -35.8 / **+3.8** / +6.2 |
| 6342f1e6 | 287/408 = 0.70 | -109.8 / **-77.2** / -40.0 | -26.6 / **+3.9** / +6.6 |
| 9f084d34 | 283/405 = 0.70 | -86.8 / **-78.7** / -40.8 | -56.2 / **+4.6** / +6.7 |

A **successful legal capture returned ≈ -75**; a **do-nothing hover timeout
returned ≈ +4**. The completed samples carry `terminal_region_active=True`,
`hybrid_waypoint_radius_m=3.0` (full commit), `illegal_terminal_entry_count=0`,
`l≈47` decisions — i.e. clean, fast, *legal* completions, scored 80 points
below hovering. That inverts the objective and, worse, trains the SAC critic to
rank hover above capture — which is exactly the value the deployment gate reads.

## Root cause (from the code, `env/reward.py`)

In `PrecaptureReward.compute` the three continuous penalties are **rates**
integrated over the step (they carry `time_step_s`):

```
time_penalty   = -time_weight   * time_step_s
force_penalty  = -force_weight  * time_step_s * mean(force**2)
torque_penalty = -torque_weight * time_step_s * mean(torque**2)
```

The safety proximity warning was **not**:

```
safety_penalty = -safety_weight * sum(warnings)      # no time_step_s
```

`warnings` is a dimensionless [0,1] sum that fires whenever a margin is inside
the 10% buffer. Applied per 0.1 s control step without `time_step_s`, it accrues
**10× too heavily per second** relative to the physical costs. A legal terminal
approach spends ~40 s (~400 control steps) in the terminal region; in a Λ>1
co-rotating capture it necessarily rides near the closing/total-speed buffer, so
`sum(warnings)≈0.25` per step ⇒ ≈ **-100** over the approach, swamping the +20
completion event.

Decomposition of a real trained completion (r≈-75): shaping +8, time -0.9,
force/torque -0.1, event +20 ⇒ **safety ≈ -102**. Matches.

## Fix

One edit: give the safety term the same `time_step_s` factor as the other three
continuous penalties. `safety_weight` is unchanged; no weight was tuned.

```
safety_penalty = -safety_weight * time_step_s * sum(warnings)
```

This is a dimensional-consistency correction, not a reward re-shaping: it does
not touch the wait/enter decision, the event rewards, or the shaping potential,
and it leaves Pure MPC's number untouched (a clean MPC completion never enters
the buffer, so its safety term is 0 before and after).

## Verification (sandbox, `scratchpad/measure_reward.py`, adaptive task, no policy)

Summed reward components over full scripted episodes, before vs after:

| behaviour | completed | safety before | safety after | return before | return after |
|---|---|---|---|---|---|
| full-commit (Pure MPC), seed 0 | yes | 0.00 | 0.00 | **+26.55** | **+26.55** |
| full-commit (Pure MPC), seed 1 | yes | 0.00 | 0.00 | +26.33 | +26.33 |
| full-commit, seed 2 | no (timeout) | 0.00 | 0.00 | +3.98 | +3.98 |
| inertial hold, seed 0 | no (timeout) | 0.00 | 0.00 | +6.56 | +6.56 |
| inertial hold, seed 1 | no (hard fail) | -5.14 | -0.51 | -23.88 | -19.25 |
| inertial hold, seed 2 | no (hard fail) | -5.55 | -0.55 | -24.78 | -19.79 |

- A **clean legal completion is unchanged at +26** — Pure MPC stays a fair
  strong baseline; nothing was shaped to make it lose.
- **Hard failures stay strongly negative** (dominated by the -20 event); the
  safety term shrinks by 10× as intended but does not flip their sign.
- Applying the same 10× to the trained grazing completions (safety -102 → -10.2)
  lands them at ≈ **+16.8**, now cleanly **above** hover (+4–6). Ordering
  restored: clean completion (+26) > grazing completion (+17) > hover (+4) >
  hard failure (≤ -19). The hard-failure wall at -20 is untouched, so there is
  no new incentive to cross a boundary to finish faster.

## Discipline

- One principled physical-scale fix (the pre-authorised knob), not a grid.
- No manufactured gap: Pure MPC's clean completion is bit-for-bit unchanged.
- Reward does not encode the answer: the fix is uniform across all margins and
  says nothing about when to wait vs enter.
- Retrain from zero (all three seeds); the old critic is discarded because it
  was trained under the inverted signal.
