# T1P_ROUND1_PRECHECK / extended_08f52b6 — upper extended test of T1-P (2026-10-18, 14:30–15:07 UTC)

User instruction: "继续测，再给你半小时". Code: science `08f52b6` (**not frozen**; under test).
Development openings 292000–292002 only (the pre-numbered three; no opening was swapped).

**Verdict: FAIL 3/3 — no pre-numbered opening completes the chain; conditional GO not met; no training.**

## What changed against 8296c59 (all in `08f52b6`)

- `env/t1_p_stage.py`: the local SCP range cap uses the stopping distance of the full relative
  velocity `|v|^2 / (2 F/m)` (was the radial component only).
- `env/t1_p_review.py`: `geometric_rule` revised four times during the window (pathway diagnosis
  only, before any formal evaluation). Revision 4 = `08f52b6`: staging point 8 m from the target
  on the LOS meridian at the port's latitude about the angular-momentum axis, desired velocity =
  (staging − q) / 20 s capped at 0.15 m/s, mapped to the stage action and **clipped per axis**.
  Revisions 1–3 were working-tree edits whose code is not preserved (their numbers in
  `rule_revisions_summary.json` are exploratory).

## Results

| Rule | 292000 | 292001 | 292002 | Pattern |
|---|---|---|---|---|
| original (8296c59) | timeout, 13–28 m | unsafe distance (30 m) at 113 s | timeout, 15–23 m | outward drift |
| rev 1 | unsafe 164 s | unsafe 114 s | unsafe 167 s | chases the rotating mouth point, flung outward |
| rev 2 | timeout, min 6.35 m | timeout, min 6.13 m | timeout, min 6.12 m | parks at the 6 m sphere edge at high LOS latitude |
| rev 3 | timeout, min 9.4 m | timeout, min 7.3 m | timeout, min 7.4 m | inside 9 m before the port arrives |
| **rev 4 (08f52b6)** | timeout, min 7.96 m | timeout, min 7.02 m | timeout, min 7.10 m | holds 7–10 m, bridge never starts |

No truth-safety violation for rev 2–4. Bridge and descent never entered with any rule.

## Bridge diagnosis (rev 4, read-only wrapper `bridge_diag.py`, files `diag_2920xx.json`)

| Opening | steps with r ≤ 9 m | of those envelope ≥ 0.02 | steps with all geometric conditions (r ≤ 9, envelope, angle ≤ 25° at some tau) | conditional G* status at those steps |
|---|---|---|---|---|
| 292000 | 64 | 64 | 23 (first at 222 s) | 23 × `beyond_deadline` |
| 292001 | 73 | 73 | 20 (first at 114 s) | 18 × `beyond_deadline`, 2 × `infeasible` |
| 292002 | 28 | 28 | 0 (port angle ≥ 80° while near) | — |

The geometric readiness is reached, but the conditional G* (T1-N `deadline_margin_s = 60`) only
plans an arrival ≤ 240 s. The port passes the chaser's LOS once per ~150 s; the ready windows fall
at ≥ 222 s (292000) or the early window is infeasible (292001, 114 s).

## Diagnostic with the deadline margin relaxed to 20 s (`margin_diag.py`; not a candidate, not committed code)

| Opening | event | first bridge | first descent | min range |
|---|---|---|---|---|
| 292000 | timeout | 230 s | 266 s | 3.23 m |
| 292001 | timeout | — | — | 7.02 m |
| 292002 | timeout | — | — | 7.10 m |

Even with arrival allowed to 280 s only one opening reaches the descent, too late to capture in 300 s.

## Not done

The research-layer repair order `upper/T1_P_EMERGENCY_CLOSED_LOOP_REPAIR_20261018.md` (collab
`98e8d9f`, items A–E: in-stage re-planning to the fixed stage goal, d_max 0.70 → 0.90 if needed,
bridge at r ≤ 10.5 m without the angle prefilter, Euclidean-normalised rule) was **not implemented**
in this window (it was read only at report time). Rev 4 still clips per axis (item D).

Reproduce (rev 4): `python -B -m experiments.t1p_check pipeline --opening 2920XX --output p.json`;
diagnostics: `PYTHONPATH=. python -B evidence/T1P_ROUND1_PRECHECK/extended_08f52b6/bridge_diag.py 2920XX out.json`,
`... margin_diag.py 2920XX out.json 20`.
