# T1B_V5_PRECHECK — the missing long-horizon signal, and the full-horizon MPC counterfactual (upper, 2026-10-19)

## 1. V4 sandbox (formal hyper-parameters, E-TCPF, seed 0, 5k; V3 teacher `9c91cb76…`)

Mechanics fixed: no decision with a live plan; 53 % non-earliest decisions after warm-up; both
counterfactual slots valid; entropy 0.86 (2k) → 0.61 (5k), actor gradient ≥ 0k. But the critic
still ranks the earliest first in 8/8 first-review states with a uniform −0.7 step
(`v4_sandbox_e_tcpf_5k_audit.json`). In the sandbox run's own real decisions, economy × latest
succeeds 8/12 vs economy × earliest 10/14: single real returns do not show the advantage.

## 2. Why: the value of the arrival time is a small, late signal under a large variance

Truth scan per candidate (`candidate_features.jsonl` + V1/V2 scans): later arrival usually spends
less total delta-v (292142: 1.06 → 0.80 m/s; 292145: 1.31 → 1.03) — the transfer bound is only
0.0–0.24 m/s, the difference accrues in the descent after arrival — worth +2..+3 return when the
deadline slack allows; too late → timeout (292143, 292127, 292146); rare budget-critical cases
(292140) swing ±186. Single real returns vary by ±100..200 between success, exhaustion and
timeout, so 20k steps (~400 decisions) cannot resolve a +2..+3 difference, and a 6 s MPC prefix
cannot see it (V3 precheck).

## 3. The frozen MPC on its model, to the end of the episode, reproduces the truth ranking

`full_horizon_mpc_counterfactual_stride5.jsonl` (`full_horizon_mpc_counterfactual.py`; shadow =
nominal RK4 model + the same MPC, QP every 5 controls; continuation = V1 teacher; 15
three-candidate dev states vs truth):

| | agreement |
|---|---|
| best candidate (within 1.0 of the truth best) | **11/15** |
| pairwise sign where truth differs by > 0.5 | **31/38** |
| outcome event | 37/45 |

Misses are budget-marginal (292140, 292141, 292145: model delta-v ~0.1 m/s high). QP every 10
controls is ~1.8× faster but loses accuracy (`…_stride10.jsonl`); stride 5 is kept.

## 4. V5 (E-TCPF only; `env/t1_b_feasible_v5.py`, `train/t1_b_v5_sac.py`, `--interface v5`)

At the first learning decision of each episode, the factual intent and one other verified
candidate (fixed hash) are rolled out with the MPC on its model until the episode ends or a new
learning decision is due (physical reviews with a live plan are kept, as in the real env). The
relative target is the factual-anchored paired model return (no critic bootstrap when both end);
admitted only if the factual branch reproduces the real outcome (same terminal event within 10
steps, or the same plan loss at the same step; |return error| ≤ 5; weight 1 − error/5). Same 5 %
calibrated / 10 % capped share. Real interval TD (MC for committed intents) stays the primary value
basis. E-SAC under `--interface v5` has no counterfactual and is identical to V4.

Smoke (600 steps, `v5_smoke_*`): factual model vs real 67.2/67.8, 61.2/62.9, −133.7/−130.4
(timeout), 75.2/76.3; plan losses reproduced at the exact step (25/25, 14/14, 28/28). Rollout wall
10–42 s each; sandbox 1.46 s per transition (4 envs) vs 0.75 for V3.

## 5. V5.1 (2026-10-20, after the lower's V4 5k delivery `2c7283b`)

V4 seed 0 (`evidence/T1B_V4_RESOURCE_RESTART1_20261020/`): with ~40 % non-earliest real decisions
both arms still rank the earliest first in 15/15 states at 5k (E-SAC step −1.15 → −7.85, E-TCPF
−0.51 → −1.40); 97 % of the 6 s labels follow the bootstrap sign. Coverage was fixed; the missing
piece is the long-horizon signal — V5 had not been run (the lower ran V4, then the host crashed).

Changes before the first V5 run:
- every other verified candidate is paired at the first decision (2 labels per episode);
- the model branch runs to the end of the episode; a model-side re-decision after a lost plan takes
  the earliest candidate (exactly the validated scan of §3). The first V5 smoke showed the model
  losing plans that the real system kept (stride-5 QP tracking), which the gate then rejected;
- the factual branch must end in the real terminal event (±10 steps, |return error| ≤ 5);
- the counterfactual gradient share is 10 % calibrated / 20 % capped (V1–V4: 5 / 10): the label is
  independent full-horizon model information, no longer a bootstrap echo.

Smoke (500 steps, `v5_1_smoke_*`): 4 of 5 decisions admitted (8 labels); factual model vs real
65.6/66.4, 63.0/65.9, −117.4/−116.9 (timeout), −131.9/−134.4 (timeout); e.g. chosen 120 s → real
timeout while the model predicts 80 s → success (+66); one model exhaustion not matched by the real
episode was rejected. 0.86 s per transition in the sandbox (V4: 0.70).
