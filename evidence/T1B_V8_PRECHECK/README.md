# T1B_V8_PRECHECK — upper pre-check for V8 (ECPG + risk-triggered deployment verification)

Sandbox, offline except the one closed-loop smoke below. Every number points at
`V8_PRECHECK_OFFLINE.json` (made by `v8_precheck_offline.py` from the truth / model files in git and
`candidate_features.jsonl`, the G* first-decision candidate features from the deterministic reset,
made by `candidate_features.py`). Data: 79 openings with truth for every first-decision candidate
(dev 292120-147 = 15, confirm 292500-531 = 32, independent 293100-131 = 32; continuation V1 teacher).

## A. The learning space is a rare-event problem

| | all 79 | dev | confirm | indep |
|---|---|---|---|---|
| always earliest / first-decision oracle return | 22.34 / 28.62 | 37.34 / 51.57 | 20.96 / 22.31 | 16.68 / 24.17 |
| oracle gain from rescue states (> 50) / from the rest | 4.69 / 1.59 | 12.39 / 1.84 | 0 / 1.35 | 5.78 / 1.72 |
| rescue states (earliest fails, a later candidate captures) | 292140, 293117 | 292140 | — | 293117 |
| capture earliest / oracle | 0.785 / 0.810 | 0.867 / 0.933 | 0.781 / 0.781 | 0.750 / 0.781 |
| non-earliest candidate fails while earliest captures | 24 / 123 | 4 / 26 | 11 / 49 | 9 / 48 |

75 % of the oracle gain is two states out of 79; elsewhere the differences are a few return units
while one alternative in five is a -180 failure. The oracle capture headroom is +2.5 points
(confirm: 0), so a "+5 points capture" criterion cannot be met by any selector on this task.

## B. Preference distillation alone does not generalise (upper bound with truth labels)

A masked categorical head trained by CE to the event-aware **truth** preference (the best label
ECPG could ever receive), leave-one-block-out, 5 seeds (return; catastrophic picks per seed):

| test block (train n) | earliest | 52-D obs | obs, earliest prior +1 | obs + G* features | + features, prior +1 |
|---|---|---|---|---|---|
| dev (64) | 37.34 | 37.97 | 37.85 | 37.60 | 37.57 |
| confirm (47) | 20.96 | 8.54 (2-3) | 9.98 (2) | 0.39 (3-5) | 11.00 (1-3) |
| indep (47) | 16.68 | 12.38 (0-1) | 14.75 (0-1) | 13.84 (0-1) | 16.42 (0) |

Random-split learning curve (16 held out, 12 repeats, features + prior): gain vs earliest
-9.5 / -0.8 / -2.6 / -2.6 at 16 / 32 / 48 / 64 training states. No learned selector without
verification beat "always earliest" on an unseen block.

## C. What is learnable: the risk of the default

Earliest-candidate failure from the same inputs (small MLP, leave-one-block-out): AUC dev 0.96,
confirm 0.83, indep 0.98; the rescue states rank 1st of 15 (292140) and 5th of 32 (293117). The
real-TD critic of the V7 runs already ranks it: min Q(s, earliest) vs failure on the 15 dev
openings, AUC 1.0 for E-SAC 2k/5k/10k (`evidence/T1B_V7_ROUND/final_recovered/FOUR_LINK_CHAIN.csv`).

## D. First-decision recombination: verify only where the risk is high

Model labels = V7 side bench real-controller branches; trigger = C classifier > 0.3 (+ E-TCPF 10k
deviations); not triggered = earliest:

| | earliest | trigger rate | risk-triggered verify | random trigger, same rate | verify all | oracle |
|---|---|---|---|---|---|---|
| confirm | 20.96 | 14/32 | 21.68 | 21.47 | 22.12 | 22.31 |
| indep | 16.68 | 15/32 | 23.45 | 20.19 | 24.17 | 24.17 |

Capture 0.781 in every verified arm (= oracle). The deployed V8 trigger uses the critic's
min(Q1, Q2) of the proposal < 0 instead of a separate classifier (lower Stage I checks its AUC).

## E. Closed-loop smoke (sandbox, `sandbox_deploy_v8_0k_verify_all_293117.json`)

`experiments.t1b_v8 deploy --verify all`, untrained V8 (proposal = earliest), opening 293117:
first decision predicted 100 s exhausted -131.7, 120 s unsafe -177.4, 140 s success +53.3
(truth scans: -131.7 / -177.6 / +53.1) -> executed 140 s; second decision at 126 s verified two
successes. Truth episode: success, return 53.13 (always earliest: -131.7). 5 branches, 475 s of
verification wall on a loaded 4-core sandbox.

Status: exploratory upper evidence for the V8 design; Stage I on the lower decides.

## F. V8 learner mechanics (sandbox smoke, `sandbox_smoke_e_ecpg_400/`, `sandbox_contract_v8.json`)

`experiments.t1b_v2 train --interface v8 --arm e_ecpg --smoke --steps 400` (V3 teacher, 40
transitions): 3 first decisions released, 3 complete preference rows (6 branches, gate weights
0.91-0.98, factual return error <= 0.5); CE active from step 216 (64 usable rows, CE gradient on
the head ~0.3-0.8 vs SAC ~0.02); the head is still near uniform after 122 actor updates (lr 1e-4).
Contract: shared networks equal to V1 interface_0k, deterministic choice = earliest. Tests:
`tests/test_t1_b_v8.py` 6/6, `tests/test_t1_b_v2.py` unchanged and passing.
