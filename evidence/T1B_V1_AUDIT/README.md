# T1B_V1_AUDIT — upper offline audit of T1-B V1 (2026-10-19)

Inputs: `evidence/T1B_V1/` (`bee2d43`; V1 science `636b655`). Offline recomputation on the frozen
models and logs, plus one truth scan on the development block 292140–292147 (no learning, no formal block).

## A. Why V1 collapses to the earliest candidate (`policy_saturation_audit.json`, `cf_label_and_coverage.json`)

| Model (first review, 8 dev states) | alpha | z of mean beyond the first bin edge | actor gradient norm | categorical entropy | Q(earliest) − Q(next) |
|---|---|---|---|---|---|
| E-SAC 0k | 0.015 | −0.4 (interior) | 2.2e-3 | 1.09 | 0.00 |
| E-SAC 2k | 0.015 | 4.6–7.2 | 9.9e-6 | 4.5e-6 | 0.56 |
| E-SAC 10k | 0.015 | 5.7–8.7 | 1.1e-6 | 0 | 4.81 |
| E-TCPF 2k | 0.015 | 4.8–6.5 | 1.1e-5 | 1.4e-6 | 1.01 |
| E-TCPF 10k | 0.015 | 5.9–7.9 | 3.4e-7 | 0 | 6.12 |

1. **Temperature scale (root cause).** alpha = 0.005 × review gap = 0.015 was inherited from the 2 s
   continuous line. The exact categorical SAC optimum is p_i ∝ exp(Q_i/alpha); at 2k the Q gap of
   0.56 already gives p_other ≈ e^-37 — softmax(Q/alpha) of the frozen critics is exactly 1.000000 on
   the earliest candidate in all 16 model-states. The actor did what its objective asks: argmax.
   The Gaussian-CDF probability gradient is mathematically correct; it vanishes (×1e-3..1e-4) because
   the mean sits 5–9 σ inside the earliest bin.
2. **Coverage starvation (lock-in).** After the first 1k online steps 100 % of learning decisions
   executed the earliest candidate (E-SAC: non-earliest 74 online + 11 teacher choices in 10k; E-TCPF 63
   online). The critic's Q for the other candidates is extrapolation: on every state it is almost
   linear in tau and always lower for later arrival. The ranking formed at 2k can never be corrected.
3. **The ranking is wrong where it matters.** Truth of each first-review candidate (continuation =
   V1 teacher, `candidate_truth_scan/`): the best first choice is the earliest in 3/8, the middle in
   2/8, the latest in 3/8 openings. 292140 (economy, B = 1.53 m/s): 100 s and 120 s exhaust the budget
   (−129.8), 140 s captures (+56.0); the frozen 10k critics rank 100 s highest. Best first choice:
   8/8 captures, mean discounted raw return 60.87; always-earliest 7/8, 36.50 (V1 readout 35.58,
   strong G* 37.44). The learnable decision value exists and is state dependent (economy → later,
   urgent → earlier/middle).

## B. Why MPC feedback changes nothing in deployment (`cf_label_and_coverage.json`)

1. Consequence of A: both arms deploy argmax = earliest, G* and MPC are deterministic → identical
   trajectories. The TCPF critic is different (its Q gaps are 30–40 % steeper) but cannot reach a
   saturated actor.
2. The relative target is dominated by the critic's own bootstrap: with the frozen 5k target critic,
   sign(Y) = sign(bootstrap part) in 95.4 % of 1003 valid pairs; the MPC reward part is larger in only
   9.4 %. Under coverage starvation the bootstrap at the alternative successor is itself
   extrapolated, so the label repeats the critic's ranking.
3. The pairs did not test the decision: 1961 of 2021 valid pairs had the earliest as the factual
   action and 1668 compared "advance 15 s" with "keep the plan" (the "farthest" alternative rule never
   pairs the delay side); the 6 s prefix favours the factual action in 81 % (median −0.50).

## Minimal repair (V2, learning layer only)

alpha per task decision 0.5 (no gap scaling, fixed); behaviour epsilon 0.10 uniform over verified
candidates during training only; CF alternative drawn uniformly among the other candidates by a fixed
hash. Task, D/B, support, G*, MPC, truth, network, init, optimiser, trainer, expert Replay, CF gate
and the 5 %/10 % share unchanged. Code `env/t1_b_feasible_v2.py`, `train/t1_b_v2_sac.py`,
`experiments/t1b_v2.py`.

Scripts: `policy_saturation_audit.py` (offline), `candidate_truth_scan.py` (same logic as
`experiments.t1b_v2 scan`; run on 636b655 + V1 env, sandbox).
