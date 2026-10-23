# T1B_V4_PRECHECK — why V3 still returns to "earliest" at 10k, and the V4 repair (upper, 2026-10-19)

Input: lower midterm `evidence/T1B_V3_MIDTERM_20261019/` (`b2a65c5`; seed 0, 0/2k/5k/10k).
`v3_midterm_coverage_and_slope.json` is computed from its `REAL_DECISION_INTERVALS.csv`,
`training/*/decisions.jsonl` and `audit_*`.

## Findings

1. **V3's decision structure works** (gate M: no learning decision with a live plan; 1.5–1.6
   decisions per episode; intervals end at plan loss or the terminal).
2. **Data starvation of the unchosen intents.** E-SAC 10k: 449 real decisions, 11 % non-earliest;
   economy × latest = 7, urgent × latest = 11 (5 timeouts). With ~1.5 decisions per episode,
   epsilon 0.10 explores ~0.15 decisions per episode.
3. **The critic is linear in the arrival coordinate and extrapolates.** In every one of the 15
   three-candidate dev states Q(earliest) is top and Q drops by the same step per candidate; the
   E-SAC step grows −1.0 (2k) → −7.8 (5k) → −17.6 (10k). At 292140 (economy) E-SAC 10k gives
   −10 / −34 / −59 while the truth (commit + keep) is −130 / −130 / +56. The few urgent timeouts at
   late candidates set a global slope; the economy evidence (latest 5/7 success vs earliest 63/97
   terminal successes, 34 resource exhaustions) is too thin to bend it.
4. **The MPC counterfactual already acts in the right direction.** E-TCPF's step is −0.4 / −0.5 /
   −3.4 (2k/5k/10k), 5–8× flatter than E-SAC, and at 5k it moved 2/15 states off "earliest" (the
   small 5k return gain). With one alternative per decision it covers half of the unchosen intents
   and cannot outweigh the real-data extrapolation.

## V4 (`env/t1_b_feasible_v4.py`, `train/t1_b_v4_sac.py`, `--interface v4`)

- Behaviour epsilon 0.5 at learning decisions (both arms; ~0.75 explored decisions per episode,
  each non-earliest candidate ~1/6 of decisions). Off-policy; deterministic evaluation unchanged.
- E-TCPF pairs **every** unchosen verified candidate at each decision (one factual + ≤2 alternative
  MPC prefixes, each with its own factual-consistency gate), so the critic receives a model-based
  relative target for every intent at every decision state. Same 5 % calibrated / 10 % capped share.
- Unchanged: V3 committed decisions, task, D/B, G*, MPC, truth, network, init, temperature 0.5,
  optimiser, trainer, V3 teacher Replay (the learning decisions are V3's: the lower's V3 teacher file
  `9c91cb76…` is reused).
