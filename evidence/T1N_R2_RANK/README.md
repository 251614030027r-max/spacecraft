# T1N_R2_RANK — final technical diagnostic of the cross-timescale ranking term (2026-10-18)

Research-layer ruling of 2026-10-18 (B, revised): the auxiliary term of D1 / TCPF is a
confidence-weighted ranking loss on the sign of D = S(a~) - S(a), S = MPC 2 s predicted reward +
discounted G* plan reward to the planned entry (`env/t1_n_cf.py`, `train/t1_n_sac.py`).
Development openings 292100-292117 (292114 not finished: stopped for time), 2 review states each,
random executed actions; both branches flown in truth to the end (keep; earliest feasible at a
no-plan review). Exploratory, not a gate.

| Quantity | Value |
|---|---|
| pairs / with both plans / with confidence w > 0 | 34 / 29 / 26 (76 %), mean w 0.55 |
| zero-weight reasons | real request rejected 6, no plan 5, same arrival 4, small signal 3, alt rejected 2 |
| truth-identifiable pairs (\|truth difference\| > 0.5) | 25 |
| sign agreement with the long-run truth difference: plan only / plan + MPC | **15/25 / 15/25**; weighted pairs 14/24, confidence-weighted 0.64 |
| correlation with the truth difference: plan only / plan + MPC | 0.10 / 0.11 |
| by kind (plan+MPC) | request shift 10/16, keep vs shift 2/3, replan vs keep 3/6 |
| MPC correction: change of D (median \|D - D_plan\|) vs \|D_plan\| | 0.053 vs 1.36 (about 4 %); **0 ranking flips** |
| first 2 s reward error vs truth: MPC prediction / plan nominal | 0.006 / 0.166 |
| shadow wall per branch | 1.97 s (one 2 s MPC rollout) |

`smoke_rank.json`: 270-step TCPF smoke (16×16): the ranking term is probed (gradient ratio
0.015–0.043), its weight frozen at step 200 (lambda 2.5) and then applied; 89 + 2 intervals closed,
0 dropped.

Files: `rank_summary.json` (summary and every pair), `rank_raw.tar.gz` (per-opening JSON),
`smoke_rank.json`. Command: `python -B -m experiments.t1n_check rank --first 292100 --count 20
--reviews 2` (code of this commit).
