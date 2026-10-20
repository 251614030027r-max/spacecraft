# T1B_V7_PRECHECK — the counterfactual MPC was not the real controller (upper, 2026-10-20)

Input: lower V6 final review `evidence/T1B_V6_ROUND/final_review_20261020_2230/` (`0cf973c`).

## What V6 showed

V6 fixed the decision-level learning structure: the real-only arm (E-SAC) is now a clean negative
control (2k/5k/10k identical, 5/15 agreement, earliest everywhere), and the MPC arm moved
(7/15 → 6/15 → 7/15, first non-earliest choices, +0.05..+1.0 return) — but not decisively. The
lower's reference analysis found why the MPC advantage cannot be trusted where it matters:
292140 (140 s really captures, +56; model branch −129, budget exhausted), 292145 (+3.3 really,
−183 in the model), 18 / 70 sign errors among gated reference pairs. The errors sit exactly at the
capture / exhaustion / timeout boundaries that decide the task.

## Root cause

The full-horizon branch used `IntervalModelConfig.qp_stride_controls = 5` (from V1, a cost
shortcut): the MPC QP was solved only every 5th 0.1 s control and stale nominal controls were
played in between. That is a different, weaker controller than the one that flies; it spends
0.1–0.25 m/s more delta-v (292140: model 1.532 vs real 1.418 at 140 s; 292141: 2.05 vs 1.81),
which turns real successes into model exhaustions at the budget boundary.

Same four boundary openings, same nominal model, QP every control (`…_stride1_boundary_openings.jsonl`):

| opening | truth (100/120/140 or 80/100/120) | stride 5 (V5.1/V6) | **stride 1 (real controller)** | stride 2 |
|---|---|---|---|---|
| 292140 | −129.8 / −129.6 / **+56.0** | −129.8 / −129.6 / −129.4 | **−129.8 / −129.6 / +56.0** | −129.7 / −129.6 / +55.7 |
| 292141 | 54.0 / 50.6 / 51.6 | 48.7 / −139.8 / −139.6 | **53.8 / 51.6 / 52.1** | 53.6 / 51.1 / 48.7 |
| 292145 | 62.3 / 64.2 / **65.7** | 63.4 / 60.7 / −120.1 | **64.4 / 63.5 / 65.9** | 64.0 / 62.7 / −119.7 |
| 292146 | **47.5** / −137.4 / −137.7 | 44.3 / 46.4 / −139.0 | 46.9 / 49.9 / −137.4 | 46.1 / 49.2 / −137.6 |

Stride 1 removes three of the four boundary failures (292146: the 100 s arrival reaches the deadline
at 168 s in truth, the model captures just before it — a residual model error); stride 2 is not
enough. This is the "no wrong model" rule applied to our own counterfactual.

## V7 (`env/t1_b_feasible_v7.py`, `train/t1_b_v7_sac.py`, `--interface v7`)

- Counterfactual branches run the real MPC (QP every control) on the nominal model to the episode
  end; everything else of the V6 learner (intent replay, real anchor, pair-mean MPC advantage,
  gate) is unchanged.
- A stride-1 branch costs about one real episode, so at the first decision the environment emits a
  snapshot of the real state and the training process runs the branches in a 6-process pool while
  the episode continues; the payloads are attached at the episode's terminal transition (waiting
  if needed): deterministic. `snapshot_equivalence_check.py`: snapshot → pickle → new predictor
  reproduces the in-process rollout bitwise (q and rewards difference 0.0).

Smoke (E-TCPF, 500 steps, smoke hyper-parameters, 4 envs + 6 workers on 4 sandbox cores):
5/5 first decisions admitted (10 labels); factual model vs real 66.4/64.3, 66.0/65.9,
−116.8/−116.9 (timeout), −133.7/−134.4 (timeout), −130.5/−130.5 (exhaustion). E.g. chosen 140 s
→ real timeout while the model predicts 100 s → success (+61). Intent real-TD gradient 20.9,
MPC-advantage gradient 2.9. Wall 1161 s for 500 steps (oversubscribed sandbox).
