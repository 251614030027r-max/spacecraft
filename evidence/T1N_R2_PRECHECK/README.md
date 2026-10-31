# T1N_R2_PRECHECK — upper technical review of Research Plan R2 (2026-10-18)

Code: science `74d1f7f` (T1-N implementation, not frozen). Sandbox (4 cores), development openings
292000–293999 only (registered in `docs/current/TASK.md` §5); the formal block was not accessed.
Nothing here is a performance gate (R2 §9.3); all numbers are development / exploratory.
Review: collab `docs/collaboration/reviews/R2_UPPER_FEASIBILITY_REVIEW_20261018.md`.

| File | Command | What it shows |
|---|---|---|
| `openings.json` | `t1n_check openings --first 292000 --count 2000 --band-count 200` | 2000 initial states: **0 illegal at t = 0**; radial-envelope margin ≥ 0.217 m/s, FOV margin ≥ 35.0°; spin 10–30° off body z gives ω–H 1.9–10.1° and a port-latitude swing of 7–39° over 240 s (median 19°); tangential angular rate / spin 0.04–0.43 (median 0.19) |
| `feasibility.json` | `t1n_check feasibility --first 292000 --count 48` | Planner response at t = 0 on τ = 20…200 s (5 s grid): feasible fraction 0.68 (0.49–0.86); first feasible median 80 s (p05 55, p95 108); one feasible interval in 46/48; planned transfer Δv has an interior minimum in 43/48, median +70 s (10–155 s) after the first feasible time; 22 ms per SCP solve |
| `closed_table.json`, `closed_episodes.tar.gz` | `t1n_check closed --first 292000 --count 12 --policies earliest:0,earliest:30,earliest:60,fixed:0,random:1,reject,oldgstar` | Truth closed loop, 12 openings × 7 rows (per-episode JSON in the tar). Returns (λ_f = 20): oldgstar 62.43 (12/12), earliest:0 63.12 (12/12), earliest:30 62.05, earliest:60 28.10 (2 timeouts), **fixed:0 65.96 (12/12)**, random:1 19.62 (10/1/1), reject −146.48 (0 success, **0 unsafe**, 12 timeouts) |
| `signal_summary.json`, `signal_raw.tar.gz` | `t1n_check signal --first 292100 --count 6 --reviews 2` | 12 review states (6 at t = 0 request shifts, 6 at t = 6 s replan-vs-keep): paired rolling MPC prediction vs truth branches; error of the paired h-step difference median 0.008 / 0.021 / 0.024 (h = 1/2/3) against truth differences of median 0.11 / 0.31 / 0.45; sign agreement 12/12 at every h; the h-step difference is a median 3 % / 7 % / 9 % of the long-run (to the end) truth difference (median 4.18); truth branches with fresh solvers reproduce the real continuation to ≤ 8e-9; shadow and truth branch cost the same (1.78 s per 2 s step) |
| `plan_vs_long.json` | inline script of the review (same 12 states) | Sign of the 6 s paired reward difference vs the long-run truth difference: **2/11** agree (corr −0.06); sign of the G★ plan-level cost difference (−λ_f · planned transfer Δv − 0.05 · time to arrival): **9/11** (corr 0.24) |
| `timing.json` | `t1n_check timing --opening 292001 --steps 24` | Serial wall per real 2 s step (random actions, 10 reviews in 24 steps): plain 1.78 s; paired shadow h = 1 / 2 / 3: ×1.88 / ×2.53 / ×3.20 |
| `smoke_tcpf.json` | `train.train_t1_n --arm tcpf --steps 90` (16×16, learning_starts 20) | Every term runs end to end; 29 intervals closed (length 3), 0 dropped; paired samples 25–33 % of a batch; gradient ratio probes 0.005–0.014 |

Unit tests: `tests/test_t1_n.py` (7) and the T1 suites (133 passed in total, old behaviour unchanged).
