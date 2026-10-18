# T1P_ROUND1_PRECHECK — upper time-boxed test of T1-P (2026-10-18)

Instruction: collab `upper/T1_P_FAST_TRACK_CONDITIONAL_GO_20261018.md` (`901d78b`); design
`upper/T1_P_FINAL_REDESIGN_FOR_FEASIBILITY_20261018.md` (`746fb49`). Code: the commit that adds this
directory (candidate, **not frozen**). Test wall: unit tests 75 s + window 14:16:49–14:27:44 UTC
(10.9 min) = **12.2 min ≤ 20 min**. Development openings only.

| Check | Result | File |
|---|---|---|
| Unit contracts (`tests/test_t1_p.py`, 6) | PASS: orthonormal basis, a = 0 is the coast, \|a\| normalised, a_r > 0 inward; local SCP keeps r ≥ 6.05 m and the envelope at 0.1 s with a goal inside the sphere (soft terminal error > 0.5 m, \|F\| ≤ 1.6 N), verifier catches a node-interior violation; PBRS telescopes; pending pair weights; λ_eq must be 0, α = 0.005 × gap; CF leaves the real transitions bitwise unchanged | — |
| Stage actions (opening 292000, 7 actions, truth 12 s) | PASS: all accepted by the local G★ (planned terminal error ≤ 2e-7); truth endpoints pairwise ≥ 0.71 m apart (median 1.01 m); no violation; reference jump ≤ 0.07 m; repeat bitwise equal. **Finding:** truth end vs stage goal 0.43–0.65 m and 0.054–0.116 m/s for every action, including the coast (a = 0), with a common velocity error ≈ (0, −0.08, 0) m/s ECI — MPC execution error of the same size as the stage authority (d_max 0.761 m, 0.127 m/s) | `stages_292000.json` |
| Full task chain, pre-declared geometric rule (292000; then 292001, 292002 pre-numbered) | **FAIL ×3**: 292000 timeout (range 12.8 → 28 m, radial speed stays ≈ +0.11 m/s for ≥ 50 s although every review asks inward), 292001 unsafe (distance: reached 10 m at 42 s, then out to 30 m), 292002 timeout (16–23 m). The bridge never triggered; no capture. Stage deviations > 1 m and rejected stages (hold) recur | `pipeline_2920xx.json` |
| Stage pair (292002, first sampled review) | PASS: factual 12 s MPC prediction vs real stage e_q 0.010 m, e_v 0.0013 m/s, e_R 0.063 → w = 1; alternative's prediction vs its truth branch (diagnostic only) 0.020; predicted successor observation finite, due = 1 | `pair_292002.json` |
| TCPF smoke (160 steps) | PASS (interface): 4 pairs stored, 4 weighted at stage close, CF gradient non-zero (ratio to TD 1e-4–2e-3), intervals 1/5/6 steps closed, 0 dropped. 52 of 70 intervals have length 1 (hold reviews every 2 s) | `smoke_tcpf.json` |
| Throughput | UNTESTED (left to the lower) | — |

Verdict: the conditional GO's fact conditions are **not met** (no pre-declared opening completes the
chain; a persistent structural failure in the far field). Not frozen; no training.
