# P2: lower-controller feedback precheck

Selected replay set: 2 successful and 4 failed episodes. Selection rules and analysis thresholds were saved in the manifest before replay. All replay outcomes and end times match their source grid episodes; impulse matches within 1e-7 N s. Missing successful seed/horizon strata remain missing.

## Recording and slack audit

The MPC reads `_slack.value` after solver exceptions without clearing it. A fallback value may therefore be stale or zero. The recording wrapper excludes **all fallback steps** from slack aggregation, reports null when a macro decision contains no valid solves, and records valid-solve count alongside fallback fraction. Unit tests inject stale slack=99 on fallback steps to verify masking.

Per 2 s decision, `info` now contains fallback fraction, valid-solve count, maximum valid-solve slack, and mean/peak per-axis force and torque utilization. Source quantities already exist in the solve/command; no predicted rollout is added. Observation, reward, action defaults and MPC code are unchanged. `predicted_minimum_margin` is neither used nor promoted to an online feedback signal. Truth geometry remains the sole constraint adjudicator.

## Layer 1: whole-episode distributions

Each episode contributes one control-step-weighted mean; micro decisions are not treated as independent samples. Slack excludes null macro decisions. AUC is reported with larger values predicting failure; separation allows either direction. The predeclared descriptive signal threshold is separation ≥0.75.

| Quantity | Success mean | Failure mean | AUC | Separation |
|---|---|---|---|---|
| fallback_fraction | 0.1494 | 0.4889 | 0.75 | 0.75 |
| slack_max | 3.914e-05 | 0.2314 | 1 | 1 |
| force_mean | 0.3634 | 0.4217 | 0.625 | 0.625 |
| force_peak | 0.4784 | 0.4825 | 0.375 | 0.625 |
| torque_mean | 0.1286 | 0.1538 | 0.75 | 0.75 |
| torque_peak | 0.1676 | 0.2163 | 0.75 | 0.75 |

## Layer 2: evolution before termination

Compare bins [-15,-10), [-10,-5), [-5,-2) seconds relative to episode end. A positive trend requires all three bin values to increase; signal requires at least half the failed cases and a higher fraction than successful cases. Tiny numerical differences below 1e-12 are treated as ties. All failed/successful episodes remain in the respective denominator; missing-slack windows do not count as an increasing trend. Raw per-case bin values and evaluable counts are retained in analysis.json.

| Quantity | Failures increasing | Successes increasing | Signal |
|---|---|---|---|
| fallback_fraction | 0.25 | 0 | False |
| slack_max | 0.25 | 0 | False |
| force_mean | 0 | 0 | False |
| force_peak | 0 | 0 | False |
| torque_mean | 0 | 0 | False |
| torque_peak | 0 | 0 | False |

## Layer 3: fixed early-warning rules

Thresholds: fallback fraction ≥0.5, valid slack ≥1.0, force peak ≥0.95, torque peak ≥0.95. Require two consecutive decisions, ignore the first 10 s, and require at least 2 s lead before termination. Successful episodes define false alarms. No threshold fitting was performed.

| Rule | Failure detection | Success false-alarm rate | Median failure lead s | Signal |
|---|---|---|---|---|
| fallback_fraction | 1 | 0.5 | 20.4 | True |
| slack_max | 0.5 | 0 | 11.15 | True |
| force_peak | 0.5 | 1 | 51.15 | False |
| torque_peak | 0.5 | 1 | 26.15 | False |
| any_rule | 1 | 1 | 20.6 | False |

## Conclusion and limits

Three-layer descriptive signals: {'distribution': True, 'time_trend': False, 'early_warning': True}.
At least one check has information: the cheap feedback channel merits a later controlled ablation, subject to upper review.
Only two seeds and selected scripted trajectories, no held-out data. Episode-level descriptive AUC is not a significance test or independent generalization evidence. Action, phase and duration can confound separability. No learned-policy or causal feedback benefit established.
This precheck does not resolve the unrescued 262006 case and does not establish a two-seed rescue claim. Aggregation overhead has not been separately certified for deployment. T6/T7 remain unauthorized; stop here.

## Replay sources

| Artifact | Source | Completed | End s |
|---|---|---|---|
| `case_0_s262005_h20_success.json` | `logs\p1_a_20260910\commit_s262005_h20_t60.json` | True | 293.8 |
| `case_1_s262005_h20_failure.json` | `logs\p1_a_20260910\commit_s262005_h20_t0.json` | False | 96.1 |
| `case_2_s262005_h35_success.json` | `logs\p1_a_20260910\hold_s262005_h35_t30_r12.json` | True | 91.1 |
| `case_3_s262005_h35_failure.json` | `logs\p1_a_20260910\commit_s262005_h35_t0.json` | False | 30.2 |
| `case_4_s262006_h20_failure.json` | `logs\p1_a_20260910\commit_s262006_h20_t0.json` | False | 35.0 |
| `case_5_s262006_h35_failure.json` | `logs\p1_a_20260910\commit_s262006_h35_t0.json` | False | 29.8 |
