# P1-B: approach-rate scan

Only the seed(s) not rescued by P1-A are evaluated. The fixed grid is radial steps {0.1,0.2,0.4,0.8,1.6} m per 2 s decision × starts {0,60,120} s × horizons {20,35}. Full-state, no training, no MPC/task/reward changes.

Before the start, hold the existing frozen inertial point. At the start, initialize reference radius from the current range and decrease it each decision along the desired-position ray, stopping at the desired position. Nominal radial speed is 0.05–0.8 m/s; the initial direction change and existing waypoint clipping are not bounded by that nominal radial speed. Moving waypoints are a diagnostic device, **not a final learned-interface decision**.

Engineering acceleration uses existing runtime_diagnostics=False and cache_target_trajectory=False switches. A full 262006/h20/commit_at(0) replay matches P1-A completion, end time and fallback count, with impulse difference recorded in the manifest. On-demand propagation retains identical RK45 parameters and timestamps. Parallel runtimes have no real-time interpretation.

| Seed | h | Start s | Step m | Completed | End s | Force impulse N s | Worst normalized truth margin | Reason | Artifact |
|---|---|---|---|---|---|---|---|---|---|
| 262006 | 20 | 0 | 0.1 | False | 35.2 | 43.169 | -0.00240277 | fov_failure | `ramp_s262006_h20_t0_dr0.1.json` |
| 262006 | 20 | 0 | 0.2 | False | 33.3 | 43.865 | -0.00305063 | fov_failure | `ramp_s262006_h20_t0_dr0.2.json` |
| 262006 | 20 | 0 | 0.4 | False | 29.9 | 46.573 | -0.00222734 | fov_failure | `ramp_s262006_h20_t0_dr0.4.json` |
| 262006 | 20 | 0 | 0.8 | False | 27.4 | 48.543 | -0.00183866 | fov_failure | `ramp_s262006_h20_t0_dr0.8.json` |
| 262006 | 20 | 0 | 1.6 | False | 27.2 | 47.181 | -0.00391679 | fov_failure | `ramp_s262006_h20_t0_dr1.6.json` |
| 262006 | 35 | 0 | 0.1 | False | 33.9 | 39.113 | -0.00134779 | fov_failure | `ramp_s262006_h35_t0_dr0.1.json` |
| 262006 | 35 | 0 | 0.2 | False | 33.0 | 39.466 | -0.00159749 | fov_failure | `ramp_s262006_h35_t0_dr0.2.json` |
| 262006 | 35 | 0 | 0.4 | False | 32.0 | 38.826 | -0.00132645 | fov_failure | `ramp_s262006_h35_t0_dr0.4.json` |
| 262006 | 35 | 0 | 0.8 | False | 31.1 | 35.979 | -0.000311643 | fov_failure | `ramp_s262006_h35_t0_dr0.8.json` |
| 262006 | 35 | 0 | 1.6 | False | 29.9 | 33.576 | -0.000131829 | fov_failure | `ramp_s262006_h35_t0_dr1.6.json` |
| 262006 | 20 | 60 | 0.1 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t60_dr0.1.json` |
| 262006 | 20 | 60 | 0.2 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t60_dr0.2.json` |
| 262006 | 20 | 60 | 0.4 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t60_dr0.4.json` |
| 262006 | 20 | 60 | 0.8 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t60_dr0.8.json` |
| 262006 | 20 | 60 | 1.6 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t60_dr1.6.json` |
| 262006 | 35 | 60 | 0.1 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t60_dr0.1.json` |
| 262006 | 35 | 60 | 0.2 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t60_dr0.2.json` |
| 262006 | 35 | 60 | 0.4 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t60_dr0.4.json` |
| 262006 | 35 | 60 | 0.8 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t60_dr0.8.json` |
| 262006 | 35 | 60 | 1.6 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t60_dr1.6.json` |
| 262006 | 20 | 120 | 0.1 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t120_dr0.1.json` |
| 262006 | 20 | 120 | 0.2 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t120_dr0.2.json` |
| 262006 | 20 | 120 | 0.4 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t120_dr0.4.json` |
| 262006 | 20 | 120 | 0.8 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t120_dr0.8.json` |
| 262006 | 20 | 120 | 1.6 | False | 43.0 | 36.655 | -0.0011066 | fov_failure | `ramp_s262006_h20_t120_dr1.6.json` |
| 262006 | 35 | 120 | 0.1 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t120_dr0.1.json` |
| 262006 | 35 | 120 | 0.2 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t120_dr0.2.json` |
| 262006 | 35 | 120 | 0.4 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t120_dr0.4.json` |
| 262006 | 35 | 120 | 0.8 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t120_dr0.8.json` |
| 262006 | 35 | 120 | 1.6 | False | 47.2 | 42.127 | -0.00109654 | fov_failure | `ramp_s262006_h35_t120_dr1.6.json` |

Rescued by this stage: []. Still unrescued: [262006].
Gate: continue T4 on the remaining seeds.
This is a post-hoc action-space ceiling on the declared finite grid. It does not establish learnability, global infeasibility, or benefits from any trained coupling.
