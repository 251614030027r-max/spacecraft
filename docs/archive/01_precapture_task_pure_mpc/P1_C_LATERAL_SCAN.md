# P1-C: lateral-adjustment grid

Seed 262006 remained unrescued after the complete timing/radius and approach-rate grids. At horizons 20 and 35, test ±10/20/30 degrees in two orthogonal tangent directions plus a zero-angle control (26 cells). Full-state, no training, no MPC/solver/task changes.

At each decision, take the existing radial_local desired-pose waypoint and rotate its direction while preserving its radius. The two tangent directions use the least-aligned Cartesian axis and its orthogonal cross product. The added angle fades as min(1, distance-to-goal / goal-radius), preserving the original final goal. Requested angles still pass through existing action limits; no clipping bounds were changed. The zero-angle control is exactly the existing radial_local desired-pose policy, so any success of that control could not be credited to lateral adjustment.

| h | Axis | Requested angle deg | Completed | End s | Force impulse N s | Worst normalized truth margin | Reason | Artifact |
|---|---|---|---|---|---|---|---|---|
| 20 | 0 | 0 | False | 44.0 | 60.111 | 0.0730752 | distance_failure | `lateral_s262006_h20_axis0_angle0.json` |
| 20 | 0 | -30 | False | 45.0 | 58.604 | -0.000585413 | fov_failure | `lateral_s262006_h20_axis0_angle-30.json` |
| 20 | 0 | -20 | False | 44.9 | 58.159 | -0.00124763 | fov_failure | `lateral_s262006_h20_axis0_angle-20.json` |
| 20 | 0 | -10 | False | 47.0 | 65.736 | -0.0020041 | fov_failure | `lateral_s262006_h20_axis0_angle-10.json` |
| 20 | 0 | 10 | False | 43.8 | 59.410 | -0.00103809 | fov_failure | `lateral_s262006_h20_axis0_angle10.json` |
| 20 | 0 | 20 | False | 41.1 | 58.357 | -0.00208064 | fov_failure | `lateral_s262006_h20_axis0_angle20.json` |
| 20 | 0 | 30 | False | 27.9 | 47.166 | -0.00130798 | fov_failure | `lateral_s262006_h20_axis0_angle30.json` |
| 20 | 1 | -30 | False | 38.7 | 59.785 | -0.00224245 | fov_failure | `lateral_s262006_h20_axis1_angle-30.json` |
| 20 | 1 | -20 | False | 38.7 | 59.865 | -0.0020575 | fov_failure | `lateral_s262006_h20_axis1_angle-20.json` |
| 20 | 1 | -10 | False | 43.8 | 60.117 | 0.0290935 | distance_failure | `lateral_s262006_h20_axis1_angle-10.json` |
| 20 | 1 | 10 | False | 34.9 | 54.972 | -0.00232433 | fov_failure | `lateral_s262006_h20_axis1_angle10.json` |
| 20 | 1 | 20 | False | 33.2 | 53.027 | -0.00148406 | fov_failure | `lateral_s262006_h20_axis1_angle20.json` |
| 20 | 1 | 30 | False | 33.0 | 52.442 | -0.00104558 | fov_failure | `lateral_s262006_h20_axis1_angle30.json` |
| 35 | 0 | 0 | False | 39.0 | 49.363 | -0.00208822 | fov_failure | `lateral_s262006_h35_axis0_angle0.json` |
| 35 | 0 | -30 | False | 37.4 | 44.268 | -0.000182562 | fov_failure | `lateral_s262006_h35_axis0_angle-30.json` |
| 35 | 0 | -20 | False | 35.4 | 43.208 | -0.000241988 | fov_failure | `lateral_s262006_h35_axis0_angle-20.json` |
| 35 | 0 | -10 | False | 38.1 | 43.336 | -0.00133005 | fov_failure | `lateral_s262006_h35_axis0_angle-10.json` |
| 35 | 0 | 10 | False | 39.0 | 49.363 | -0.00208857 | fov_failure | `lateral_s262006_h35_axis0_angle10.json` |
| 35 | 0 | 20 | False | 35.2 | 43.670 | -0.00168614 | fov_failure | `lateral_s262006_h35_axis0_angle20.json` |
| 35 | 0 | 30 | False | 25.5 | 31.176 | -0.00312935 | fov_failure | `lateral_s262006_h35_axis0_angle30.json` |
| 35 | 1 | -30 | False | 37.4 | 46.765 | -0.000635685 | fov_failure | `lateral_s262006_h35_axis1_angle-30.json` |
| 35 | 1 | -20 | False | 37.4 | 46.765 | -0.000638742 | fov_failure | `lateral_s262006_h35_axis1_angle-20.json` |
| 35 | 1 | -10 | False | 38.4 | 48.497 | -0.000535857 | fov_failure | `lateral_s262006_h35_axis1_angle-10.json` |
| 35 | 1 | 10 | False | 43.9 | 50.561 | -0.00212175 | fov_failure | `lateral_s262006_h35_axis1_angle10.json` |
| 35 | 1 | 20 | False | 31.9 | 39.836 | -0.00187488 | fov_failure | `lateral_s262006_h35_axis1_angle20.json` |
| 35 | 1 | 30 | False | 31.0 | 38.681 | -0.00112964 | fov_failure | `lateral_s262006_h35_axis1_angle30.json` |

Rescued in this stage: []. Total seeds rescued across P1-A/B/C: [262005]. Still unrescued: [262006].
Continue T5 because P1 did rescue one seed. The two-seed difficulty set has **not** been fully rescued. Failure of this finite scripted lateral family does not prove physical unreachability or failure of every possible approach policy.
Engineering acceleration and four-process reachability execution match T3; no parallel timing is used as a compute result. Unit validation: zero-angle action exactly equals the baseline and lateral perturbations preserve the radial action.
