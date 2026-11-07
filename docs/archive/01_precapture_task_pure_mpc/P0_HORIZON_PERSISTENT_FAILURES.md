# P0: tested-horizon persistent failures (2026-09-10)

Baseline `3e5694c`; T0 parent `ca42c3a`. Full-state / perception=None / precapture_planning, fixed reference, analytic_local, frozen attitude reference, one episode per seed and horizon. No MPC, task, reward or solver changes.

## Completion grid

| Seed | h20 | h30 | h35 | h50 |
|---|---|---|---|---|
| 262000 | completed | completed | completed | completed |
| 262001 | **failed** | completed | completed | completed |
| 262002 | completed | completed | completed | completed |
| 262003 | completed | completed | completed | completed |
| 262004 | completed | completed | completed | completed |
| 262005 | **failed** | **failed** | **failed** | **failed** |
| 262006 | **failed** | **failed** | **failed** | **failed** |
| 262007 | completed | completed | completed | completed |
| 262008 | completed | completed | completed | completed |
| 262009 | completed | completed | completed | completed |
| 262010 | completed | completed | completed | completed |
| 262011 | **failed** | **failed** | **failed** | completed |

Four-horizon intersection F = [262005, 262006]; |F| = 2.
**Gate: continue T2.**

This only establishes behavior in the tested horizon range; it does not establish failure at every possible horizon or physical unreachability.

## Pure MPC descriptive rows

Time and impulse means below are conditional on completion. Completion rate uses all 12 episodes. For h20/h50 the original pre-transition final_time_s is converted to actual elapsed time as steps × 0.1 s; impulse is sum(force_norm_n × 0.1 s). Their original traces do not record all truth margins, so that entry is unavailable rather than inferred.

| Horizon | Completed | Mean completion time s | Mean force impulse N s | Worst normalized truth margin, successful episodes |
|---|---|---|---|---|
| h20 | 8/12 | 128.475 | 208.181 | unavailable in historical traces |
| h30 | 9/12 | 97.444 | 172.150 | 0.0198697 |
| h35 | 9/12 | 88.478 | 167.187 | 0.0196038 |
| h50 | 10/12 | 76.680 | 201.008 | unavailable in historical traces |

## Difficulty metrics

Supplemental h20/h50 replays fill missing telemetry; their completion outcomes must match the preserved originals. New terminal values include the final RK45 transition. Utilization is max-axis absolute command divided by the per-axis limit; force norm must not be divided by a single-axis limit.

| Seed | h | End s | Range m | Reason | FOV peak deg | Force mean/peak utilization | Torque mean/peak utilization | Illegal crossings | QP infeasible fraction | Fallback fraction | Truth margin |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 262005 | 20 | 96.1 | 10.626 | fov_violation_steps | 50.130 | 0.568/1.000 | 0.249/1.000 | 5 | 0.179 | 0.179 | -0.00259102 |
| 262005 | 30 | 32.0 | 10.435 | fov_violation_steps | 50.921 | 0.756/1.000 | 0.248/1.000 | 1 | 0.100 | 0.100 | -0.0184293 |
| 262005 | 35 | 30.2 | 10.282 | fov_violation_steps | 50.767 | 0.770/1.000 | 0.247/1.000 | 1 | 0.126 | 0.126 | -0.0153344 |
| 262005 | 50 | 28.1 | 9.455 | fov_violation_steps | 50.577 | 0.868/1.000 | 0.266/1.000 | 1 | 0.121 | 0.121 | -0.011531 |
| 262006 | 20 | 35.0 | 17.227 | fov_violation_steps | 50.077 | 0.191/1.000 | 0.060/1.000 | 1 | 0.809 | 0.809 | -0.00153161 |
| 262006 | 30 | 30.2 | 15.985 | fov_violation_steps | 50.159 | 0.172/1.000 | 0.062/1.000 | 1 | 0.828 | 0.828 | -0.00317926 |
| 262006 | 35 | 29.8 | 15.353 | fov_violation_steps | 50.124 | 0.158/1.000 | 0.059/1.000 | 1 | 0.842 | 0.842 | -0.00248618 |
| 262006 | 50 | 28.5 | 15.929 | fov_violation_steps | 50.090 | 0.112/1.000 | 0.051/1.000 | 1 | 0.888 | 0.888 | -0.0018072 |

## Deployment-equivalent serial compute

The following is the new measured compute reference. One process, no concurrent reachability workers, seed 262000, 300 steps (30 s prefix), diagnostics disabled, controller.command only; includes cold first solve. It is a sampled prefix, not proof of an entire-episode hard real-time bound. Environment RK45 propagation is recorded separately in each JSON. Repeated short runs vary with machine load.

| h | Mean ms | p95 ms | Max ms | p95 / 100 ms | Over-budget fraction |
|---|---|---|---|---|---|
| 20 | 42.024 | 47.525 | 226.140 | 0.475 | 0.0033 |
| 30 | 62.673 | 68.213 | 326.769 | 0.682 | 0.0033 |
| 35 | 72.151 | 77.439 | 371.930 | 0.774 | 0.0067 |
| 50 | 106.202 | 113.589 | 627.033 | 1.136 | 0.8967 |

## Artifacts and boundaries

- Original h20/h50 episodes: `logs/upper_na_f1/na_f1_h{20,50}_*.json` (preserved).
- New h30/h35 episodes, four supplemental difficulty replays, and compute JSON: `logs/p0_horizons_20260910/`.
- Manifest contains the fixed seed grid, settings, commands, timing/parallel separation and gate. Reproduce with `python -B -m experiments.run_p0_horizons` in a fresh output directory; the runner refuses to overwrite this evidence.
- No training, no model loading, no T6/T7 interface change. The previous unsubstantiated h30/h35 and diagnostics-disabled compute numbers remain excluded.
