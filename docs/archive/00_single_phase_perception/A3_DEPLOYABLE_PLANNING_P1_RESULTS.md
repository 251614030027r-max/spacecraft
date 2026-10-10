# A3 P1 Deployable-Planning Results

Status: formal P1, h10/h15 sensitivity and independent serial compute profile complete. No training was performed. The pre-registered P1 gap is not established.

## Decision

The deployable online h20 controller completed 60/60 episodes with zero truth-geometry violations. Its median completion time was 65.05 s and mean completed-episode force impulse was 139.87 N s. The offline estimated-state plan completed 59/60, required 81.10 s and 184.77 N s. The online controller was better in the same direction in every seed block, using 24.30% less force impulse and 19.79% less median time in aggregate. Therefore A3 does not open the B training gate.

The long exact h50 row also completed 60/60, but its quality was essentially identical to online analytic-local h20: 64.85 versus 65.05 s median and 139.52 versus 139.87 N s mean force impulse. Its serial p95 was 207.11 ms rather than 40.25 ms. There is no measured long-horizon quality advantage for SAC guidance to recover.

## Main results

Each main row contains three disjoint 20-episode blocks. Impulse values below use completed episodes; all-episode values are retained in the machine summary.

| Method | Zero-violation completion | Time median / p95 | Force mean | Torque mean | Command mean / p95 / max | >0.1 s |
|---|---:|---:|---:|---:|---:|---:|
| Online endpoint h20, estimate | 60/60 | 65.05 / 76.90 s | 139.87 N s | 3.60 N m s | 34.72 / 40.25 / 268.83 ms | 0.14% |
| Offline episode plan h20, estimate | 59/60 | 81.10 / 99.50 s | 184.77 N s | 2.39 N m s | 45.90 / 160.67 / 479.84 ms | 10.02% |
| Offline episode plan h20, truth | 60/60 | 81.45 / 98.40 s | 158.19 N s | 0.59 N m s | 45.44 / 159.55 / 468.51 ms | 10.06% |
| Online endpoint exact h50, estimate | 60/60 | 64.85 / 75.40 s | 139.52 N s | 3.61 N m s | 94.06 / 207.11 / 759.96 ms | 10.35% |

The one offline-estimate failure was seed 20290019: at 62.7 s it crossed the closing-speed margin by only `-7.89e-5 m/s`; no distance/time failure or other geometry violation occurred. This does not change the conclusion, because the online row remained safer, faster and more efficient.

## Truth margins

| Method | Axial [m] | Lateral [m] | FOV [rad] | Total speed [m/s] | Closing speed [m/s] |
|---|---:|---:|---:|---:|---:|
| Online h20 | 1.6859 | 1.0394 | 0.3242 | 0.07678 | 0.000261 |
| Offline estimate | 1.5490 | 0.0997 | 0.1728 | 0.07516 | -0.000079 |
| Offline truth | 1.6559 | 1.0735 | 0.1741 | 0.14869 | 0.005997 |
| Exact h50 | 1.6874 | 1.0519 | 0.3242 | 0.07731 | 0.000313 |

## Horizon sensitivity

On the registered seed-262000 block, h10, h15 and h20 all completed 20/20 with zero violations and the same 62.65 s median completion time. Mean force impulses were 134.46, 134.31 and 134.29 N s. The available planning quality is therefore insensitive across h10-h20; h20 was not selected to manufacture a weak result.

## Interpretation boundary

The fixed wrong-inertia model did not expose a planning gap. The endpoint-only online reference plus analytic local MPC already lies on the best observed quality/compute frontier. The fixed whole-episode minimum-jerk trajectory is an offline reference, but the measurements refute its prior “efficiency upper bound” label: it is slower and more fuel-intensive than the deployable controller. It must not be used as a target that a learned method is claimed to recover.

Do not start B training from P1. Per the upper instruction, any attempt to create a long-range planning need must return for approval as one new task factor: higher Lambda or longer initial range, never both. Runtime remains an implementation-specific secondary column; no flight-processor claim is made.

Raw JSON and the machine summary remain under `logs/a3_deployable_planning/`. Figures are `docs/a3_deployable_planning/p1_main.png` and `p1_frontier.png`.
