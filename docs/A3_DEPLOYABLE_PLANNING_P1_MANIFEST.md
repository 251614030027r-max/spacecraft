# A3 Deployable Planning P1 Pre-registration

Status: pre-registration executed and P1 complete. The gap criterion did not pass; see `A3_DEPLOYABLE_PLANNING_P1_RESULTS.md`. No training was performed.

## Frozen task-side factor

A3 derives directly from the frozen A2 guidance-free perception task. Truth dynamics retain the nominal 225 kg target and nominal inertia matrix. MPC target prediction and EKF mean propagation share one deterministic estimated model generated once by `sample_target_parameters(mismatch=0.20, seed=260903)`:

- estimated mass: 261.441989193512 kg;
- estimated inertia [kg m2]: `[[39.2794880557, 0.1094284335, -0.2394987568], [0.1094284335, 40.5839200744, -0.6314899093], [-0.2394987568, -0.6314899093, 37.8756816355]]`.

Camera, pixel noise, EKF covariance settings, 29D schema, S1-v2 distribution, endpoint, corridor/FOV/speed constraints, continuous wrench limits, truth RK45 dynamics and reward remain unchanged. The model mismatch seed and 20% scale may not be changed after observing results.

## Controller rows prepared for P1

1. `deployable_online`: current estimated state to terminal endpoint exponential reference, finite-horizon MPC, analytic local kinematic linearization and periodic drift correction using the fixed mismatched target model.
2. `offline_estimate_bound`: A2 episode-level plan plus h20 exact tracking, estimated state and the same mismatched target model; efficiency upper bound only.
3. `offline_truth_bound`: identical offline plan/tracker and mismatched prediction model, but truth relative state; pure state-information upper bound.
4. `long_exact_h50`: online endpoint reference and h50 exact prediction, estimated state and mismatched model; non-real-time quality control.

Post-solve diagnostic rollouts are disabled in all A3 controller rows because they do not affect the action or in-QP constraints; truth margins are scored externally by the evaluator. Raw evaluation retains completion/time, translational and rotational impulse, serial command time and five truth margins.

## Horizon calibration and revised compute boundary

The upper review withdrew the `0.1 s / flight slowdown factor` rule because it confounded Python/CVXPY framework cost with planning depth. The revised reference is the 0.1 s development-machine control period; absolute runtime is a secondary implementation-specific column and never gates the scientific conclusion. One process and one thread were used; each candidate received 100 commands on seed 262000. Central-difference local linearization was replaced by a clean analytic local affine model that exactly interpolates the nonlinear local predictor at the operating point.

| Horizon | Analytic-local mean | Analytic-local p95 | Maximum | Budget result |
|---:|---:|---:|---:|---|
| 10 | 22.43 ms | 29.86 ms | 136.12 ms | p95 within 0.1 s |
| 15 | 33.66 ms | 45.31 ms | 191.56 ms | p95 within 0.1 s |
| 20 | 43.45 ms | 58.43 ms | 231.65 ms | p95 within 0.1 s; frozen maximum candidate |

H20 is therefore the strongest and largest pre-registered candidate within the revised p95 reference; h10 and h15 are retained as one-block sensitivity rows. Flight slowdown is discussion-only caveat. No deployment-solver/code-generation stage is part of A3.

## Formal protocol and decision rule

Formal P1 uses three disjoint blocks 262000, 20288000 and 20290000, 20 episodes per method: deployable online h20 estimated-state MPC, offline-plan estimated-state efficiency bound, matched offline truth-state bound, and online endpoint exact h50 estimated-state control. H10 and h15 each run only block 262000 as sensitivity. Trajectory batches may run in parallel; compute evidence is regenerated separately in one process.

The P1 gap is established only if the h20 online row is worse than the offline estimate bound in the same direction in all three blocks and by at least 5% in aggregate mean completed-episode force impulse or median completion time, or if it loses completion/true-geometry safety, while the offline bound and h50 control remain high-quality. Runtime does not establish the gap. If h20 is safe, fast and fuel-efficient near the offline bound, no training starts and any Lambda/distance change returns to upper review. If the offline bound fails, treat reachability first.

Only the four mainstream outputs are decision-facing: completion/time, translational and rotational impulse, serial command timing, and five truth margins. No SAC training, S1-v2 change, camera/EKF tuning, deployment solver, or Lambda/distance increase is authorized in P1.

Calibration JSON is local under `logs/a3_deployable_planning/horizon_profile_analytic/` and is ignored by Git.
