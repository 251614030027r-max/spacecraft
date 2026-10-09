# A2 Guidance-Free Results

Status: formal trajectory evaluation and independent serial compute profile complete. No training was performed.

## Decision

A2 closes the control/reachability question but does not open a learning gate. The explicit episode endpoint plan plus horizon-20 MPC tracking completed 60/60 episodes with zero truth-geometry violations, matching the oracle's 60/60 and completion time (82.58 s estimated state versus 82.49 s oracle). It also improved on the accepted A1 guided control (90.77 s) while using less mean force impulse (184.82 versus 216.24 N s). Therefore removal of `corridor_guidance_velocity` does not make the task MPC-hard when a competent non-learning planner is supplied.

The strict 0.1 s p95 compute gate is not met. Planning+tracking averages 52.17 ms per command but has p95 163.02 ms and 10.0% of commands above 0.1 s. The periodic exact RK45 linearization refresh has p95 123.14 ms and is the dominant tail; ordinary QP and rollout p95 are 11.82 ms and 9.28 ms. This is a prototype Python/CVXPY/CLARABEL profile on the development CPU, not evidence of flight-processor feasibility.

## Formal table

Each row contains 60 episodes from the disjoint 262000, 20288000 and 20290000 seed blocks. Every successful row was 20/20 in every block with zero QP fallback and zero predicted-safe/truth-unsafe events.

| Method | Complete and zero violation | Mean completion time | Mean force impulse | Serial command mean / p95 | >0.1 s |
|---|---:|---:|---:|---:|---:|
| A1 guided estimate control, h50 | 60/60 | 90.77 s | 216.24 N s | 109.91 / 218.69 ms | 18.85% |
| Fixed terminal, h10 | 60/60 | 164.13 s | 230.74 N s | 31.11 / 143.61 ms | 10.01% |
| Receding endpoint plan, h50 | 3/60 | 187.00 s among 3 | 360.07 N s all episodes | 111.05 / 222.26 ms | 22.60% |
| Episode plan + tracking, h20 | 60/60 | 82.58 s | 184.82 N s | 52.17 / 163.02 ms | 10.00% |
| Same planner/tracker, oracle | 60/60 | 82.49 s | 158.18 N s | 53.89 / 167.66 ms | 10.06% |

The receding-plan row timed out in 55 episodes and violated the corridor in 2; it continually resets a rest-to-rest reference at the current state, making near-term progress too weak. Because the fixed-terminal and explicit-plan rows both solve the same task, this failure is controller-reference design evidence, not task infeasibility or a learning gap.

## Perception and energy

All four successful methods retained 5/5 visible features at every recorded step and had zero no-measurement intervals. For the matched planning/tracking pair, worst-axis position covariance p95 was 0.20361 m under estimated-state control and 0.19987 m under oracle (+1.87%); velocity covariance p95 was 0.03436 and 0.03434 m/s (+0.06%). The successful estimated-state trajectory therefore did not create a material visibility or covariance penalty. Correlation between FOV angle and covariance exists, but with full feature visibility and near-identical matched covariance it is geometric variation, not a demonstrated action-related observability bottleneck.

Estimated-state planning used 16.84% more mean force impulse than oracle and 306.35% more torque impulse, while completion time changed by only 0.09 s and safety was unchanged. Perception uncertainty currently manifests as an energy/attitude-control cost, not as a success or safety gap.

## Boundary for the next decision

- Do not start SAC or hierarchical SAC-MPC training from A2: a strong classical planner closes the task and current sensing never becomes feature-limited.
- A compute-only continuation may isolate or replace the periodic exact refresh, but it must remain separate from claims about active perception and must be re-profiled serially.
- If the paper still requires active perception, the upper level must explicitly authorize one physically motivated sensing-hard factor in a later stage; A2 itself must remain frozen and must not tune camera/EKF noise after seeing these results.

Raw evidence and machine-readable aggregate remain under `logs/a2_guidance_free/`; the tracked figures are `docs/a2_guidance_free/a2_main.png` and `docs/a2_guidance_free/a2_compute.png`.
