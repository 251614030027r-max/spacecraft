# A2 Guidance-Free Pre-registration

Status: implementation frozen; formal results not yet available. This stage performs no training.

## Single factor and inherited stack

`perception_guidance_free` is derived directly from the accepted A1/G0 perception environment. Camera geometry/noise, EKF process and measurement noise, 29D schema and normalization, S1-v2 initial distribution, endpoints, truth constraints, continuous wrench limits, RK45 truth dynamics and target tumble are unchanged. A1 already exposes the estimated relative velocity itself in the 29D channel, so A2 preserves that field rather than manufacturing a second schema change. A2 changes only: terminal reward velocity error is measured relative to zero terminal velocity; A2 MPC rows use only the current state and terminal endpoint to construct their reference, never `corridor_guidance_velocity`.

Truth remains restricted to dynamics, reward, termination and scoring. All realistic A2 rows use `observed_relative`; only `planning_tracking_oracle` uses truth. The G0 truth-access sentinel remains a required test. Scripted controllers are excluded.

## Frozen rows

All MPC rows share state/input scales, state/input/terminal weights, CLARABEL tolerances, constraints, actuator bounds, local prediction model, RK45 reference model, one outer iteration, exact linearization and refresh interval 10. Deliberate method differences are:

| Row | Reference | Horizon | Controller state |
|---|---|---:|---|
| `guided_a1_control` | accepted corridor guidance | 50 | estimate |
| `terminal_short` | fixed terminal state | 10 | estimate |
| `receding_plan_long` | regenerated minimum-jerk endpoint plan | 50 | estimate |
| `planning_tracking` | episode-level minimum-jerk plan | 20 | estimate |
| `planning_tracking_oracle` | same episode plan and tracker | 20 | truth |

The two plans use no corridor law or hand-coded closing schedule. Duration is computed from endpoint distance with a frozen peak-speed cap of 0.18 m/s and a 10 s minimum. This is a generic endpoint trajectory, not privileged guidance.

## Formal protocol and metrics

Use disjoint blocks 262000, 20288000 and 20290000, 20 episodes per method per block. Main table: joint completion, truth-geometry zero-violation completion, completion time, discounted return, fuel impulses, command time and environment time. Error table: position, attitude, translational velocity and angular velocity. Perception table: visible feature count/time, longest no-measurement interval, target-axis position/velocity error and `sqrt(diag(P))`, worst-axis error/std, and correlations between FOV angle, visibility and worst-axis covariance. Energy reporting remains mandatory because G0 estimate control used about 26% more force impulse than oracle.

Trajectory-run timing is explicitly invalid for compute claims when three seed blocks run in parallel. The separate compute profile must run alone in one terminal/process with no other A2 process. Report command mean/p95/max and fraction above 0.1 s, plus QP, model-linearization, rollout and exact-refresh components. Record Python/CVXPY/CLARABEL versions and CPU; spacecraft-class processors may be one to two orders slower.

## Frozen interpretation branches

- Safe, real-time, near-oracle strong non-learning control with no action-related visibility/covariance change: no demonstrated learning gap.
- Safe and real-time control plus method-dependent observability: active perception is motivated; A3 may proceed without claiming success yet.
- Short endpoint MPC fails but planning baselines succeed, with only compute tails problematic: weak implementation/reference is ruled out; isolate compute engineering.
- Oracle also fails: treat reachability/task feasibility as the blocker; do not train around it.

No camera/EKF noise tuning, SAC training, scripted controller, new window/obstacle, discrete thruster or online identification is authorized in A2.
