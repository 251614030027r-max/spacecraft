# Precapture planning task specification

Status: task, environment, MPC semantics, metrics and evaluator wiring implemented. No
baseline diagnosis or training result is claimed by this document.

## Research variable restored

The chaser starts 15--20 m from the target centre, outside the terminal corridor and
without co-rotation initialization. The controller must choose when to synchronize and
when to enter the rotating terminal corridor; reset, reward and the Pure-MPC reference do
not provide an intermediate path.

## Frozen V0.1 defaults

- Control period 0.1 s; episode limit 300 s; distance failure at 30 m.
- Keepout radius 2 m; terminal latch radius 8 m; target-frame speed transition band
  14--8 m.
- Outside the latch, inertial COM-relative speed is limited to 0.50 m/s. Target-frame
  speed is unconstrained beyond 14 m and contracts linearly from 1.10 m/s at 14 m to
  0.35 m/s at 8 m.
- After latch, corridor, FOV, 0.35 m/s target-frame total-speed and the existing
  range-dependent closing-speed law remain active even if a prediction retreats beyond
  8 m.
- Completion remains 0.25 m position, 10 deg attitude, 0.05 m/s target-frame speed,
  0.02 rad/s relative angular speed, held for 1 s.
- The diagnostic full-state observation is 24D. It is not the final learned-policy
  observation.
- Reward discount is 0.999 at 10 Hz. Potential shaping uses all four terminal residual
  groups normalized by their completion tolerances; it contains no waypoint, phase bonus,
  corridor guidance velocity or position-only progress term.

## MPC contract

Pure MPC and the future hybrid share model, horizon, solver, constraints, limits and
terminal cost. The new QP has a fixed row count; inactive rows have zero Jacobian and a
large positive margin. Before latch, each nominal horizon state independently activates
terminal rows when its predicted range crosses 8 m. After latch, all horizon indices use
terminal rows. `MPCController.command()` therefore requires an explicit
`terminal_latched` argument for this task.

The future SAC action is a 3D waypoint in a target-centred, inertially oriented frame.
Holding that waypoint fixed means inertial holding, not implicit co-rotation. No learned
controller is implemented or trained yet.

## Reproducible checks

```powershell
python -B -m eval.validate_precapture_semantics
python -B -m pytest -q
```

The baseline diagnostic is the next gate and must use the unified evaluator with
`h in {1, 3, 10, 50, 200}`; h1/h3 are diagnostic rows only, not the main Pure-MPC
baseline. Long training remains prohibited until that one-shot diagnosis establishes the
planning-timescale structure.
