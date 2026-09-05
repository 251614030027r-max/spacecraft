# Precapture planning task specification

Status: task, environment, MPC semantics, metrics and evaluator wiring implemented. No
baseline diagnosis or training result is claimed by this document.

## Research variable restored

The chaser starts 15--20 m from the target centre, outside the terminal corridor and
without co-rotation initialization. The controller must choose when to synchronize and
when to enter the rotating terminal corridor; reset, reward and the Pure-MPC reference do
not provide an intermediate path.

## Target tumble rate: the number the whole task rests on

The target tumbles at **0.041231 rad/s** -- `phase2_target_tumble_scale = 0.20` times
`|TARGET_BASE_TUMBLE_RAD_S| = 0.20616`. This is the frozen S1-v2 rate every other number
in this project is measured at, and the one the reference-paper comparison rests on
(`References/北航.pdf` uses 0.0173 rad/s).

Until 2026-09-04 the task did not actually run at it. `_select_phase2_stage` returns
early for the precapture task, so the episode kept the constructor's
`target_tumble_scale` (0.5) and ran at **0.10308 rad/s, 2.5x too fast**, while
`precapture_planning_environment_config` asked for 0.20. The line was dead.

That is not a cosmetic difference. Sustained co-rotation at radius `r` costs
`m * omega^2 * r`:

| omega | cost per metre | 5 N per-axis runs out at | 8.66 N body diagonal runs out at |
|---|---|---|---|
| 0.10308 rad/s (what ran) | 1.126 N/m | **4.4 m** | **7.7 m** |
| 0.041231 rad/s (intended) | 0.180 N/m | 27.8 m | 48.1 m |

So at the rate that was running, no controller of any kind could hold the outer region --
the task was actuator-infeasible past 7.7 m, and the 8--9 m stall the first baseline
batch reported was measuring the thruster limit, not a planning defect. Every
`logs/precapture_planning_baseline/*.json` predates the fix and is void.
`tests/test_precapture_task.py::test_precapture_episode_runs_at_the_configured_tumble_rate`
pins the rate so this cannot regress silently.

## Frozen V0.2 defaults

- Control period 0.1 s; episode limit 300 s; distance failure at 30 m.
- Target tumble 0.041231 rad/s; chaser 106 kg; +-5 N and +-0.6 N*m per axis.
- Keepout radius 2 m; terminal latch radius **6 m** (was 8 m).
- Outside the latch two limits apply, both on the *inertial* COM-relative velocity:
  total speed <= **1.20 m/s**, and inward radial speed <=
  `sqrt(2 * 0.020 * (r - 2))` -- a braking envelope that is dynamically self-consistent
  (following it needs a constant 0.020 m/s^2, 42% of the guaranteed single-axis
  authority) and that does not constrain the tangential component, so when to begin
  co-rotating stays a free decision. It meets the terminal closing-speed limit at the
  latch (0.400 vs 0.20 m/s at 6 m).
- The 14--8 m target-frame speed transition band is **withdrawn**. It was dynamically
  infeasible: sailing along it needed 0.075--0.1375 m/s^2 against a 0.0472 m/s^2
  guarantee. Target-frame speed is unconstrained until the latch.
- After latch, corridor, FOV, 0.35 m/s target-frame total-speed and the existing
  range-dependent closing-speed law remain active even if a prediction retreats beyond
  6 m.
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
terminal rows when its predicted range crosses 6 m. After latch, all horizon indices use
terminal rows. `MPCController.command()` therefore requires an explicit
`terminal_latched` argument for this task.

The future SAC action is a 3D waypoint in a target-centred, inertially oriented frame.
Holding that waypoint fixed means inertial holding, not implicit co-rotation. No learned
controller is implemented or trained yet.

That waypoint reaches the QP through `reference_source="external_local"`, which must be
written in the *state's* coordinates: rows 3:6 are the exponential translation
`rho = J_l(phi)^-1 p`, not the position, and rows 9:12 are the relative velocity in the
**chaser body** frame, not the target-frame position rate. Until 2026-09-04 the velocity
rows were filled with a target-frame rate; since the chaser's roll at reset is uniform on
`[-pi, pi]`, that is an arbitrarily rotated velocity command worth up to `2 * omega * r`
(about 0.9 m/s at 16 m against a 1.10 scale), and it saturated the first step of every
episode that used the interface.

## Fair Pure-MPC baseline

The Pure-MPC row is **horizon 20 with `terminal_weight = 1000`**, fixed setpoint. A
declared sensitivity sweep over `input_weight` and `terminal_weight` established that the
default weighting (`0.01 / 100`) fails for a reason that is not structural: it parks
0.264 m short of a 0.25 m tolerance even when given 400 s, which is a steady-state offset
of a quadratic-cost regulator. One weight change removes it. On seed 262000 the fair row
completes in 113.1 s with equivalent delta-v 2.426 m/s, worst normalised margin +0.145
and 81.5 ms per step (0.82x of the control period) -- faster than the offline oracle
(228.5 s), as safe as it (+0.148), and 1.60x more expensive in propellant (1.512 m/s).
Do not use the default-weight rows as a baseline; they are sensitivity evidence.

## Reproducible checks

```powershell
python -B -m eval.validate_precapture_semantics
python -B -m pytest -q
```

The offline feasibility gate is the coast-then-match oracle:

```powershell
python -B -m experiments.evaluate_precapture_oracle --episodes 5 --seed 262000 --output OUT.json
```

It must clear all five acceptance checks before any Pure-MPC comparison row is read as
evidence, and before any training starts. Its purpose is narrow: it certifies that a
cheap admissible path *exists*, so a controller that fails to find one has a planning
defect rather than an infeasible task. It is truth-aware and offline by construction --
it reads the cached target trajectory to choose when to synchronize -- and is therefore
never a baseline row.

Long training remains prohibited until the Pure-MPC comparison block establishes the
planning-timescale structure and supports the single Commit-5 Go/No-Go decision.
