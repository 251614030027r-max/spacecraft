# Working notes for this repository

High-fidelity 6-DoF SE(3) pre-capture control for a tumbling non-cooperative
target. Chaser 106 kg, target 225 kg free-tumbling, 500 km / 45 deg circular
orbit, RK45 truth with central gravity, second moments, gravity gradient and
J2, 0.1 s control period, 200 s episode cap, +-5 N / +-0.6 N*m per axis.

## Research line

Three-way comparison: Pure SAC, Pure MPC, SAC-MPC hybrid. The hybrid is
hierarchical -- an upper learned layer decides, a lower MPC layer enforces
feasibility and safety -- and is **architecture-uniform across the whole
mission**, not switched on for the terminal phase only. Switching architecture
mid-episode would confound the comparison with the phase switch itself.

**The common benchmark is the end-to-end `full_mission`**, cleared for that use
by `eval/validate_phase2_semantics.py` (20/20 scripted completion). Phase-I is
a SAC *training* curriculum and the source of the coupling analysis, not a
separate headline result. Training may be staged; evaluation may not.

**The entropy temperature was the driver of critic overestimation, and
pinning it fixed the critic.** `auto_0.005` never converged: alpha rose
monotonically in every run ever measured, and calibration error was monotone
in alpha across two observation schemas and two update-to-data ratios.
Raising `gradient_steps` to 4 only made alpha diverge faster (0.0041 -> 0.109
by 125k, tripping the abort gate, calibration error 45.7), so that factor was
reverted. The likely cause is that `target_entropy = -6.0` is unreachable for
a 6-dimensional tanh-squashed Gaussian whose scale already saturates, so the
tuner raises alpha without bound and the entropy bonus, entering every
bootstrap over a 333-step effective horizon, inflates the soft targets.

| alpha | calibration error | run |
|---|---|---|
| 0.005 fixed | **-0.46** | fixed-alpha 100k |
| 0.005 fixed | **+0.94** | fixed-alpha 200k |
| 0.0037 | 4.9 | v3 200k |
| 0.0146 | 18.4 | v3 final |
| 0.0149 | 21.5 | v2 400k |
| 0.0203 | 32.2 | v2 final |
| 0.0605 | 45.7 | UTD 4 at 100k |

With `ent_coef = 0.005` fixed, the run at 100k-200k is the **first learned
policy to beat the trivial baselines**: Gate 6/20 (previous best 2/20, at half
the steps), survival 96.2 s at 200k against the 53.3 s zero-action baseline,
and a critic calibrated to within one unit of a reachable return span of 16.
The 100k periodic evaluation on the disjoint 20288000 block gave the same
0.30 Gate rate, so it is not seed luck.

**Replay eviction is not the cause of the late decay.** Doubling
`buffer_size` to 600k, so nothing is evicted at all within a 500k run, changed
nothing: calibration error at 500k went 23.661 -> 23.536 and `critic_q1` 3.181
-> 2.801. The two runs are bitwise identical through 300k (same seed, and the
buffers only differ once more than 300k transitions exist), which is the
control this comparison needed. The factor is reverted.

The decay begins before eviction could matter -- calibration error is +0.94 at
200k and 14.159 at 300k, while the 100k-200k data would not leave a 300k
buffer until 400k. The behaviour collapses first and the critic follows: the
true soft return falls from -0.4 to -13.3 between 200k and 300k, and only
afterwards does `actor_action_std` slide monotonically (0.67 at 300k -> 0.37
at 500k) with saturation reaching 0.34-0.64. The remaining explanation is the
actor/critic feedback loop: with alpha pinned at 0.005 the entropy term is
negligible against a Q that has grown to order 1, so the actor turns greedy,
narrows the data distribution, and the critic extrapolates.

Current state: **the result to defend is the 100k-200k checkpoint, not the
500k one.** Before any of it is reported upward it needs the >= 3 training
seeds the rules below require; the late decay is a characterised limitation,
not a blocker.

## Trusted entry points

| Purpose | Entry |
|---|---|
| Environment | `env.phase2_env.phase2_environment_config("phase1_pretrain" \| "full_mission")` |
| Task | `env.phase2_env.phase2_s1v2_mission_config()` |
| SAC config | `train.configs.PURE_SAC` (single instance, pinned by tests) |
| Train | `python -B -m train.train` |
| Evaluate | `python -B -m eval.evaluate_policy` |
| Phase-I reachability | `python -B -m eval.validate_phase1_semantics` |
| Full-mission reachability | `python -B -m eval.validate_phase2_semantics` |
| Critic calibration | `python -B -m eval.diagnose_value_calibration` |
| One-screen run digest | `python -B -m eval.digest_run --run logs/<run>` |

## Rules

- **Strictly one interpretable factor per experiment**, recorded in the
  manifest and in its own commit. Never mix a task-parameter change with a
  hyperparameter change.
- **S1-v2 task parameters are frozen.** Changing them invalidates the
  comparability of eleven prior single-factor rounds. Change one only on
  measured evidence of a definitional defect, never on judgement alone.
- **Do not judge short experiments by Gate rate** -- it lags and its variance
  is large. Use critic calibration error and survival time.
- No Phase-II entry, no initialising from an existing checkpoint, no resuming
  a mid-run checkpoint. Every long run is fresh actor / critic / replay.
- **`n=1` training seeds are not conclusive.** Independent re-evaluation on a
  fresh seed block flipped 8 of 9 comparable published numbers. Any conclusion
  reported upward needs >= 3 training seeds.
- Report results through `eval/digest_run.py`, not by shipping log files.

## Reference numbers (measured, seed block 262000 unless noted)

| Policy | Gate | Survival | Discounted return |
|---|---|---|---|
| Zero action | 0/20 | 53.3 s | -8.71 |
| Uniform random | 0/20 | 53.5 s | -- |
| Station-keeping (2.50 N) | 0/20 | 200 s, never dies | -1.51 |
| Gate-PD cruise 0.15 | 20/20 | 24.1 s | +3.66 |
| Scripted full mission | 20/20 gate, 20/20 completion | 95.2 s | +5.21 |

Reachable discounted return spans only about 16, so a calibration error of
several units means the critic carries no information about policy quality.
That was true of every `auto_0.005` model (4.9 to 45.7); it is no longer true
at fixed alpha, where the error is under one unit through 200k.

Target body rate 0.0412 rad/s puts the body-fixed Gate on a 0.33 m/s circle,
so co-rotation needs a *sustained* 1.44 N at the Gate and 2.16 N at 12 m, plus
about 1.31 N of Coriolis. Phase-I is therefore attitude-orbit coupled tracking,
not translational rendezvous -- this is the physical motivation for the hybrid.

## Traps

1. `env.task.Phase2MissionConfig` defaults to the **old** S1 task. S1-v2 only
   exists via `phase2_s1v2_mission_config()`. Always pass the config explicitly.
2. `eval/evaluate_policy.py::main` compares a manifest against the current
   canonical config and raises on any difference, so pre-v3 schema models
   cannot be loaded through the CLI. Call the library functions with the
   manifest's own config instead (`diagnose_value_calibration.py` shows how).
3. `train/train.py` writes `translational_observation_frame`,
   `translational_velocity_observation` and `mission_task_version` as string
   literals. Fix those before ever changing the observation schema.
4. `logs/phase2_mission_s1_semanticfix_validation/` predates the current reward
   implementation and cannot support any current claim. Its replacement is
   `logs/phase2_mission_s1v2_validation/`.
5. `terminal_constraint_failure` is evaluated on the Gate transition step
   itself, so an inadmissible arrival ends the episode immediately. Phase-I's
   0.30 m/s Gate speed tolerance does not by itself guarantee admissibility
   against the 0.28 m/s closing-speed limit that applies there.
6. The repository has no `conftest.py` and is not installed as a package, so
   run the suite as `python -B -m pytest -q` from the repository root. A bare
   `pytest` can resolve to a different interpreter and fails collection on all
   thirteen files at once.
