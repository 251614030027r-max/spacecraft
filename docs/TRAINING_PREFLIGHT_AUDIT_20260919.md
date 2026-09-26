# Adaptive mainline training preflight audit

Date: 2026-09-19 JST. Base reviewed: remote-synchronised commit `42067947057a6ab86433d9c500b749327548ba18`; the guards and documentation corrections described here are delivered in the same preflight change set. Scope is correctness/readiness only. No SAC training was started.

## Verdict

**Ready for a fresh three-seed retrain after this change set is committed.** The previous reward inversion is fixed in code and now pinned by tests. No second sign/units inversion was found. The old critics, models and checkpoints remain invalid and must not be resumed. One additional preflight inconsistency was found and fixed: this machine has CUDA available, so `--device auto` contradicted the runsheet's CPU-only parallel plan; hybrid training now defaults to `cpu` and the active commands state `--device cpu`.

## Reward audit

- `PrecaptureReward.compute`: time, normalized force, normalized torque and safety proximity are continuous rates and each carries `time_step_s`; force uses action indices 3:6 and torque 0:3, matching `GeneralizedForce` ordering.
- Success/failure are one-shot events (+20/-20), applied only on completion or hard failure and not multiplied by `dt`; timeout has no event bonus.
- The hybrid wrapper removes the 0.1 s micro-step potential shaping, sums physical/event terms over the 2 s decision, then applies one SAC-consistent macro shaping term with gamma 0.99. Event reward is therefore neither lost nor duplicated.
- Adaptive task adds no phase legality gate, favourability reward or outer hard approach corridor. Truth RK45 geometry remains the completion/safety arbiter.
- Residual zero maps to full commit/Pure MPC; residual -1 reaches the inertial hold. Observation/action dimensions are pinned at 38D/2D.

New regression guards in `tests/test_precapture_reward_units.py` assert continuous-cost scaling at dt=0.05/0.1/0.2, one-shot event signs/magnitudes, and the CPU training default. `tests/test_adaptive_task.py` additionally pins 38D/2D, phase gate off and no outer corridor.

## Exact-config closed-loop measurements

Configuration: h35, `arrival_condition`, execution feedback, monotone commit, baseline-anchored residual, adaptive task; no policy; exact training environment.

| seed / behaviour | outcome | time | undiscounted episode return |
| --- | --- | ---: | ---: |
| 262410, residual 0 / immediate full commit | legal completion | 97.4 s | **+17.072** |
| 262410, residual -1 / full inertial hold | timeout | 300.0 s | **+7.260** |
| 262411, residual -1 / full inertial hold | timeout | 300.0 s | **+5.378** |

The critical ordering is restored: completion > hover/hold. The remaining honest risk is that hover is still positive and can be a local attractor; this is why 10k is a disaster check (completed-episode return must exceed hover), 30k is trend confirmation, and 60k is the formal training endpoint. A weak 10k success rate alone is not a direction gate.

## Training-path audit

- Fresh directories: `logs/adp_rf_262410`, `logs/adp_rf_262411`, `logs/adp_rf_262412`; training refuses an existing run directory.
- Fresh SAC and replay buffer are constructed in each process; no checkpoint is an input.
- Three unique seeds and run names prevent write collisions.
- h35 and all task/coupling flags are recorded in manifest; the manifest now explicitly names the 3D staging-direction observation and the reward units contract.
- This host reports `torch.cuda.is_available() == True`; CPU is explicitly locked to avoid three processes silently contending for one GPU while MPC remains the bottleneck.
- The formerly conflicting health-check wording is resolved in the current runsheet: 10k reward-disaster check, 30k trend check, 60k completion.

## Cleanup performed

- Deleted about 185 MiB of invalid interrupted models, checkpoints and TensorBoard events under `logs/adp_262410/411/412`.
- Retained only three manifests and three Monitor CSVs (93.5 KiB) in `local_artifacts/reward_units_bug_20260919/` as auditable evidence.
- Deleted obsolete duplicate calibration directory `eval/cal/`; retained current `eval/cal2/` locally.
- Archived four loose handoff files under `docs/handoffs/archive_202609/` and replaced stale README/document indexes with the current mainline.
- Added ignore rules for local adaptive logs and evaluation output so a completed run does not dirty Git accidentally.

## Start boundary

Do not start until the preflight commit is pushed or otherwise fixed as the run's recorded code state. Then execute the three commands in `docs/ADAPTIVE_MAINLINE_RUNSHEET.md` section 2b in separate terminals. Do not reuse `adp_262410/411/412`, alter reward weights, change the task, or skip the 10k return-ordering check.
