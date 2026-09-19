# Window handoff -- 2026-09-19

Read this, then `CLAUDE.md` (top banner + the two 2026-09-18 discipline blocks),
then `docs/REWARD_UNITS_FIX_20260919.md` and `docs/ADAPTIVE_MAINLINE_RUNSHEET.md`.
This file is the spine; those are the detail. Nothing below reopens a settled
call -- it records them so the next window does **not** re-litigate the direction.

---

## 0. One-paragraph state

The mainline is frozen and the full trainable/comparable loop is built and
green (258 tests). The first 3-seed adaptive training run hit a **reward units
bug** (the safety proximity warning was not time-integrated, so a legal capture
scored ~-75 while a do-nothing hover scored ~+4 -- objective inverted, and the
SAC critic the deployment gate reads was being poisoned). It is **fixed** (one
line, `* time_step_s`), verified by direct measurement, and documented. **The
only pending action is to retrain the 3 seeds from zero with the fixed reward**
(`adp_rf_262410/411/412`) and read a 10k early health check. Do not redesign
anything before that retrain reports.

---

## 1. The mainline (frozen -- do not reopen)

**Adaptive sync-entry decision task + ordinary strong MPC + baseline-anchored
SAC-MPC coupling.** Entry point `precapture_adaptive_capture_environment_config()`.

The paper problem: during pre-capture of a fast-tumbling non-cooperative target
(Lambda = omega*r/v_max > 1, so station-keeping is inadmissible and the chaser
must co-rotate), **how the chaser autonomously trades continuous synchronisation
/ co-rotation against staging-then-opportunistic-entry**, while a strong
constrained MPC does the short-horizon 6-DoF tracking, constraint satisfaction
and execution.

> **RL decides *how to do the task* (the long-horizon, state-dependent
> sync-vs-enter resource decision). MPC decides *how to fly the current intent
> safely*.** That is the principled reason RL belongs here: hand rules are
> brittle, and the short-horizon MPC cannot see the trade-off. The
> generalisation this buys -- the *same* policy changing its behaviour as tumble
> rate and initial phase/position vary -- is RL's real advantage and the paper's
> claim.

Architecture: RL is a **bounded reference generator on a slow (2 s) decision
timescale**; MPC is the fast tracking subsystem; the residual is bounded; a
baseline/fallback reference always exists. Keep it analysable (later
recursive-feasibility / tracking-stability work depends on it). **Do not build a
black box.**

## 2. Locked calls -- the user's explicit demands (violating these = drift)

These are decisions the user made deliberately, several after correcting an
earlier version of me. Treat them as constraints, not suggestions.

1. **Opportunity is a continuous cost structure, NOT a legality gate.** No phase
   gate, no favourability legality, no new hard far-range constraint.
   Co-rotating while the port is misaligned costs more fuel/actuator/margin --
   that is all. **Pure MPC must always keep a legal path (co-rotate the whole
   way and complete) and stay a genuine strong baseline. Never shape the task so
   Pure MPC must lose.** Far-range "don't loop in from 20 m" is handled by sane
   init + loose reference shaping, not a hard constraint. Real safety stays with
   keep-out / FOV / speed / terminal only.
2. **Thesis framing (the user corrected this explicitly -- do not revert):**
   - Nominal regime: the coupling must **RETAIN** Pure MPC's completion and
     safety (baseline retention). It is *not* required to beat MPC on fuel in the
     nominal case.
   - The learning layer's value is shown **across regimes**: the same policy
     changes its sync-vs-enter behaviour as tumble rate and initial phase/pos
     vary, keeping a better overall completion/fuel/time/margin trade-off.
   - **Do NOT claim "RL wins, MPC loses."** That framing is banned.
3. **Decision-margin calibration is ONE restricted measurement, never again a
   "direction life/death gate."** (This was a hard-won lesson: earlier windows
   kept killing the direction on small probes.) 3 tumble rates (slow /
   2.36 deg/s nominal / ~3.5 deg/s), small seed set, two extreme *reasonable*
   strategies only (`--control desired_pose` early co-rotate vs `--control
   timed_entry` stage-then-enter), identical MPC/constraints/execution. Read only
   completion / equivalent-dv / time / margin. A clear trade-off -> implement. A
   weak one -> **tune near-field distance / initial range / realistic tumble
   range to build the physical decision space, then continue the same mainline.**
   Never a window-angle sweep, a gate, ten commit-times, or a direction change on
   an ugly result.
4. **Action space:** V1 reuses the 2D `arrival_condition`, renamed *capture
   progress / reference blend* (a[0] = commit blend inertial-hold<->desired-pose,
   a[1] = hold radius). An explicit `sync_level` dynamic reference generator is a
   **V2 method optimisation**, only if training shows staging and partial
   co-rotation are inseparable -- not a task failure.
5. **Build first, judge on the formal loop.** A small-scale probe result is **not**
   grounds to veto the direction. Form the complete trainable/comparable/iterable
   closed loop, run >=3 seeds, judge on that. Do not keep overturning the plan.
6. **Don't front-load** the stability proofs or the final-figure / big-ablation /
   real-time-profiling / perception-stress work. Get the base loop converging
   first, then research the coupling mechanism (MPC feasibility / slack / value
   feedback into the high-level update) systematically.

## 3. Research execution discipline (adopted 2026-09-16, still binding)

- **No probe proliferation.** Once a version's design logic holds:
  `implement -> smoke -> formal train (>=3 seeds) -> formal evaluate`. Not
  `probe -> probe -> oracle -> grid -> seed archaeology`.
- **Smoke is a correctness check, not a direction gate** (runs, shapes, no NaN,
  solver sane, fallback fires, residual 0 recovers nominal). Its rate never
  decides direction.
- **>=3 training seeds** before any conclusion. One seed failing/passing proves
  nothing.
- **Do not switch the research question on a single probe/seed/metric.** After a
  version fails, first check implementation/convergence/whether the coupling did
  what it should. Only consistent multi-seed formal negatives reopen anything.
- **Every new module must name its main claim** ("delete it -- does the paper
  still stand?"). If not, it is not added.

## 4. Non-negotiables (scientific integrity -- never relaxed)

- Truth RK45 geometry is the **only** arbiter of safety/violation. Never the
  observation, never the QP "solved" flag, never predicted margins.
- **No manufactured gap:** no artificial obstacles, unmotivated noise, shortened
  horizon, throttled thrust, deliberately wrong model, or any rule change that
  only hurts the nominal case / only hurts Pure MPC.
- **Reward never encodes the answer** (no wait/enter timing, no per-seed success
  times baked in).
- No cherry-picking seeds or checkpoints; report the distribution.
- **Every number that enters a decision or the paper must point at an artifact**
  (JSON/CSV/manifest/commit/script) or be marked exploratory/unreproduced.
- Pure MPC edits are allowed **only** for a named code bug or a genuine
  input-semantics change -- never a Q/R/horizon/corridor/terminal/tolerance tune.
  (The reward fix below is *not* a Pure MPC edit: the reward is the RL training
  signal, which the Pure MPC controller does not use. Pure MPC's number is
  unchanged.)

## 5. What just happened -- the reward units fix (the live event)

**Symptom (artifacts: 3 uploaded `train.monitor.csv`, seeds 262410/411/412,
~30k):** completed episodes returned median ~-75 (min -110), non-completed ~+4;
completion ~70%. A successful legal capture scored 80 points *below* hovering.

**Root cause (`env/reward.py` `PrecaptureReward.compute`):** the time/force/torque
penalties are rates carrying `time_step_s`; the safety proximity warning was a
raw per-step `sum(warnings)` with no `time_step_s`. At dt=0.1 s it accrued 10x
too heavily per second, so a legal terminal approach (which in a Lambda>1 capture
necessarily rides near the closing/total-speed buffer for its ~40 s / ~400 steps
in the terminal region) banked ~-100, swamping the +20 completion event.

**Fix:** one line -- multiply the safety term by `time_step_s`, matching the
other three penalties. `safety_weight` unchanged; no weight tuned; no grid.
This is the pre-authorised "one physical-scale fix", framed as a dimensional bug.

**Direct verification (sandbox, real MPC rollouts, adaptive task, post-fix):**

| behaviour | completes | safety | return |
|---|---|---|---|
| full hold (commit -1) | no | 0 | +6.7 |
| partial commit (mid) | no | 0 | +4.7 to +5.5 |
| full commit / residual=0 | yes | -9.79 | **+16.7** |
| clean completion (other trajectory) | yes | 0 | +26 |

- The grazing completion was **reproduced directly**: reverse the fix (x10) ->
  safety -98 -> return ~-71, matching the training logs' -75. Post-fix +16.7,
  cleanly above hover.
- `residual=0` == non-residual full commit bitwise in closed loop (+16.66 both):
  the "residual 0 = Pure MPC" anchor holds.
- Landscape is healthy: completion is the global max; hover/partial are +4.7-6.7.
- 258 tests pass.

**Honest open risks (do NOT hide these from the user):**
- Hover (+6.7) is still a *positive* local attractor. The fix flips the gradient
  from strongly-anti-completion (old: -75 vs +4) to pro-completion (+16.7 vs
  +6.7), but SAC convergence out of hover is **not guaranteed**. Hence the **10k
  early health check** -- fail fast, don't burn another day.
- Per-episode reproducibility has noise: the *same* config measured +26 (clean)
  and +16 (grazing). Both beat hover, so the fix is robust, but final-number
  reproducibility will need the fixed seed block + final model (repo rule).

## 6. Immediate next action (do this before anything else)

Retrain from zero, fixed reward, fresh run-names, 3 parallel processes. Command
is in `docs/ADAPTIVE_MAINLINE_RUNSHEET.md` section 2b:

```
python -B -m train.train_hybrid --steps 60000 --seed <S> --run-name adp_rf_<S> \
  --horizon 35 --parametrization arrival_condition \
  --baseline-anchored-residual --adaptive-task --device auto
# S in {262410, 262411, 262412}, one process each, no GPU.
```

**10k health gate:** completed-episode return must be clearly > hover
(expect completion ~+16 to +26, hover ~+4 to +7). If not, stop and diagnose --
do not run to 60k on a poisoned signal. Then formal evaluate per runsheet 3.

## 7. Honest ceiling (don't oversell to the user or in the paper)

Per-episode baseline-preserving upper bound from T12 is ~**38/48 (+6)** over Pure
MPC's 32/48. This is a **mechanism paper with a modest, real gain**, not a
blowout. The arbitration pattern itself is known (南航 / AC4MPC) -- concede it;
claim the *combination* + the *task-level* opportunity + the `Lambda>1`
non-cooperative regime. The positive T12 finding that motivates the whole thing:
the learned policy *rescues* states Pure MPC fails (262401: +6) but *destroys*
states it succeeds (-4); the coupling exists to turn that complementarity into a
stable net gain without breaking the baseline.

## 8. What is HISTORY / superseded (do not resurrect)

- **"Prove RL beats MPC" and "find a factor that makes MPC fail"** are closed
  (T12 26/48 vs 32/48; non-coop probe 2 31/48 vs 32/48 -- both negative). The
  question is now the coupling *mechanism*, not beating MPC.
- **The opportunity task with a hard outer inertial approach corridor**
  (2026-09-17, `outer_approach_half_angle_rad`, `precapture_opportunity_*`) is
  **superseded** -- that corridor made Pure MPC violate a new hard constraint =
  the manufactured-gap story we avoid. Code stays off by default; do not turn it
  on as the mainline.
- **The entry-phase-gate / Path-2 scaffolding on the branch**
  (`entry_phase_gate_cos`, favourability legality, `fd03ded`/`1f0e46a`/`26c7e05`
  etc.) predates the 2026-09-18 locked call #1 (no phase gate, no favourability
  legality). It is scaffolding, **off by default; do not adopt it as the
  mechanism.** Opportunity is a cost, not a gate.
- The 24D mission schema / Waypoint / single_phase / A1-A3 perception line is the
  Historical research line (CLAUDE.md bottom). Lessons transfer; the code is not
  live.
- **Void, do not cite:** the return audit (+14.11/-24.0) and the action-space
  audit (a0=-0.76..-0.81) -- their scripts never existed and were never reproduced.

## 9. How the work is run (collaboration model)

- Long training runs on the **user's machine** (`D:\py\DRL2`), not the sandbox.
  This window does: code review, `pytest`, short smoke runs, diagnostic probes,
  reward/interface measurement, writing docs.
- Delivery to the user is a **zip + copy-paste terminal commands** (PyCharm
  terminal, parallel windows for multi-seed). The user runs them and sends back
  logs / `train.monitor.csv` / eval JSON for the window to diagnose.
- **Do not `git push` without explicit user permission.** The user manages
  commits/pushes on their side. (Standing instruction, stated forcefully.)
- Sandbox has no numeric stack by default: build a venv and
  `pip install numpy scipy cvxpy gymnasium stable-baselines3 pytest` (+ `torch
  tensorboard` for the full suite; the pytorch CDN is proxy-blocked, install from
  default PyPI). Run tests as `python -B -m pytest -q` from repo root
  (no conftest, not installed as a package). Reward-only measurements don't need
  torch.

## 10. Interaction style the user expects (learned over this collaboration)

- **Plain language ("说人话"), no jargon-for-its-own-sake, no filler ("不废话").**
- **Honesty over optimism.** Report what failed, with the artifact. Do not
  cheerlead. If a claim is inferred not measured, say so.
- **Do not re-litigate settled direction.** The user has pivoted several times
  and is tired of windows that overturn the plan on thin evidence. Build and
  measure; bring a recommendation, not a survey.
- **Automate and deliver.** When told to proceed, proceed without a stream of
  clarifying questions; give a complete, runnable deliverable.
- When the user pushes back ("你确定吗"), **re-verify by direct measurement**, do
  not defend. The reward fix above was upgraded from inference to direct
  measurement exactly because the user pushed.

## 11. Git state at handoff

- Remote `origin/claude/sac-mpc-coupling-design-ns7g6i` tip `164c128` has **all
  code byte-identical to the verified fix** (reward fix, adaptive task, coupling,
  gate, CLAUDE.md). Confirmed: no code file differs.
- **Missing from the remote** (the user's `git commit -am` did not stage new
  files): `docs/ADAPTIVE_MAINLINE_RUNSHEET.md`, `docs/REWARD_UNITS_FIX_20260919.md`,
  `docs/FAST_TUMBLING_CAPTURE_MOTIVATION.md`, `docs/MAINLINE_TRAIN_EVAL_RUNSHEET.md`,
  `docs/OPPORTUNITY_MAINLINE_RUNSHEET.md`, `tests/test_adaptive_task.py`,
  `tests/test_deployment_gate.py`, `tests/test_opportunity_task.py`, and this
  handoff file. They exist on the user's disk (from the delivered zip); they just
  need `git add` + commit + push. Until then the next window should treat the
  runsheet/fix docs as authoritative from the zip, not the remote.

## 12. Reference papers (what each is FOR / NOT)

- `南航.pdf`: motivation (its Limitation 1: LTI prediction assuming *moderate*
  tumbling). Do NOT claim the layered architecture as novel -- 南航 is already
  upper-proposes/lower-filters. Ours: the upper layer is *learned* and the regime
  is Lambda>1.
- `AC4MPC`: critic-as-terminal-cost + parallel double-solve "no worse than" bound.
  Both halves are closed to us (learned terminal cost measured harmful; double
  solve = 1.5x budget). The infeasibility is itself a publishable contrast.
- `北航编队.pdf`: the RL-supplies-a-schedule-to-MPC interface pattern (not its
  impulsive model). `ecc26_rollout.pdf`: learning supplies a good nominal,
  optimisation improves in its neighbourhood.
- `上海交大.pdf`: 3-way comparison table format only. **Its method (online MPC
  cost-weight adaptation) is ruled out.**
- `哈工大.pdf`: SE(3) modelling base (not a contribution). `北航.pdf`: constraint
  handling; its 0.0173 rad/s target + absent speed cap put our regime outside it.
- `引入死区迟滞...pdf`: the user's own prior paper -- do not cite.
