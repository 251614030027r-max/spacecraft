# Working notes for this repository

High-fidelity 6-DoF SE(3) **pre-capture** of a fast-tumbling non-cooperative
target (chaser 106 kg, +-2 N / +-0.6 N*m per axis; target 225 kg box, spin
2.1-2.7 deg/s, unknown inertia; 500 km / 45 deg orbit, RK45 truth with central
gravity and J2, 0.1 s control). Current line: the **T2 staged approach**
(`env/t2_stage.py`, `train/t2_mve_sac.py`, `experiments/t2*.py`). SAC gives a
2-D stage intent every 16 s in the target's spin frame (closing speed, along-
sweep speed; latitude regulated onto the port's sweep cone); a stage planner and
a far-field tracking law fly it; when the port's pass is reachable a window-
committed conditional G* plans the entry and the frozen MPC flies entry, descent
and capture. **EACC** (executor-calibrated counterfactual coupling) is the
training-time upward path: counterfactual labels for K = 8 candidate intents from
the executor's own model (M1), anchored to the executed stage (M2), Cauchy-
weighted (M3), entry outcomes by historical execution replay, two-stage labels
(soft Retrace + one-step stage-2 rows), calibration shared by the workers, labels
ending at the deadline / budget. Arms: `e_sac` (no model) and `e_mve` (EACC).

**State (2026-10-29):** env `t2-stage-0.4.2`, learner `t2-learner-2.6.1`, code
`8e612e7`, 46 tests. The feedback's own gain has not been shown: on the standard
task EACC and SAC tie (T2.5.1 `mixed`); on the time-pressure task (urgent
deadline U(140, 200) s, T2.6W, `e1166eb`) EACC was behind (-9.6 / -13.2 paired
over 1000-4000), all on urgent missions. The lower window found the cause that is
confirmed: stage-2 labels did not end at the deadline (fixed in `ca96793`). The
check run order `T2_6W1_CUTOFF_20261029` (collab) **waits for the user's GO**.
The full story, the pre-agreed next branches and every rule are in collab
`docs/collaboration/HANDOFF.md` (sections 1, 5, 7) and the research evolution
(stages 20-26). The T1 line below (T1-L, T1-O, T1-N, T1-B) is finished.

## Start here

**First read the shared three-window documents** on branch `collab/spacecraft`:
`docs/collaboration/HANDOFF.md` (research question, scenario problem, user's
requirements, assets, current state, each window's discipline) and
`docs/collaboration/RESEARCH_EVOLUTION_AND_COUPLING_DESIGN_20261012.md` (how the
thinking and experiments evolved, pitfalls). Then this file and the pages below.

| What | Where |
|---|---|
| **T2 (current): environment, labels, learner, driver, readouts, decisions, audit, figures, tests** | `env/t2_stage.py`, `train/t2_mve_sac.py`, `experiments/t2.py`, `experiments/t2_readout.py`, `experiments/t2_decide.py`, `experiments/t2_audit.py`, `experiments/t2_figures.py`, `tests/test_t2_stage.py` (map in collab HANDOFF 6.2) |
| T2 evidence | `evidence/T2_PRECHECK`, `T2_STAGE_A/B`, `T2_2_TEST`, `T2_3_TEST`, `T2_4_TEST`, `T2_5_VALIDATE`, `T2_5_1_VALIDATE`, `T2_6_PRESSURE`, `T2_6W_WIDE_140_200` (with the lower's diagnosis) |
| User's slides (paper narrative) | collab `docs/collaboration/wang23.pdf`; mapping to the code in HANDOFF 2.8 (the reward slide differs from the code) |
| T1 task family, physics, G*, review interface, opening blocks | `docs/current/TASK.md` |
| Method components (nothing frozen) | `docs/current/METHOD.md` |
| Learning-space admission protocol (screen, Gates A-E) and results | `docs/current/EVALUATION.md` |
| Commands | `docs/current/REPRODUCE.md` |
| Branches, tags, evidence directories, local artifacts | `docs/archive/GIT_ASSETS_20261018.md` |
| One-page research timeline | `docs/HISTORY.md` |
| **Every direction already tried and refuted** | `docs/DEAD_ENDS.md` -- read before proposing any method, interface, reward or task change |
| Live project state, run orders, hand-offs, experiment registry | branch `collab/spacecraft`, `docs/collaboration/` (`PROJECT_STATE.md` wins over anything here) |
| Pre-cleanup repository (all old paths and SHAs) | tag `archive/pre-cleanup-20261010`; map in `docs/archive/PATH_MAP.csv` |

## Roles and branches

- The research layer (user + discussion window) owns the paper question and
  success criteria. The upper window reviews, writes run orders and code; it
  may question but not redefine them. Claim-breaking issues are reported as:
  what, which claim, why not a limitation, minimal check, method change needed.
- The lower window runs long jobs on the user's machine in the single science
  checkout `D:/py/DRL2` (the old `DRL2_T1*` clones are retired; local assets are
  mapped in collab `lower/handoffs/cleanup_inventory_20261018/`), never
  switches branches or pulls in a working tree while a job runs.
- **Exactly three branches** (user, 2026-10-18):
  `claude/sac-mpc-coupling-design-ns7g6i` holds all science code and evidence;
  `collab/spacecraft` only `docs/collaboration/` (fetch first, never
  force-push); `main` is merged only when the experiments are over. Do not
  open new long-lived branches: frozen science states are commits (and, if
  needed, `archive/*` tags made by the lower window). The retired experiment
  branches were fast-forward ancestors and are merged. No PRs unless the user
  asks.
- This sandbox cannot push tags or delete remote branches (proxy 403); the
  lower window does those from the user's machine.
- **Git holds facts, not discussion.** Code, executed run orders, results,
  evidence, registry, hand-offs and reviews of finished results go in git.
  Direction ideas, proposals, brainstorming and window-to-window discussion
  stay in the conversations; a proposal enters git only as an authorised
  run order, and a conclusion only with the artifact behind it.

## Non-negotiables

- **Truth RK45 geometry is the only arbiter of safety**, never observation,
  QP "solved" or predicted margins.
- **Pure MPC is frozen** (no Q/R, horizon, corridor, terminal or tolerance
  edits except a named code bug) and is re-run clean on every formal block.
- **No manufactured gap**: no artificial obstacles, unmotivated noise,
  shortened horizon, throttled thrust, wrong model, or a rule that only hurts
  the baseline. Reward never encodes the answer.
- **Every training run starts from zero** (no resume, no warm start); >= 3
  seeds; report the distribution -- no swapping seeds, picking checkpoints or
  dropping failed seeds.
- One interpretable factor per experiment, in its own commit, with a manifest.
- Evaluation is deterministic and serial; parallel timings are never a
  real-time claim.
- **Any number entering a decision or the paper points at an artifact**
  (JSON/CSV/manifest/commit/script) or is marked exploratory.
- Development validation != paper formal validation != flight certification.
  Smoke tests check correctness, never direction; one seed or probe never
  switches the research question.

## Traps

T2 line (current):

- **Mechanism readings are not label correctness.** w2, pi_gap / sigma, grad_cos,
  two_stage_share passed in T2.5.1 and T2.6W while returns tied or fell. Before
  training on a changed task distribution or label branch, audit the labels against
  truth branches on the critical cases (near the deadline, near-field direct
  capture, lost entry); `experiments.t2_audit` and the T2.6 timeout diagnosis are
  the tools.
- **Terminal semantics.** A modelled stage that crosses the deadline or the budget
  ends there (`_stage_cutoff`, kind 0); an entry is kind 1 (anchored) / 2 (outcome
  model only). Stage-1 labels exist only after a completed factual stage, so the
  deadline matters in stage 2 (decisions within 32 s of it).
- **Paired means are flips.** One capture / timeout flip is about +-200, i.e. +-8 on
  a 24-opening mean; report medians, without-top-2, urgent / economy and deadline
  bins; both-captured openings only for fuel / time.
- **The standard task has no headroom:** SAC reaches ~0.9 training capture by ~1000
  decisions and ~54 / 23-of-24 by 3000-4000.
- **SAC arm reuse:** the SAC arm generates no labels (candidates = 0); label or
  learner-model changes leave it bitwise on the same code path, so it may be reused
  across EACC-only rounds -- say so in the run order.
- **Lower hand-offs since T2.2 live in collab `lower/handoffs/` at the repository
  root**, not under `docs/collaboration/lower/handoffs/`.
- **Check pytest by its exit code**, not by piping to `tail`.
- Do not fine-grid task parameters to find a profile where EACC wins (T2.6B was
  withdrawn); widen the distribution and pre-register the bins instead.

T1 line (historical):

- **Solver-cache numerics.** A long-running MPC's CVXPY/CLARABEL cache carries
  its own solve history; a shadow clone cannot be bitwise equal to it (~1e-8).
  Branch from a snapshot with `CounterfactualPredictor.sync(fresh_solvers=True)`
  for every branch and report the difference to the original continuation as
  the numerical repeatability.
- **`tau_min_feasible` is not observed** in the T1-L / T1-O interfaces (computed
  inside `apply_intent`; rules and policies must not read it). T1-N observes a
  read-only earliest-feasible summary instead, shared by every arm and rule.
- **T1-N shadow = simulator cost.** The 2 s MPC shadow solves the MPC every
  0.1 s; it costs as much as a truth branch and is nearly as accurate. Do not
  claim it is cheaper or more informative than the simulator.
- **T1-N ranking signal = G* plan + MPC 2 s correction.** Its long-horizon
  content comes from the G* plan; report the plan-only and corrected signals
  separately and never credit the plan's part to the MPC. Only its sign is
  used (ranking loss); nothing is bootstrapped from a planned arrival state.
- **T1-N 0k is a constant mid-range request.** The untrained deterministic actor
  outputs a ~ 0: tau_req ~ 110 s at t = 0, i.e. the fixed:0 rule.
- **Terminal semantics.** success / unsafe / timeout are terminal (no
  bootstrap); only a numerical failure is a truncation.
- **Fuel weight.** Re-score any weight exactly from `info["delta_v_step_m_s"]`;
  never rerun physics per weight. The counterfactual gate `e_r = 0.06` was set
  at weight 5 and is not valid at another weight.
- **G* planned delta-v covers only the transfer**, not station-keeping or
  descent; it is biased towards long waits as a cost proxy.
- **Do not edit modules a running job imports**: evaluation pools spawn new
  processes per stage and import the current working tree.
- **Opening blocks** are listed in `docs/current/TASK.md`; formal block
  300000-999999 never accessed; training openings >= 1,000,000.
- Cross-machine numerics: the same commit can differ in the last bits between
  the sandbox and the lower's machine; compare verdicts, not bits, across
  machines.

F01 line (historical, `train/stopping.py` and friends):

1. The raw per-key margins in `info` are not gated; compare
   `minimum_truth_normalized_margin` (from
   `normalized_precapture_truth_margins`).
2. `predicted_minimum_margin` is the optimiser's forecast, only exists under
   `runtime_diagnostics=True`, and is not a truth margin.
3. `maximum_slack` can be stale on a fallback step; read it only from solved
   steps.
4. The wrapper is transparent only when `horizon_steps ==
   external_reference_hold_steps`.
5. `target_entropy` is inert while `ent_coef` is fixed (0.005). Automatic
   temperature drove critic overestimation (DEAD_ENDS D02).
6. Loading a checkpoint against the wrong observation dimension is refused by
   the evaluators; match the run's manifest, do not guess flags.
7. The `max` of a compute measurement is step zero (cold start).
8. h35 vs h50 is separated by steps over the control period (2/300 vs
   269/300), not by p95.
9. No `conftest.py`, not installed as a package: run
   `python -B -m pytest -q` from the repository root.
10. An **illegal entry-plane crossing is a diagnostic count**, not a safety
    violation; it does not latch or end the episode.
11. **Training-log completion is behaviour-policy statistics** (stochastic
    actions plus handoffs sampled at random times); it is not deployment
    performance and never a gate.
12. The formal evaluator builds its hybrid config without
    `decision_discount_factor`; trajectories agree bitwise with training, only
    the shaped reward differs. Compare outcomes and states, not shaped reward.

## Trusted entry points

T2 line (current; exact commands in the collab run orders, e.g. `T2_6W1_CUTOFF_20261029.md`):

| Purpose | Entry |
|---|---|
| Tests | `python -B -m pytest -q tests/test_t2_stage.py` (46) |
| Fixed policies / G* / pure MPC screen | `python -B -m experiments.t2 screen --openings A-B --policies rally,c_10_0,...,gstar,pure [--urgent-deadline LO,HI]` |
| Train | `python -B -m experiments.t2 train --arm {e_sac,e_mve} --seed S --steps N --workers W --checkpoints ... --output DIR [--urgent-deadline LO,HI]` |
| Evaluate a checkpoint | `python -B -m experiments.t2 eval --checkpoint ZIP --label L --openings A-B --workers W --output DIR [--urgent-deadline LO,HI]` |
| Readouts | `python -B -m experiments.t2_readout {screen,curve,window,pressure}` |
| Pre-registered decisions | `python -B -m experiments.t2_decide --criteria {t23,t24,v25,v251,t26,t26w} [--checkpoints ...] --readout R --train-root DIR` |
| Label audit against truth | `python -B -m experiments.t2_audit {warmup,labels,summary}` |
| Figures | `python -B -m experiments.t2_figures {traj,plot}` |

T1 line (historical; commands in `docs/current/REPRODUCE.md`):

| Purpose | Entry |
|---|---|
| Task profiles | `env/t1_env.py`, `env/t1_l_task.py`, `env/t1_o_task.py` (`p1` / `p2` / `o`), `env/t1_n_task.py` (T1-N) |
| T1-N interface, shadow, learner | `env/t1_n_review.py`, `env/t1_n_cf.py`, `train/t1_n_sac.py`, `python -B -m train.train_t1_n`, checks `experiments.t1n_check`, evaluation `experiments.t1n_eval` |
| G* and review interface | `env/t1_gstar.py`, `env/t1_task_review.py`, counterfactual `env/t1_option_cf.py` |
| Learners and training | `train/t1_option_sac.py`, `train/t1_tcpf.py`, `python -B -m train.train_t1_option` |
| Openings (geometry only) | `python -B -m experiments.t1o_openings`, `experiments.t1o_screen openings` |
| Fixed candidates, tables | `python -B -m experiments.t1o_screen {run,verdict,table}` |
| Gates A-E (pre-registered) | `experiments/t1o_v2.py`, `experiments/t1o_gates.py`, `experiments/t1o_cf_check.py`, chain `experiments/t1o_v2_chain.py` |
| T1-L evaluation / readout | `python -B -m experiments.t1_l_eval`, `experiments.t1_l_readout` |
| Pure MPC row | `python -B -m experiments.t1_gstar_probe run --row pure --task ...` |

F01 line (historical):

| Purpose | Entry |
|---|---|
| Mainline configuration | `train.mainline.mainline_v3e_configs()`, regimes in `train.regimes` |
| Stopping method | `train/stopping.py` (`STOPPING`) |
| Train | `python -B -m train.train_stopping --steps 60000 --seed S --run-name N --regime w2.36_r15` |
| Evaluate rows / devcheck / readout | `python -B -m experiments.v3_stopping {evaluate,devcheck,readout,readout-value}` |
| Nominal row, regime screen | `python -B -m experiments.regime_screen` |
| 96-opening tables | `python -B -m experiments.final_tables` |
| Replay + counterfactual (eps_H, eps_C) | `python -B -m experiments.stopping_replay` |
| Training health (read-only, safe while training) | `python -B -m experiments.v3_stopping_training_health` |
| Pure MPC / scripted / oracle | `experiments.evaluate_mpc`, `experiments.evaluate_hybrid_scripted`, `experiments.evaluate_precapture_oracle` |
| Serial MPC cost | `python -B -m experiments.profile_precapture_mpc --no-diagnostics` |

## Artifacts

Training artifacts (models, checkpoints, TensorBoard, per-episode JSON) are
git-ignored and archived locally by experiment ID. Publish evidence only by
force-adding individual files under `evidence/<experiment-id>/` and record the
SHA-256 in the registry. Never commit an artifact directory (the pre-cleanup
`logs/` held 1.6 GB of model ZIPs; it is out of the tree now, see
`docs/archive/ARTIFACT_INDEX.csv`). Do not rewrite history: cited SHAs are the
evidence chain.

The sandbox has no numeric stack by default; build a venv with
`numpy scipy cvxpy gymnasium stable-baselines3 pytest` (PyPI torch works;
download.pytorch.org is blocked).

## References

See `References/README.md` (what each paper is for and what not to claim).
