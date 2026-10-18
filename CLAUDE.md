# Working notes for this repository

High-fidelity 6-DoF SE(3) **pre-capture** of a fast-tumbling non-cooperative
target. Current line: the **T1 task family** (chaser 106 kg, target 225 kg box,
500 km / 45 deg orbit, RK45 truth with central gravity and J2, 0.1 s control,
+-2 N / +-0.6 N*m per axis, 300 s). SAC chooses task intents at reviews
(arrival delay / in-flight shift, descent pace kappa, entry speed), G* turns
them into executable minimum-delta-v plans, the shared constrained MPC (h35)
flies them and feeds 2 s shadow predictions back into training.

**State (2026-10-18):** two learning-space checks (T1-L; T1-O V2, stopped at
Gate C) found no observable state-dependent decision value beyond a simple
uniform policy. The next task version is being designed by the research
layer; no method is frozen. The earlier F01 value-stopping (handoff) line is
finished and archived (`docs/archive/06_f01_value_stopping/`).

## Start here

**First read the shared three-window handoff:** branch `collab/spacecraft`,
`docs/collaboration/HANDOFF.md` (background, current state, each window's role
and discipline). Then this file and the pages below.

| What | Where |
|---|---|
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
- The lower window runs long jobs on the user's machine (one independent clone
  per line: `D:\py\DRL2`, `DRL2_T1`, `DRL2_T1_L`, `DRL2_T1_O`), never
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

T1 line (current):

- **Solver-cache numerics.** A long-running MPC's CVXPY/CLARABEL cache carries
  its own solve history; a shadow clone cannot be bitwise equal to it (~1e-8).
  Branch from a snapshot with `CounterfactualPredictor.sync(fresh_solvers=True)`
  for every branch and report the difference to the original continuation as
  the numerical repeatability.
- **`tau_min_feasible` is not observed.** It is computed by G* inside
  `apply_intent`; state rules and policies must not read it.
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

T1 line (current; commands in `docs/current/REPRODUCE.md`):

| Purpose | Entry |
|---|---|
| Task profiles | `env/t1_env.py`, `env/t1_l_task.py`, `env/t1_o_task.py` (`p1` / `p2` / `o`) |
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
