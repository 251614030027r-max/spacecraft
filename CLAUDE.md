# Working notes for this repository

High-fidelity 6-DoF SE(3) **pre-capture** of a fast-tumbling non-cooperative target (chaser 106 kg, +-2 N / +-0.6 N*m
per axis; target 225 kg box, spin 2.1-2.7 deg/s, unknown inertia; 500 km / 45 deg orbit, RK45 truth with central
gravity and J2, 0.1 s control).

**Method (round 2, `docs/final_method/ROUND2.md`):** ECC with RWG. SAC gives a 2-D stage intent every 16 s in the
target's spin frame (closing speed, along-sweep speed; latitude regulated onto the port's sweep cone); a stage planner and
a far-field tracking law fly it. The coordination keeps the intent when its own mode prediction reaches an entry,
otherwise flies the critical opportunity of least change (nearest grid intent at a last-chance window, else the direct
entry at the last usable pass); the frozen planners and MPC fly entry, descent and capture. Standard SAC (actor and
critics read the same 17-value observation; the first draft's critic inputs z are removed, DEAD_ENDS D46); single
time-limited task (deadline U(140, 200) s, 3.0 m/s); training potential = opportunity-reachability potential.
Configurations `ECC`, `ECC_NO_CORE`, `SAC_MPC`, `TD3_MPC`.

**State (2026-11-19):** first draft done (`evidence/PAPER_READOUT/`, paper `docs/paper/latex/`, ECC_LE 251/288; that
code is science `72a8d41`). Round 2: repository cleaned, method consolidated, potential implemented and adopted
without a separate check (user 2026-11-20); next the feedback-mechanism development round (`dev` stage, seeds 0-2,
3000 decisions, openings 295300-295395; design to be discussed), then the formal design. History: `docs/HISTORY.md`; refuted directions:
`docs/DEAD_ENDS.md`; live state: collab `PROJECT_STATE.md` (wins over anything here).

## Start here

1. Collab branch `collab/spacecraft`: `docs/collaboration/PROJECT_STATE.md` and `HANDOFF.md`.
2. `README.md`, `docs/final_method/ROUND2.md`, `docs/paper/facts/`, `docs/final_method/METHOD_REGISTRY.json`,
   `docs/engineering/REPRODUCE.md`.
3. `docs/DEAD_ENDS.md` before proposing any method, interface, reward or task change.

## Roles and branches

- The research layer (user + discussion window) owns the paper question and success criteria. The upper window reviews,
  writes run orders and code; it may question but not redefine them.
- The lower window runs long jobs on the user's machine in the science checkout `D:/py/DRL2`; it never switches branches
  or pulls in a working tree while a job runs.
- **Exactly three branches:** `claude/sac-mpc-coupling-design-ns7g6i` (science code and evidence), `collab/spacecraft`
  (only `docs/collaboration/`; fetch first, never force-push), `main` (merged only when the experiments are over).
  Frozen states are commits and `archive/*` tags (made by the lower window; this sandbox cannot push tags). No PRs
  unless the user asks.
- **Git holds facts, not discussion.** Code, executed run orders, results, evidence, registry, hand-offs and reviews of
  finished results go in git; proposals enter git only as authorised run orders.

## Non-negotiables

- **Truth RK45 geometry is the only arbiter of safety**, never observation, QP "solved" or predicted margins.
- **The executor is frozen**: MPC, planners, entry rule, physics (`env/t1_*`, `env/se3_rendezvous_env.py`,
  `controllers/`, `dynamics/`) behave bitwise as at `e0a61753`. The only source changes are `t1_env.py`'s read-only
  audit and the bitwise-neutral speed-ups of 2026-11-13 (`dynamics/fastmath.py`, `controllers/mpc/cvxpy_fastpath.py`
  and their call sites; `docs/engineering/AUDIT.md` section 5). Any further speed-up must be shown bitwise equal on the
  `evidence/CLEANUP_EQUIV` harness; the round-2 coordination is shown bitwise equal to the first draft's ECC_LE
  execution on `evidence/R2_EQUIV`. Pure MPC is re-run clean on every formal block.
- **No manufactured gap**: no artificial obstacles, unmotivated noise, shortened horizon, throttled thrust, wrong model,
  or a rule that only hurts the baseline. Reward never encodes the answer.
- **Every training run starts from zero**; >= 3 seeds; report the distribution -- no swapping seeds, picking
  checkpoints or dropping failed seeds.
- **The formal block 300000-999999** is used only under an authorised formal run order; the first draft's openings
  300000-300295 and seeds 20-22 are never reused.
- Any change: spec and criteria in git first, then the implementation; frozen paths stay bitwise reproducible; judge
  pytest by its exit code; one interpretable factor per commit; commits carry the collaboration trailer.
- Any number entering a decision or the paper points at an artifact (JSON / CSV / manifest / commit / script) or is
  marked exploratory. Development validation != paper formal validation != flight certification.

## Traps

- **Checkpoints load by class path**: SB3 zips name `train.t2_mve_sac.StageReplay`, `T2SAC`, `T2TD3`, `TD3Replay`. Do
  not rename these modules or classes. First-draft models (48-value observation, `SplitViewPolicy`) load only at science
  `72a8d41`.
- **Multi-worker training is not bitwise repeatable** (asynchronous arrival order, `worker_order`); equivalence checks
  run one worker. Evaluation is deterministic and serial per episode; parallel timings are never a real-time claim.
- **Solver-cache numerics**: CVXPY / CLARABEL caches carry their solve history; the certification uses its own planner
  instances per episode so it never touches the executor's solver state. A fresh clone of a running solver differs
  by ~1e-8.
- **Terminal semantics**: success / unsafe / timeout / resource exhaustion are terminal (no bootstrap); only a numerical
  failure is a truncation (a non-capture outcome in the readout). An `error` record means the episode did not run.
- **Paired means are flips**: one capture / timeout flip is about +-200 on one episode; report captures, flips, medians,
  per-seed rows and both-captured openings only for time / fuel.
- **Mode prediction is not value**: a predicted event is the executor model's mode switch, not a capture; "descent"
  corrections are reported per unsafe episode (DEAD_ENDS O7).
- **Do not edit modules a running job imports**: evaluation pools spawn processes that import the current tree.
- **Cross-machine numerics**: the same commit can differ in the last bits between the sandbox and the lower's machine;
  compare verdicts, not bits, across machines.
- **Shaped reward is training only**: evaluation reports the true return (`return_true`).
- No `conftest.py`, not installed as a package: run `python -B -m pytest -q` from the repository root.

## Trusted entry points

| Purpose | Entry |
|---|---|
| Tests | `python -B -m pytest -q` (judge by the exit code) |
| Train / evaluate / screen / trajectories / completeness / resolve | `python -B -m experiments.final {train,eval,screen,traj,check,resolve}` (`docs/engineering/REPRODUCE.md`) |
| Learning curves | TensorBoard logs `<run>/tb` written by every training run |
| First-draft paper data | `python -B -m experiments.paper_readout` (reads `evidence/FORMAL/`; template of the round-2 readout) |

## Artifacts

Training artifacts (models, checkpoints, per-episode JSON) are git-ignored and archived locally by experiment ID.
Publish evidence only by force-adding individual files under `evidence/<experiment-id>/` with their SHA-256 (index:
`evidence/README.md`). Never commit an artifact directory. Do not rewrite history: cited SHAs are the evidence chain.
Everything removed from the working tree on 2026-11-12 is in commit `fcd8d2ee` (`docs/archive/PATH_MAP_20261112.csv`); the
development-period evidence removed on 2026-11-19 is in `c15ece12` (`docs/archive/PATH_MAP_20261119.csv`).

The sandbox has no numeric stack by default; build a venv from `requirements-dev.txt` (PyPI torch works;
download.pytorch.org is blocked).

## References

See `References/README.md` (what each paper is for and what not to claim).
