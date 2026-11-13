# Working notes for this repository

High-fidelity 6-DoF SE(3) **pre-capture** of a fast-tumbling non-cooperative target (chaser 106 kg, +-2 N / +-0.6 N*m
per axis; target 225 kg box, spin 2.1-2.7 deg/s, unknown inertia; 500 km / 45 deg orbit, RK45 truth with central
gravity and J2, 0.1 s control).

**Method (frozen): ECC with RWG** -- `docs/paper/METHOD.md` is the only method source. SAC gives a 2-D stage intent
every 16 s in the target's spin frame (closing speed, along-sweep speed; latitude regulated onto the port's sweep cone);
a stage planner and a far-field tracking law fly it; when the port's pass is reachable the window-committed conditional
G* plans the entry and the frozen MPC flies entry, descent and capture (RWG). ECC: the critic reads the executor feedback
z_E (6 items) during training; the executor's opportunity evaluation certifies the intent in training and execution
alike (config `ECC_CERT_LOOP` = `e_priv` + `cut_priv` + `certify`; behaviour reference science `e0a61753`).

**State (2026-11-12):** engineering wrap-up done -- the code holds only what the formal experiment runs (bitwise equal to
`fcd8d2ee` / `e0a61753` on the frozen paths), the formal design is pre-registered (`docs/paper/EXPERIMENTS.md`), the
economy missions carry the 2.0 m/s budget in training and evaluation (urgent 3.0 m/s; no separate R condition, user
2026-11-13), the formal run order `RUN_FORMAL` awaits the user's separate authorization.
History: `docs/HISTORY.md`; refuted directions: `docs/DEAD_ENDS.md`; live state: collab `PROJECT_STATE.md` (wins over
anything here).

## Start here

1. Collab branch `collab/spacecraft`: `docs/collaboration/PROJECT_STATE.md` and `HANDOFF.md`.
2. `README.md`, `docs/paper/METHOD.md`, `docs/paper/EXPERIMENTS.md`, `docs/final_method/METHOD_REGISTRY.json`,
   `docs/REPRODUCE.md`.
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
  and their call sites; `docs/paper/AUDIT.md` section 5). Any further speed-up must be shown bitwise equal on the
  `evidence/CLEANUP_EQUIV` harness. Pure MPC is re-run clean on every formal block.
- **No manufactured gap**: no artificial obstacles, unmotivated noise, shortened horizon, throttled thrust, wrong model,
  or a rule that only hurts the baseline. Reward never encodes the answer.
- **Every training run starts from zero**; >= 3 seeds; report the distribution -- no swapping seeds, picking
  checkpoints or dropping failed seeds.
- **The formal block 300000-999999 and the formal seeds 20-22** are used only under an authorised formal run order.
- Any change: spec and criteria in git first, then the implementation; frozen paths stay bitwise reproducible; judge
  pytest by its exit code; one interpretable factor per commit; commits carry the collaboration trailer.
- Any number entering a decision or the paper points at an artifact (JSON / CSV / manifest / commit / script) or is
  marked exploratory. Development validation != paper formal validation != flight certification.

## Traps

- **Old checkpoints load by class path**: SB3 zips name `train.t2_mve_sac.SplitViewPolicy`, `SliceExtractor`,
  `StageReplay`, `T2SAC`, `T2TD3`, `TD3Replay`. Do not rename these modules or classes.
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
  certifications are reported per unsafe episode (DEAD_ENDS O7).
- **Do not edit modules a running job imports**: evaluation pools spawn processes that import the current tree.
- **Cross-machine numerics**: the same commit can differ in the last bits between the sandbox and the lower's machine;
  compare verdicts, not bits, across machines.
- **Shaped reward is training only**: evaluation reports the true return (`return_true`).
- No `conftest.py`, not installed as a package: run `python -B -m pytest -q` from the repository root.

## Trusted entry points

| Purpose | Entry |
|---|---|
| Tests | `python -B -m pytest -q` (judge by the exit code) |
| Train / evaluate / screen / trajectories / geometry scan / completeness / resolve | `python -B -m experiments.final {train,eval,screen,traj,geoscan,check,resolve}` (`docs/REPRODUCE.md`) |
| Formal readout, episode table | `python -B -m experiments.formal_readout {formal,episodes,prescreen}` |
| Figures and tables | `python -B -m experiments.formal_figures` |

## Artifacts

Training artifacts (models, checkpoints, per-episode JSON) are git-ignored and archived locally by experiment ID.
Publish evidence only by force-adding individual files under `evidence/<experiment-id>/` with their SHA-256 (index:
`evidence/README.md`). Never commit an artifact directory. Do not rewrite history: cited SHAs are the evidence chain.
Everything removed from the working tree on 2026-11-12 is in commit `fcd8d2ee` (`docs/archive/PATH_MAP_20261112.csv`).

The sandbox has no numeric stack by default; build a venv from `requirements-dev.txt` (PyPI torch works;
download.pytorch.org is blocked).

## References

See `References/README.md` (what each paper is for and what not to claim).
