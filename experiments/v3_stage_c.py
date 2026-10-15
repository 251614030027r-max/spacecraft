"""Stage C: learn when handing off to Pure MPC is safe, then test it closed-loop.

The coordinator decides, at every decision on the learned branch, whether to
hand off to Pure MPC now (one-way). Stage B measured, on the frozen 60k
policies, the exact outcome of a handoff at every decision of every learned
failure (B1) and every 10th decision of every learned success (B2-lite). The
estimated object here is

    p_M(s) = P(Pure MPC taking over from s ends in a clean completion | s)

which depends on the physical state and Pure MPC only -- not on which learned
policy brought the vehicle there -- so its input is restricted to the
policy-independent observation blocks (``FEATURE_BLOCKS``) and its labels pool
all three policies. The online rule is: hand off at the first decision with
p_M(s) >= tau; otherwise continue the learned policy.

Subcommands (the reading rules are in
``docs/STAGE_C_PREREGISTRATION_20261002.md``; nothing here is tuned after
seeing a result):

    record    replay a policy's learned episodes on the B block and store the
              observation at every decision; each episode is checked bitwise
              against the B1 rerun of the same seed
    fit       grouped cross-validation (folds by initial seed), the frozen
              tau rule, the out-of-fold first-trigger analysis, and the final
              classifier trained on all labelled states
    evaluate  closed loop on a fresh block: rows pure / learned / hybrid
    readout   per-model counts and the preregistered verdict
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from experiments.v3_common import load_manifest, load_policy, make_env_for_run, parse_seed_range
from experiments.v3_handoff_scan import ImpulseMeter, _outcome
from train.mainline import mainline_v3e_configs
from train.train_hybrid import _code_provenance
from train.v3_values import sha256_file

#: Policy-independent blocks: the full physical state, the target attitude
#: (phase) and the remaining time. Masked: the learned branch's private state
#: (task_state, applied_direction), the previous decision's actuator feedback
#: (depends on which controller flew it) and the episode-frozen staging
#: direction (identifies the episode, does not drive Pure MPC).
FEATURE_BLOCKS = ("core", "target_attitude", "remaining_time")
FEATURE_SIZES = {"core": 24, "target_attitude": 6, "remaining_time": 1}

N_FOLDS = 6
ENSEMBLE_SEEDS = (0, 1, 2, 3, 4)
HIDDEN = 64
EPOCHS = 400
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TAU_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
TARGET_PRECISION = 0.95

B_BLOCK = "262000-262047"
C3_BLOCK = "266000-266047"
MAX_DESTROY = 2
MIN_PASSING_MODELS = 2
MIN_SHARED_FOR_EFFICIENCY = 5


# -- features and labels ---------------------------------------------------------


def feature_index(slices: dict[str, slice]) -> np.ndarray:
    parts = []
    for name in FEATURE_BLOCKS:
        block = slices[name]
        if block.stop - block.start != FEATURE_SIZES[name]:
            raise ValueError(f"observation block {name} has an unexpected size")
        parts.append(np.arange(block.start, block.stop))
    return np.concatenate(parts)


def labels_from_scans(b1: dict[str, Any] | None, b2: dict[str, Any] | None) -> dict[int, bool]:
    """k -> clean completion of a handoff at decision k (B1 every k, B2 every 10th)."""

    labels: dict[int, bool] = {}
    for scan in (b1, b2):
        if scan is None or not scan.get("scanned"):
            continue
        for row in scan["handoffs"]:
            labels[int(row["k"])] = bool(row["clean_completion"])
    return labels


def fold_of(seed: int) -> int:
    return (int(seed) - 262000) % N_FOLDS


def build_dataset(
    records: dict[str, dict[str, np.ndarray]],
    b1: dict[str, dict[int, dict[str, Any]]],
    b2: dict[str, dict[int, dict[str, Any]]],
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    """One row per labelled (model, seed, k); each trajectory's rows sum to weight 1."""

    rows: dict[str, list] = {"x": [], "y": [], "w": [], "seed": [], "k": [], "model": [], "source": []}
    for model, record in records.items():
        for seed in sorted(set(int(s) for s in record["seed"])):
            mask = record["seed"] == seed
            obs = record["obs"][mask]
            ks = record["k"][mask]
            b1_scan = b1[model].get(seed)
            b2_scan = b2[model].get(seed)
            labels = labels_from_scans(b1_scan, b2_scan)
            if not labels:
                continue
            source = "B1" if b1_scan is not None and b1_scan.get("scanned") else "B2"
            by_k = {int(k): i for i, k in enumerate(ks)}
            weight = 1.0 / len(labels)
            for k, label in sorted(labels.items()):
                rows["x"].append(obs[by_k[k]][features])
                rows["y"].append(float(label))
                rows["w"].append(weight)
                rows["seed"].append(seed)
                rows["k"].append(k)
                rows["model"].append(model)
                rows["source"].append(source)
    return {
        "x": np.asarray(rows["x"], dtype=np.float32),
        "y": np.asarray(rows["y"], dtype=np.float32),
        "w": np.asarray(rows["w"], dtype=np.float32),
        "seed": np.asarray(rows["seed"], dtype=np.int64),
        "k": np.asarray(rows["k"], dtype=np.int64),
        "model": np.asarray(rows["model"]),
        "source": np.asarray(rows["source"]),
    }


# -- classifier ------------------------------------------------------------------


class HandoffClassifier:
    """Ensemble of small MLPs on standardised policy-independent features."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, states: list[dict], tau: float | None,
                 features: np.ndarray) -> None:
        self.mean, self.std, self.states, self.tau = mean, std, states, tau
        self.features = np.asarray(features, dtype=np.int64)
        self._nets = [self._net(state) for state in states]

    @staticmethod
    def _make(dimension: int):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(dimension, HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN, 1),
        )

    def _net(self, state: dict):
        net = self._make(len(self.mean))
        net.load_state_dict(state)
        net.eval()
        return net

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, w: np.ndarray, features: np.ndarray,
            epochs: int = EPOCHS) -> "HandoffClassifier":
        import torch

        torch.set_num_threads(1)
        mean = x.mean(axis=0)
        std = x.std(axis=0) + 1e-6
        xt = torch.as_tensor((x - mean) / std, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        wt = torch.as_tensor(w / w.sum(), dtype=torch.float32)
        states = []
        for seed in ENSEMBLE_SEEDS:
            torch.manual_seed(seed)
            net = cls._make(x.shape[1])
            optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            for _ in range(epochs):
                optimiser.zero_grad()
                logits = net(xt).squeeze(-1)
                loss = (wt * torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, yt, reduction="none")).sum()
                loss.backward()
                optimiser.step()
            states.append({k: v.detach().clone() for k, v in net.state_dict().items()})
        return cls(mean, std, states, None, features)

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        import torch

        xt = torch.as_tensor((np.atleast_2d(x) - self.mean) / self.std, dtype=torch.float32)
        with torch.no_grad():
            p = [torch.sigmoid(net(xt).squeeze(-1)).numpy() for net in self._nets]
        return np.mean(p, axis=0)

    def p_handoff(self, observation: np.ndarray) -> float:
        return float(self.predict_features(np.asarray(observation)[self.features])[0])

    def save(self, path: Path, meta: dict[str, Any]) -> None:
        import torch

        torch.save({"mean": self.mean, "std": self.std, "states": self.states, "tau": self.tau,
                    "features": self.features, "meta": meta}, path)

    @classmethod
    def load(cls, path: Path) -> "HandoffClassifier":
        import torch

        blob = torch.load(path, weights_only=False)
        return cls(blob["mean"], blob["std"], blob["states"], blob["tau"], blob["features"])


# -- tau and the first-trigger analysis ----------------------------------------------


def choose_tau(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float | None, list[dict]]:
    """Smallest tau on the grid whose trajectory-weighted precision >= TARGET_PRECISION."""

    curve = []
    chosen = None
    for tau in TAU_GRID:
        hit = p >= tau
        mass = float(w[hit].sum())
        precision = float(w[hit & (y > 0.5)].sum() / mass) if mass > 0 else None
        curve.append({"tau": tau, "weighted_precision": precision, "weighted_coverage": mass / float(w.sum())})
        if chosen is None and precision is not None and precision >= TARGET_PRECISION:
            chosen = tau
    return chosen, curve


def first_trigger(p_all: np.ndarray, ks_all: np.ndarray, labels: dict[int, bool], tau: float) -> dict[str, Any]:
    """Online rule on one recorded trajectory: hand off at the first k with p >= tau."""

    order = np.argsort(ks_all)
    for i in order:
        if p_all[i] >= tau:
            k = int(ks_all[i])
            return {"k": k, "label": labels.get(k)}  # None: k not on the labelled grid
    return {"k": None, "label": None}


def trigger_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """B1 failures: rescued / premature (handoff fails) / never. B2 successes:
    kept by never triggering, triggered on the grid (clean or destroying), or
    triggered off the grid (outcome unknown)."""

    out: dict[str, Any] = {}
    for source in ("B1", "B2"):
        sub = [r for r in rows if r["source"] == source]
        trig = [r for r in sub if r["k"] is not None]
        out[source] = {
            "trajectories": len(sub),
            "never_triggered": len(sub) - len(trig),
            "triggered_clean": sum(r["label"] is True for r in trig),
            "triggered_fails": sum(r["label"] is False for r in trig),
            "triggered_off_grid": sum(r["label"] is None for r in trig),
            "trigger_k": sorted(r["k"] for r in trig),
        }
    return out


# -- closed loop ------------------------------------------------------------------------


def run_closed_loop(
    env, policy, seed: int, row: str, classifier: HandoffClassifier | None,
    meter: ImpulseMeter, max_decisions: int | None = None,
) -> dict[str, Any]:
    """One episode of a row: 'pure' (baseline from reset), 'learned', or 'hybrid'
    (learned until the first p >= tau, then Pure MPC to the end)."""

    observation, _ = env.reset(seed=int(seed))
    meter.force_impulse_n_s = 0.0
    zero = np.zeros(env.action_space.shape, dtype=np.float64)
    branch = "baseline" if row == "pure" else "learned"
    handoff_k = 0 if row == "pure" else None
    p_trace: list[float] = []
    info: dict[str, Any] = {}
    decision = 0
    learned_decisions = 0
    while max_decisions is None or decision < max_decisions:
        if row == "hybrid" and branch == "learned":
            assert classifier is not None and classifier.tau is not None
            p = classifier.p_handoff(observation)
            p_trace.append(p)
            if p >= classifier.tau:
                branch, handoff_k = "baseline", decision
        if branch == "learned":
            action, _ = policy.predict(observation, deterministic=True)
            learned_decisions += 1
        else:
            action = zero
        observation, _, terminated, truncated, info = env.step_with_branch(
            np.asarray(action, dtype=np.float64), branch=branch
        )
        decision += 1
        if terminated or truncated:
            break
    return {
        "seed": int(seed),
        "row": row,
        **_outcome(env, info),
        "decisions": decision,
        "handoff_k": handoff_k,
        "learned_decisions": learned_decisions,
        "equivalent_delta_v_m_s": meter.force_impulse_n_s / float(env.env.chaser_parameters.mass),
        "p_trace": p_trace,
    }


# -- readout --------------------------------------------------------------------------


def model_verdict(pure: dict[int, dict], learned: dict[int, dict], hybrid: dict[int, dict]) -> dict[str, Any]:
    seeds = sorted(pure)
    clean = lambda rows, s: bool(rows[s]["clean_completion"])
    violated = lambda rows, s: not bool(rows[s]["zero_violation"])
    n_pure = sum(clean(pure, s) for s in seeds)
    n_learned = sum(clean(learned, s) for s in seeds)
    n_hybrid = sum(clean(hybrid, s) for s in seeds)
    v_pure = sum(violated(pure, s) for s in seeds)
    v_hybrid = sum(violated(hybrid, s) for s in seeds)
    destroy = [s for s in seeds if clean(pure, s) and not clean(hybrid, s)]
    rescue = [s for s in seeds if not clean(pure, s) and clean(hybrid, s)]
    shared = [s for s in seeds if clean(pure, s) and clean(learned, s) and clean(hybrid, s)]

    def median_excess(rows: dict[int, dict], key: str) -> float | None:
        if not shared:
            return None
        return float(np.median([rows[s][key] - pure[s][key] for s in shared]))

    excess = {
        name: {"time_s": median_excess(rows, "survival_s"), "delta_v_m_s": median_excess(rows, "equivalent_delta_v_m_s")}
        for name, rows in (("learned", learned), ("hybrid", hybrid))
    }
    efficiency_ok = None
    if len(shared) >= MIN_SHARED_FOR_EFFICIENCY:
        efficiency_ok = all(
            excess["hybrid"][key] <= max(0.0, 0.5 * excess["learned"][key])
            for key in ("time_s", "delta_v_m_s")
        )
    handoffs = [hybrid[s]["handoff_k"] for s in seeds if hybrid[s]["handoff_k"] is not None]
    return {
        "clean_completions": {"pure": n_pure, "learned": n_learned, "hybrid": n_hybrid},
        "violation_episodes": {"pure": v_pure, "learned": sum(violated(learned, s) for s in seeds), "hybrid": v_hybrid},
        "rescued": rescue,
        "destroyed": destroy,
        "shared_clean_seeds": len(shared),
        "median_excess_over_pure_on_shared": excess,
        "efficiency_acceptable": efficiency_ok,
        "handoff_k": sorted(handoffs),
        "never_handed_off": len(seeds) - len(handoffs),
        "learned_share": float(np.mean([hybrid[s]["learned_decisions"] / max(hybrid[s]["decisions"], 1) for s in seeds])),
        "passes": n_hybrid > n_pure and v_hybrid <= v_pure and len(destroy) <= MAX_DESTROY,
    }


def overall_verdict(models: dict[str, dict[str, Any]], fidelity_ok: bool) -> str:
    if not fidelity_ok:
        return "FIDELITY_FAIL"
    passing = [m for m, v in models.items() if v["passes"]]
    if len(passing) < MIN_PASSING_MODELS:
        return "C_STOP"
    efficient = [m for m in passing if models[m]["efficiency_acceptable"]]
    return "A_GO_TO_E" if len(efficient) >= MIN_PASSING_MODELS else "B_GO_TO_D"


# -- CLI -----------------------------------------------------------------------------------


def _pairs(values: list[str]) -> dict[str, str]:
    out = {}
    for value in values:
        name, _, path = value.partition("=")
        if not path:
            raise ValueError(f"expected MODEL=PATH, got {value!r}")
        out[name] = path
    return out


def _scans(directory: str | None) -> dict[int, dict[str, Any]]:
    if directory is None:
        return {}
    return {int(p.stem.split("_")[1]): json.loads(p.read_text()) for p in sorted(Path(directory).glob("seed_*.json"))}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    os.replace(temporary, path)


def _env_for(run_dir: Path, allow_incomplete: bool):
    manifest = load_manifest(run_dir, require_completed=not allow_incomplete)
    env = make_env_for_run(manifest)
    environment, hybrid = mainline_v3e_configs()
    if asdict(env.hybrid_config) != asdict(hybrid) or asdict(env.environment_config) != asdict(environment):
        raise ValueError("this run is not the canonical V3e mainline configuration")
    return manifest, env


def cmd_record(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, env = _env_for(args.run_dir, args.allow_incomplete_run)
    policy = load_policy(args.run_dir, args.model_name)
    b1 = _scans(args.b1_dir)
    slices = {name: [s.start, s.stop] for name, s in env.policy_observation_slices().items()}
    obs, seeds, ks = [], [], []
    try:
        for seed in parse_seed_range(args.seeds):
            observation, _ = env.reset(seed=seed)
            rewards: list[float] = []
            decision = 0
            while args.max_decisions is None or decision < args.max_decisions:
                obs.append(np.asarray(observation, dtype=np.float32).copy())
                seeds.append(seed)
                ks.append(decision)
                action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step_with_branch(
                    np.asarray(action, dtype=np.float64), branch="learned")
                rewards.append(float(info["hybrid_integrated_reward_without_shaping"]))
                decision += 1
                if terminated or truncated:
                    break
            reference = b1.get(seed)
            if reference is None:
                raise RuntimeError(f"seed {seed}: no B1 file to check the replay against")
            if rewards != reference["learned_full"]["task_rewards"] or decision != reference["learned_full"]["decisions"]:
                raise RuntimeError(f"seed {seed}: replay differs from the B1 learned episode")
            print(f"seed {seed}: {decision} decisions, matches B1", flush=True)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, obs=np.stack(obs), seed=np.asarray(seeds), k=np.asarray(ks))
    _write_json(args.output.with_suffix(".json"), {
        "run_dir": str(args.run_dir), "model_sha256": sha256_file(args.run_dir / args.model_name),
        "seeds": args.seeds, "rows": len(seeds), "observation_slices": slices,
        "npz_sha256": sha256_file(args.output), **_code_provenance(),
    })


def cmd_fit(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "c2_report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    records = {m: dict(np.load(p)) for m, p in _pairs(args.record).items()}
    meta = {m: json.loads(Path(p).with_suffix(".json").read_text()) for m, p in _pairs(args.record).items()}
    slices = {name: slice(*bounds) for name, bounds in next(iter(meta.values()))["observation_slices"].items()}
    features = feature_index(slices)
    b1 = {m: _scans(d) for m, d in _pairs(args.b1).items()}
    b2 = {m: _scans(d) for m, d in _pairs(args.b2).items()}
    data = build_dataset(records, b1, b2, features)
    folds = np.array([fold_of(s) for s in data["seed"]])

    oof = np.zeros(len(data["y"]), dtype=np.float64)
    fold_models = {}
    for fold in range(N_FOLDS):
        train = folds != fold
        model = HandoffClassifier.fit(data["x"][train], data["y"][train], data["w"][train], features, args.epochs)
        oof[folds == fold] = model.predict_features(data["x"][folds == fold])
        fold_models[fold] = model
        print(f"fold {fold}: {int((folds == fold).sum())} held-out states", flush=True)
    tau, curve = choose_tau(oof, data["y"], data["w"])

    trigger_rows = []
    if tau is not None:
        for model_name, record in records.items():
            for seed in sorted(set(int(s) for s in record["seed"])):
                labels = labels_from_scans(b1[model_name].get(seed), b2[model_name].get(seed))
                if not labels:
                    continue
                mask = record["seed"] == seed
                p_all = fold_models[fold_of(seed)].predict_features(record["obs"][mask][:, features])
                hit = first_trigger(p_all, record["k"][mask], labels, tau)
                source = "B1" if b1[model_name].get(seed, {}).get("scanned") else "B2"
                trigger_rows.append({"model": model_name, "seed": seed, "source": source, **hit})

    weighted_auc = None
    try:
        pos, neg = data["y"] > 0.5, data["y"] < 0.5
        diff = oof[pos][:, None] - oof[neg][None, :]
        pair_w = data["w"][pos][:, None] * data["w"][neg][None, :]
        weighted_auc = float(((diff > 0) * pair_w + 0.5 * (diff == 0) * pair_w).sum() / pair_w.sum())
    except (ValueError, ZeroDivisionError):
        pass

    report: dict[str, Any] = {
        "preregistration": "docs/STAGE_C_PREREGISTRATION_20261002.md",
        "feature_blocks": list(FEATURE_BLOCKS),
        "constants": {"n_folds": N_FOLDS, "ensemble_seeds": list(ENSEMBLE_SEEDS), "hidden": HIDDEN,
                      "epochs": args.epochs, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                      "tau_grid": list(TAU_GRID), "target_precision": TARGET_PRECISION},
        "labelled_states": int(len(data["y"])),
        "labelled_trajectories": int(len(set(zip(data["model"].tolist(), data["seed"].tolist())))),
        "positive_fraction_weighted": float((data["w"] * data["y"]).sum() / data["w"].sum()),
        "oof_weighted_auc": weighted_auc,
        "precision_curve": curve,
        "tau": tau,
        "c2_verdict": "TAU_FOUND" if tau is not None else "C_STOP",
        "first_trigger_oof": {m: trigger_summary([r for r in trigger_rows if r["model"] == m]) for m in records},
        "first_trigger_rows": trigger_rows,
        **_code_provenance(),
    }
    if tau is not None:
        final = HandoffClassifier.fit(data["x"], data["y"], data["w"], features, args.epochs)
        final.tau = tau
        classifier_path = args.output_dir / "handoff_classifier.pt"
        final.save(classifier_path, {"report": "c2_report.json", "constants": report["constants"]})
        report["classifier_sha256"] = sha256_file(classifier_path)
    _write_json(report_path, report)
    print(json.dumps({k: report[k] for k in ("c2_verdict", "tau", "oof_weighted_auc", "labelled_states")}, indent=1))


def cmd_evaluate(args: argparse.Namespace) -> None:
    manifest, env = _env_for(args.run_dir, args.allow_incomplete_run)
    policy = None if args.row == "pure" else load_policy(args.run_dir, args.model_name)
    classifier = HandoffClassifier.load(args.classifier) if args.row == "hybrid" else None
    if args.row == "hybrid" and classifier.tau is None:
        raise ValueError("the classifier carries no tau (C2 did not pass)")
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    provenance = {
        "run_dir": str(args.run_dir), "row": args.row,
        "model_sha256": None if policy is None else sha256_file(args.run_dir / args.model_name),
        "classifier_sha256": None if classifier is None else sha256_file(args.classifier),
        "max_decisions": args.max_decisions, **_code_provenance(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for seed in parse_seed_range(args.seeds):
            path = args.output_dir / f"seed_{seed}.json"
            if path.exists():
                print(f"skip {path} (exists)", flush=True)
                continue
            lock = path.with_suffix(".lock")
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                print(f"skip {path} (claimed)", flush=True)
                continue
            started = perf_counter()
            result = run_closed_loop(env, policy, seed, args.row, classifier, meter, args.max_decisions)
            result.update(provenance, wall_clock_s=perf_counter() - started)
            _write_json(path, result)
            lock.unlink()
            print(f"seed {seed}: done", flush=True)
    finally:
        env.close()


def cmd_readout(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = parse_seed_range(args.seeds)
    pure = _scans(args.pure)
    learned = {m: _scans(d) for m, d in _pairs(args.learned).items()}
    hybrid = {m: _scans(d) for m, d in _pairs(args.hybrid).items()}
    problems = []

    def check(rows: dict[int, dict], name: str, row: str) -> None:
        missing = sorted(set(expected) - set(rows))
        if missing:
            problems.append(f"{name}: missing seeds {missing}")
        commits = {r.get("code_commit") for r in rows.values()}
        if len(commits) != 1 or None in commits:
            problems.append(f"{name}: {len(commits)} code commits")
        if any(r.get("code_dirty") for r in rows.values()):
            problems.append(f"{name}: dirty checkout")
        if any(r.get("row") != row or r.get("max_decisions") is not None for r in rows.values()):
            problems.append(f"{name}: wrong row or truncated")
        if row == "hybrid" and len({r.get("classifier_sha256") for r in rows.values()}) != 1:
            problems.append(f"{name}: more than one classifier")

    check(pure, "pure", "pure")
    for m in learned:
        check(learned[m], f"learned {m}", "learned")
        check(hybrid[m], f"hybrid {m}", "hybrid")
        if {r.get("model_sha256") for r in learned[m].values()} != {r.get("model_sha256") for r in hybrid[m].values()}:
            problems.append(f"{m}: learned and hybrid rows use different model files")
    models = {} if problems else {
        m: model_verdict({s: pure[s] for s in expected}, {s: learned[m][s] for s in expected},
                         {s: hybrid[m][s] for s in expected})
        for m in learned
    }
    result = {
        "preregistration": "docs/STAGE_C_PREREGISTRATION_20261002.md",
        "seeds": args.seeds, "problems": problems,
        "constants": {"max_destroy": MAX_DESTROY, "min_passing_models": MIN_PASSING_MODELS,
                      "min_shared_for_efficiency": MIN_SHARED_FOR_EFFICIENCY},
        "verdict": overall_verdict(models, not problems),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"verdict": result["verdict"], "problems": len(problems),
                      "models": {m: {k: v[k] for k in ("clean_completions", "violation_episodes", "passes", "efficiency_acceptable")}
                                 for m, v in models.items()}}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-name", default="final_model.zip")
    p.add_argument("--seeds", default=B_BLOCK)
    p.add_argument("--b1-dir", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=None, help="tests only")
    p.add_argument("--allow-incomplete-run", action="store_true")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("fit")
    p.add_argument("--record", action="append", required=True, help="MODEL=c1_<model>.npz")
    p.add_argument("--b1", action="append", required=True, help="MODEL=B1 scan dir")
    p.add_argument("--b2", action="append", required=True, help="MODEL=B2-lite scan dir")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=EPOCHS, help="tests only")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("evaluate")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-name", default="final_model.zip")
    p.add_argument("--row", choices=("pure", "learned", "hybrid"), required=True)
    p.add_argument("--classifier", type=Path, default=None)
    p.add_argument("--seeds", default=C3_BLOCK)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-decisions", type=int, default=None, help="tests only")
    p.add_argument("--allow-incomplete-run", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("readout")
    p.add_argument("--pure", required=True)
    p.add_argument("--learned", action="append", required=True, help="MODEL=dir")
    p.add_argument("--hybrid", action="append", required=True, help="MODEL=dir")
    p.add_argument("--seeds", default=C3_BLOCK)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=cmd_readout)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
