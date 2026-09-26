"""M3: fit V_L and V_B for one run from its M2 files.

    G_t = sum_{j>=t} 0.99^(j-t) r_j   (decision level, to the end of the episode)

    V_L  <- every state of the learned_full episodes
    V_B  <- every state of the baseline_full episodes
            + probe states at and after the handoff (the learned prefix of a
              probe is NOT a V_L sample: its return includes the Pure MPC tail)

Seeds 270040-270047 (``--holdout``) are held out and only used to report MAE
and R^2; the ensembles are fitted on the rest. Writes ``values_L.pt``,
``values_B.pt`` and ``m3_report.json`` (with the SHA-256 of both value files)
into ``--output-dir``.

    python -B -m experiments.v3_fit_values --data eval/v3d/m2_262420_*.npz --output-dir eval/v3d/values_262420
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train.v3_values import (
    DECISION_GAMMA,
    ValueEnsemble,
    baseline_mask,
    discounted_returns,
    fit_ensemble,
)


def load_data(paths: list[Path]) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    """Concatenate M2 files, renumbering episodes globally."""

    columns: dict[str, list[np.ndarray]] = {}
    episodes: list[dict] = []
    slices = None
    for path in sorted(paths):
        data = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        if slices is None:
            slices = meta["observation_slices"]
        elif slices != meta["observation_slices"]:
            raise ValueError("M2 files disagree on the observation layout")
        offset = len(episodes)
        for episode in meta["episodes"]:
            episodes.append({**episode, "index": episode["index"] + offset, "source": path.name})
        for name in data.files:
            values = data[name]
            if name == "episode":
                values = values + offset
            columns.setdefault(name, []).append(values)
    assert slices is not None
    return {k: np.concatenate(v) for k, v in columns.items()}, episodes, slices


def returns_to_go(data: dict[str, np.ndarray], episodes: list[dict]) -> np.ndarray:
    out = np.zeros(data["reward"].size, dtype=np.float64)
    for episode in episodes:
        index = np.flatnonzero(data["episode"] == episode["index"])
        order = index[np.argsort(data["decision"][index])]
        out[order] = discounted_returns(data["reward"][order], DECISION_GAMMA)
    return out


def select(data: dict[str, np.ndarray], episodes: list[dict], which: str) -> np.ndarray:
    kind = {e["index"]: e["kind"] for e in episodes}
    per_state_kind = np.array([kind[int(i)] for i in data["episode"]])
    if which == "L":
        return per_state_kind == "learned_full"
    if which == "B":
        return (per_state_kind == "baseline_full") | (
            (per_state_kind == "probe") & ~data["learned_branch"]
        )
    raise ValueError(which)


def score(model: ValueEnsemble, x: np.ndarray, y: np.ndarray) -> dict:
    if y.size == 0:
        return {"states": 0}
    heads = model.head_values(x)
    mu = heads.mean(axis=0)
    sd = heads.std(axis=0)
    error = mu - y
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "states": int(y.size),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "mean_sd": float(np.mean(sd)),
        "corr_sd_abs_error": float(np.corrcoef(sd, np.abs(error))[0, 1]) if y.size > 2 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="M2 .npz files (the .json next to each is read too)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout", default="270040-270047")
    parser.add_argument("--fit-seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from experiments.v3_common import parse_seed_range

    data, episodes, slices_list = load_data(args.data)
    dimension = int(data["observation"].shape[1])
    slices = {name: slice(a, b) for name, (a, b) in slices_list.items()}
    returns = returns_to_go(data, episodes)
    seed_of = {e["index"]: e["seed"] for e in episodes}
    state_seed = np.array([seed_of[int(i)] for i in data["episode"]])
    holdout = np.isin(state_seed, parse_seed_range(args.holdout))

    report: dict = {
        "data": [str(p) for p in args.data],
        "gamma": DECISION_GAMMA,
        "holdout_seeds": args.holdout,
        "episodes": len(episodes),
        "states": int(returns.size),
    }
    masks = {"L": np.ones(dimension, dtype=np.float32), "B": baseline_mask(slices, dimension)}
    for offset, which in enumerate(("L", "B")):
        chosen = select(data, episodes, which)
        train = chosen & ~holdout
        test = chosen & holdout
        model = fit_ensemble(
            data["observation"][train],
            returns[train],
            data["episode"][train],
            input_mask=masks[which],
            seed=args.fit_seed + offset,
        )
        model.meta.update({"value": which, "gamma": DECISION_GAMMA, "observation_slices": slices_list})
        sha = model.save(args.output_dir / f"values_{which}.pt")
        report[f"V_{which}"] = {
            "file": str(args.output_dir / f"values_{which}.pt"),
            "sha256": sha,
            "train": score(model, data["observation"][train], returns[train]),
            "holdout": score(model, data["observation"][test], returns[test]),
            "heads": model.meta["heads"],
            "masked_blocks": [n for n in slices if masks[which][slices[n]].sum() == 0],
        }
    (args.output_dir / "m3_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k.startswith("V_")}, indent=1, default=str))


if __name__ == "__main__":
    main()
