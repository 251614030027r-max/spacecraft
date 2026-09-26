"""M2: collect value-regression data for one trained V3 run.

For every seed (default block 270000-270047, disjoint from evaluation
262000-262047 and calibration 262100-262111) four episodes:

    baseline_full   Pure MPC from reset to the end           -> V_B data
    learned_full    the deterministic policy to the end      -> V_L data
    probe x 2       policy for k decisions, then Pure MPC    -> V_B data on states
                    (controller.reset at the handoff)           the policy visits

k is drawn uniformly from [0, len(learned_full)) with a generator seeded by
(--probe-seed, seed), so a seed's probes do not depend on how the block is
split across processes. Each process writes one .npz (per-decision arrays)
and one .json (per-episode metadata); M3 takes any number of them.

    python -B -m experiments.v3_collect_value_data --run-dir logs/v3d_262420 --seeds 270000-270011 --output eval/v3d/m2_262420_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.v3_common import (
    always,
    learned_then_baseline,
    load_manifest,
    load_policy,
    make_env_for_run,
    parse_seed_range,
    run_episode,
)

KINDS = {"baseline_full": 0, "learned_full": 1, "probe": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="final_model.zip")
    parser.add_argument("--seeds", default="270000-270047")
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--probe-seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, required=True, help="path stem; writes .npz and .json")
    parser.add_argument("--allow-incomplete-run", action="store_true", help="smoke tests only")
    args = parser.parse_args()
    npz_path = args.output.with_suffix(".npz")
    json_path = args.output.with_suffix(".json")
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.run_dir, require_completed=not args.allow_incomplete_run)
    policy = load_policy(args.run_dir, args.model_name)
    env = make_env_for_run(manifest)
    slices = {name: [s.start, s.stop] for name, s in env.policy_observation_slices().items()}

    columns: dict[str, list] = {
        "observation": [],
        "reward": [],
        "learned_branch": [],
        "episode": [],
        "decision": [],
    }
    episodes: list[dict] = []
    started = perf_counter()
    try:
        for seed in parse_seed_range(args.seeds):
            learned = run_episode(env, policy, seed, always("learned"))
            rng = np.random.default_rng([args.probe_seed, seed])
            ks = [int(k) for k in rng.integers(0, len(learned["rewards"]), size=args.probes)]
            plan = [("baseline_full", None, always("baseline")), ("learned_full", None, None)]
            plan += [("probe", k, learned_then_baseline(k)) for k in ks]
            for kind, k, schedule in plan:
                result = learned if kind == "learned_full" else run_episode(env, policy, seed, schedule)
                index = len(episodes)
                n = len(result["rewards"])
                columns["observation"].append(result["observations"])
                columns["reward"].append(result["rewards"])
                columns["learned_branch"].append(
                    np.array([b == "learned" for b in result["branches"]], dtype=bool)
                )
                columns["episode"].append(np.full(n, index, dtype=np.int32))
                columns["decision"].append(np.arange(n, dtype=np.int32))
                episodes.append(
                    {
                        "index": index,
                        "seed": seed,
                        "kind": kind,
                        "handback_decision": k,
                        "decisions": n,
                        "completed": result["completed"],
                        "terminated": result["terminated"],
                        "truncated": result["truncated"],
                        "failure": result["failure"],
                        "qp_zero_fallbacks": result["qp_zero_fallbacks"],
                        "undiscounted_return": float(result["rewards"].sum()),
                    }
                )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "learned_decisions": len(learned["rewards"]),
                        "probes": ks,
                        "elapsed_s": round(perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    finally:
        env.close()
    np.savez_compressed(
        npz_path,
        **{
            name: np.concatenate(values) if values else np.zeros(0)
            for name, values in columns.items()
        },
    )
    json_path.write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "model_name": args.model_name,
                "code_commit": manifest.get("code_commit"),
                "seeds": args.seeds,
                "probes": args.probes,
                "probe_seed": args.probe_seed,
                "observation_slices": slices,
                "kinds": KINDS,
                "episodes": episodes,
                "wall_clock_s": perf_counter() - started,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
