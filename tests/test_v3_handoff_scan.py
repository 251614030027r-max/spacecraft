"""Stage-B handoff scan: snapshot handoffs must equal the original prefix path.

The scan copies the environment at each decision boundary instead of
re-running the learned prefix from reset. These tests pin that the copy is
exact (bitwise task rewards, survival time and outcome against
``run_episode`` + ``learned_then_baseline``), that handoff at k = 0 is Pure
MPC from reset, and that the built-in verification actually catches a
non-exact handoff (a controller that is not reset).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import SAC

from experiments.v3_handoff_scan import ImpulseMeter, scan_seed
from train.mainline import MAINLINE_V3E_EVALUATION_FLAGS, mainline_v3e_env

REPOSITORY = Path(__file__).resolve().parents[1]
SEED = 263002
DECISIONS = 3


def _policy(env) -> SAC:
    return SAC("MlpPolicy", env, seed=0, device="cpu", policy_kwargs={"net_arch": [16]})


def _scan(policy, *, break_reset: bool = False) -> dict:
    env, env_b, env_verify = mainline_v3e_env(), mainline_v3e_env(), mainline_v3e_env()
    dt_s = float(env.environment_config.dt_s)
    meter, meter_b = ImpulseMeter(env.controller, dt_s), ImpulseMeter(env_b.controller, dt_s)
    if break_reset:
        env_b.controller.reset = lambda: None  # type: ignore[method-assign]
    try:
        return scan_seed(
            env, env_verify, env_b.controller, meter, meter_b, policy, SEED,
            scan="failures", stride=1, verify_prefix="0,1,2", max_decisions=DECISIONS,
        )
    finally:
        env.close()
        env_b.close()
        env_verify.close()


def test_snapshot_handoffs_equal_the_prefix_path_and_k0_is_pure_mpc() -> None:
    env = mainline_v3e_env()
    policy = _policy(env)
    result = _scan(policy)
    assert result["scanned"]  # a truncated episode is not a completion
    assert [row["k"] for row in result["handoffs"]] == [0, 1, 2]
    assert result["verified_prefix_ks"] == [0, 1, 2]

    # k = 0 is Pure MPC from reset, on an independent environment.
    observation, _ = env.reset(seed=SEED)
    rewards = []
    for _ in range(DECISIONS):
        observation, _, terminated, truncated, info = env.step_with_branch(
            np.zeros(2), branch="baseline"
        )
        rewards.append(float(info["hybrid_integrated_reward_without_shaping"]))
        assert not (terminated or truncated)
    k0 = result["handoffs"][0]
    assert k0["task_rewards_after_handoff"] == rewards
    assert k0["survival_s"] == float(env.env.time_seconds)
    env.close()

    # Bookkeeping: utility and Delta-v add the learned prefix to the MPC suffix.
    learned = result["learned_full"]
    for row in result["handoffs"]:
        k = row["k"]
        assert row["learned_task_utility_to_go"] == pytest.approx(sum(learned["task_rewards"][k:]))
        assert row["task_utility"] == pytest.approx(
            sum(learned["task_rewards"][:k]) + row["task_utility_to_go"]
        )
        assert row["equivalent_delta_v_m_s"] > 0.0


def test_verification_catches_a_handoff_that_is_not_exact() -> None:
    env = mainline_v3e_env()
    policy = _policy(env)
    env.close()
    with pytest.raises(RuntimeError, match="prefix-recompute"):
        _scan(policy, break_reset=True)


def test_cli_writes_one_resumable_file_per_seed(tmp_path: Path) -> None:
    env = mainline_v3e_env()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _policy(env).save(run_dir / "final_model.zip")
    manifest = {
        "run_name": "test_run",
        "code_commit": None,
        "status": "completed",
        "waypoint_parametrization": "task_state_v3",
        "monotone_commit": True,
        "baseline_anchored_residual": False,
        "evaluation_flags": MAINLINE_V3E_EVALUATION_FLAGS,
        "hybrid": asdict(env.hybrid_config),
        "observation_dimension": int(env.observation_space.shape[0]),
    }
    env.close()
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "scan"
    command = [
        sys.executable, "-B", "-m", "experiments.v3_handoff_scan",
        "--run-dir", str(run_dir), "--seeds", str(SEED), "--scan", "failures",
        "--verify-prefix", "last", "--max-decisions", "2", "--output-dir", str(output),
    ]
    subprocess.run(command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    written = json.loads((output / f"seed_{SEED}.json").read_text())
    assert written["schema"] == "v3_handoff_scan/1"
    assert written["verified_prefix_ks"] == [1]
    assert len(written["model_sha256"]) == 64
    assert not list(output.glob("*.lock"))
    rerun = subprocess.run(command, cwd=REPOSITORY, check=True, capture_output=True, text=True)
    assert "skip" in rerun.stdout
