"""The canonical V3e mainline entry point is the configuration the runs used.

``train.mainline`` only names the existing path. Two things pin that it
changed nothing:

1. its environment and hybrid configs equal, field for field, the three v3e
   training manifests (``eval/v3e/manifests/``, copied from the round-2
   evidence package; SHA-256 in
   ``docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md``);
2. the formal evaluator's Pure MPC row (``--control desired_pose``) and
   nominal row (``--control v3_nominal``) follow bitwise the same path as the
   mainline environment's baseline branch and full-rate learned branch: same
   policy observations, same commanded force impulse, same final state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from experiments.v3_handoff_scan import ImpulseMeter
from train.mainline import MAINLINE_V3E_EVALUATION_FLAGS, mainline_v3e_configs, mainline_v3e_env

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFESTS = sorted((REPOSITORY / "eval" / "v3e" / "manifests").glob("v3e_*_manifest.json"))
SEED = 263003
DECISIONS = 2


def _json_normalised(value):
    return json.loads(json.dumps(value))


def test_three_v3e_manifests_are_present() -> None:
    assert [path.name for path in MANIFESTS] == [
        "v3e_262420_manifest.json",
        "v3e_262421_manifest.json",
        "v3e_262422_manifest.json",
    ]


@pytest.mark.parametrize("path", MANIFESTS, ids=lambda path: path.name)
def test_mainline_equals_the_trained_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text())
    environment_config, hybrid_config = mainline_v3e_configs()
    assert manifest["status"] == "completed"
    assert manifest["evaluation_flags"] == MAINLINE_V3E_EVALUATION_FLAGS
    assert _json_normalised(asdict(hybrid_config)) == manifest["hybrid"]
    assert _json_normalised(asdict(environment_config)) == manifest["training_environment"]


@pytest.mark.parametrize(
    ("control", "branch", "action"),
    [("desired_pose", "baseline", 0.0), ("v3_nominal", "learned", 1.0)],
)
def test_evaluator_pure_and_nominal_rows_follow_the_mainline_path_bitwise(
    tmp_path: Path, control: str, branch: str, action: float
) -> None:
    output = tmp_path / f"{control}.json"
    subprocess.run(
        [
            sys.executable, "-B", "-m", "experiments.evaluate_hybrid_policy",
            "--episodes", "1", "--seed", str(SEED),
            *MAINLINE_V3E_EVALUATION_FLAGS.split(),
            "--control", control,
            "--max-decisions", str(DECISIONS),
            "--record-observations",
            "--output", str(output),
        ],
        cwd=REPOSITORY, check=True, capture_output=True,
    )
    record = json.loads(output.read_text())["records"][0]

    env = mainline_v3e_env()
    meter = ImpulseMeter(env.controller, float(env.environment_config.dt_s))
    observation, _ = env.reset(seed=SEED)
    observations = []
    for _ in range(DECISIONS):
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        observation, _, terminated, truncated, info = env.step_with_branch(
            np.full(2, action), branch=branch
        )
        assert not (terminated or truncated)
    np.testing.assert_array_equal(
        np.asarray(record["policy_observations"], dtype=np.float32), np.stack(observations)
    )
    assert record["force_impulse_n_s"] == meter.force_impulse_n_s
    assert record["survival_s"] == float(env.env.time_seconds)
    env.close()
