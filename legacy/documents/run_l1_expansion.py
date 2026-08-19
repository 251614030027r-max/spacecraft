"""Fail-closed L1 pure-SAC expansion: preflight, train, evaluate, and summarize."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from run_l0_repro_gate import (
    PYTHON_EXE,
    assert_source_unchanged,
    run_checked,
    source_snapshot,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "sac_l1_v4_400k_s20260812"
TRAINING_SEED = 20260812
EVALUATION_SEED = 20272300
TRAINING_STEPS = 400_000
AUTOMATION_DIR = PROJECT_ROOT / "logs" / "l1_expansion_automation"
STATUS_PATH = AUTOMATION_DIR / "status.json"
MODEL_PATH = PROJECT_ROOT / "models" / RUN_NAME / "final_model.zip"
MANIFEST_PATH = PROJECT_ROOT / "logs" / RUN_NAME / "manifest.json"
EVALUATION_PATH = (
    PROJECT_ROOT
    / "logs"
    / RUN_NAME
    / f"evaluation_baseline15_100_s{EVALUATION_SEED}.json"
)
SUMMARY_PATH = PROJECT_ROOT / "logs" / "l1_expansion_gate_s20260812.json"
AUDIT_PATH = PROJECT_ROOT / "logs" / "audit_l1_physical_v4_s20260813.json"


def write_status(phase: str, **details: Any) -> None:
    write_json_atomic(
        STATUS_PATH,
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            **details,
        },
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def binned_rates(
    records: list[dict[str, Any]],
    values: list[float],
    edges: tuple[float, ...],
) -> list[dict[str, Any]]:
    if len(records) != len(values):
        raise ValueError("records and bin values must have equal length")
    result: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        selected = [
            record
            for record, value in zip(records, values)
            if value >= lower and (value < upper or (index == len(edges) - 2 and value <= upper))
        ]
        count = len(selected)
        result.append(
            {
                "lower_inclusive": lower,
                "upper_inclusive_if_last": upper,
                "episodes": count,
                "joint_success_rate": (
                    sum(bool(item["joint_success"]) for item in selected) / count
                    if count
                    else None
                ),
                "completed_rate": (
                    sum(bool(item["completed"]) for item in selected) / count
                    if count
                    else None
                ),
                "distance_failure_rate": (
                    sum(bool(item["distance_failure"]) for item in selected) / count
                    if count
                    else None
                ),
            }
        )
    return result


def validate_audit() -> None:
    audit = load_json(AUDIT_PATH)
    if audit.get("environment_profile") != "rationalized_l1_physical_v4":
        raise RuntimeError("L1 audit profile mismatch")
    reset = audit["reset_audit"]
    controllability = audit["controllability_audit"]
    if reset.get("samples") != 2000 or not reset.get(
        "all_initial_states_inside_distance_limit"
    ):
        raise RuntimeError("L1 reset audit did not pass")
    if (
        controllability.get("episodes") != 20
        or controllability.get("success_rate") != 1.0
        or controllability.get("distance_failure_rate") != 0.0
    ):
        raise RuntimeError("L1 geometric-PD audit did not pass")


def validate_manifest() -> dict[str, Any]:
    if not MODEL_PATH.is_file() or not MANIFEST_PATH.is_file():
        raise RuntimeError("L1 final model or manifest is missing")
    manifest = load_json(MANIFEST_PATH)
    expected = {
        "status": "completed",
        "run_name": RUN_NAME,
        "algorithm": "sac",
        "profile": "sac_rationalized_l1",
        "environment_profile": "rationalized_l1_physical_v4",
        "seed": TRAINING_SEED,
        "requested_steps": TRAINING_STEPS,
        "actual_model_timesteps": TRAINING_STEPS,
        "fresh_initialization": True,
        "external_artifacts_loaded": False,
        "checkpoint_frequency": 50_000,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    environment = manifest["environment"]
    environment_expected = {
        "max_time": 120.0,
        "max_distance": 15.0,
        "position_shell_inner_m": 1.0,
        "position_shell_outer_m": 5.0,
        "inertial_relative_velocity_component_limit_m_s": 0.075,
        "max_relative_attitude_rad": float(np.deg2rad(45.0)),
        "target_tumble_scale": 0.5,
        "max_force_per_axis_n": 5.0,
        "max_torque_per_axis_nm": 0.6,
        "include_chaser_disturbance": False,
        "include_uncertainty": False,
        "reward_mode": "se3_physical_storage_v4",
    }
    for key, value in environment_expected.items():
        actual = environment.get(key)
        if isinstance(value, float):
            if not np.isclose(float(actual), value):
                raise RuntimeError(f"environment {key} drifted: {actual!r}")
        elif actual != value:
            raise RuntimeError(f"environment {key} drifted: {actual!r}")
    return manifest


def training_command(*, validate_only: bool) -> list[str]:
    command = [
        str(PYTHON_EXE),
        "-B",
        "-m",
        "train.train",
        "--algorithm",
        "sac",
        "--profile",
        "sac_rationalized_l1",
        "--env-profile",
        "rationalized_l1_physical_v4",
        "--steps",
        str(TRAINING_STEPS),
        "--seed",
        str(TRAINING_SEED),
        "--device",
        "cuda",
        "--run-name",
        RUN_NAME,
        "--checkpoint-freq",
        "50000",
    ]
    if validate_only:
        command.append("--validate-only")
    return command


def main() -> None:
    for path in (
        AUTOMATION_DIR,
        PROJECT_ROOT / "logs" / RUN_NAME,
        PROJECT_ROOT / "models" / RUN_NAME,
        SUMMARY_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    validate_audit()
    initial_source = source_snapshot()
    write_json_atomic(AUTOMATION_DIR / "source_snapshot_sha256.json", initial_source)

    write_status("testing")
    run_checked(
        [str(PYTHON_EXE), "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q"],
        AUTOMATION_DIR / "pytest.log",
        timeout_seconds=10 * 60,
        stop_on_nan=False,
    )
    assert_source_unchanged(initial_source)

    write_status("validating_training_request")
    run_checked(
        training_command(validate_only=True),
        AUTOMATION_DIR / "training_request.json",
        timeout_seconds=10 * 60,
        stop_on_nan=False,
    )
    assert_source_unchanged(initial_source)

    write_status("training", run_name=RUN_NAME, requested_steps=TRAINING_STEPS)
    run_checked(
        training_command(validate_only=False),
        AUTOMATION_DIR / "training_stdout.log",
        timeout_seconds=6 * 60 * 60,
        stop_on_nan=True,
    )
    manifest = validate_manifest()
    assert_source_unchanged(initial_source)

    write_status("evaluating", episodes=100, evaluation_seed=EVALUATION_SEED)
    run_checked(
        [
            str(PYTHON_EXE),
            "-B",
            "-m",
            "eval.evaluate_policy",
            "--algorithm",
            "sac",
            "--model",
            str(MODEL_PATH),
            "--episodes",
            "100",
            "--seed",
            str(EVALUATION_SEED),
            "--device",
            "cpu",
            "--attitude-threshold-deg",
            "15",
            "--output",
            str(EVALUATION_PATH),
            "--quiet",
        ],
        AUTOMATION_DIR / "evaluation_stdout.log",
        timeout_seconds=2 * 60 * 60,
        stop_on_nan=True,
    )
    assert_source_unchanged(initial_source)
    evaluation = load_json(EVALUATION_PATH)
    if (
        evaluation.get("episodes") != 100
        or evaluation.get("base_seed") != EVALUATION_SEED
        or not evaluation.get("deterministic")
        or not evaluation.get("success_threshold_override_applied")
    ):
        raise RuntimeError("L1 evaluation metadata mismatch")
    records = evaluation["episode_records"]
    rates = evaluation["rates"]
    acceptance = {
        "passed": bool(
            rates["joint_success"] >= 0.50
            and rates["position_success"] >= 0.95
            and rates["distance_failure"] <= 0.05
        ),
        "criteria": {
            "minimum_joint_success_rate": 0.50,
            "minimum_position_success_rate": 0.95,
            "maximum_distance_failure_rate": 0.05,
        },
    }
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "pure_sac_l1_single_seed_expansion_gate",
        "run_name": RUN_NAME,
        "training_seed": TRAINING_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "model": str(MODEL_PATH),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest().upper(),
        "manifest": str(MANIFEST_PATH),
        "evaluation": str(EVALUATION_PATH),
        "rates": rates,
        "statistics": evaluation["statistics"],
        "distance_bins_m": binned_rates(
            records,
            [float(item["initial_position_error_m"]) for item in records],
            (1.0, 3.0, 4.0, 5.0),
        ),
        "attitude_bins_deg": binned_rates(
            records,
            [float(np.rad2deg(item["initial_attitude_error_rad"])) for item in records],
            (0.0, 15.0, 30.0, 45.0),
        ),
        "acceptance": acceptance,
        "training_manifest_status": manifest["status"],
    }
    write_json_atomic(SUMMARY_PATH, summary)
    write_status(
        "completed",
        gate_passed=acceptance["passed"],
        summary=str(SUMMARY_PATH),
        rates=rates,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        try:
            write_status(
                "failed",
                error_type=type(error).__name__,
                error=str(error),
            )
        finally:
            raise
