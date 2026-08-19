"""Fail-closed L2 SAC adaptation and fixed checkpoint gate.

The policy is initialized from the best accepted L0 model.  No replay buffer is
loaded, the learning counter is reset, and the reward/observation definition is
kept unchanged.  Candidate checkpoints are compared on paired episodes before
one independent 100-episode gate is run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.evaluate_policy import evaluate


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(
    r"C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\python\python.exe"
)
AUTOMATION_DIR = PROJECT_ROOT / "logs" / "l2_adaptation_automation"
STATUS_PATH = AUTOMATION_DIR / "status.json"
SNAPSHOT_PATH = AUTOMATION_DIR / "source_snapshot_sha256.json"
RUN_NAME = "sac_l2_adapt_v4_200k_s20260814"
RUN_LOG_DIR = PROJECT_ROOT / "logs" / RUN_NAME
RUN_MODEL_DIR = PROJECT_ROOT / "models" / RUN_NAME
SOURCE_MODEL = (
    PROJECT_ROOT / "models" / "sac_l0_v4_300k_s20260810" / "final_model.zip"
)
SOURCE_MODEL_SHA256 = (
    "782653A4C3FA8C7560B5499D03DC7C20DD9E957BE47977D4DEFFA3D651C376CB"
)
AUDIT_PATH = PROJECT_ROOT / "logs" / "audit_l2_compatible_v4_s20260815.json"
SMOKE_MANIFEST = (
    PROJECT_ROOT / "logs" / "smoke_l2_adapt_init_s20260814" / "manifest.json"
)
SELECTION_PATH = AUTOMATION_DIR / "checkpoint_selection_30ep.json"
OFFICIAL_PATH = RUN_LOG_DIR / "evaluation_l2_gate_100_s20273400.json"
SOURCE_DIRS = ("dynamics", "env", "eval", "train")
NUMERICAL_ANOMALY = re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def write_status(phase: str, **details: Any) -> None:
    write_json_atomic(
        STATUS_PATH,
        {"updated_at_utc": utc_now(), "phase": phase, **details},
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def source_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory_name in SOURCE_DIRS:
        for path in sorted((PROJECT_ROOT / directory_name).rglob("*.py")):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            snapshot[relative] = sha256(path)
    return snapshot


def assert_source_unchanged(initial: dict[str, str]) -> None:
    current = source_snapshot()
    if current != initial:
        changed = sorted(set(initial) | set(current))
        changed = [name for name in changed if initial.get(name) != current.get(name)]
        raise RuntimeError(f"source code changed during L2 gate: {changed}")


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT / ".venv" / "Lib" / "site-packages"), str(PROJECT_ROOT))
    )
    return environment


def run_checked(
    command: list[str], log_path: Path, *, timeout_seconds: int, stop_on_nan: bool
) -> None:
    anomaly = threading.Event()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                if stop_on_nan and NUMERICAL_ANOMALY.search(line):
                    anomaly.set()
                    if process.poll() is None:
                        process.terminate()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            reader.join(timeout=10)
            raise RuntimeError(f"child timed out: {command}")
        reader.join(timeout=10)
    if anomaly.is_set():
        raise RuntimeError(f"numerical NaN detected: {command}")
    if return_code != 0:
        raise RuntimeError(f"child exited with code {return_code}: {command}")


def training_command(*, validate_only: bool) -> list[str]:
    command = [
        str(PYTHON_EXE),
        "-B",
        "-m",
        "train.train",
        "--algorithm",
        "sac",
        "--profile",
        "sac_rationalized_adapt",
        "--env-profile",
        "rationalized_l2_compatible_physical_v4",
        "--steps",
        "200000",
        "--seed",
        "20260814",
        "--device",
        "cuda",
        "--run-name",
        RUN_NAME,
        "--checkpoint-freq",
        "25000",
        "--init-model",
        str(SOURCE_MODEL),
    ]
    if validate_only:
        command.append("--validate-only")
    return command


def validate_preflight() -> None:
    if not PYTHON_EXE.is_file() or not SOURCE_MODEL.is_file():
        raise FileNotFoundError("training runtime or source model is missing")
    if sha256(SOURCE_MODEL) != SOURCE_MODEL_SHA256:
        raise RuntimeError("source L0 model hash drifted")
    if AUTOMATION_DIR.exists() or RUN_LOG_DIR.exists() or RUN_MODEL_DIR.exists():
        raise FileExistsError("L2 automation or target run artifacts already exist")
    if shutil.disk_usage(PROJECT_ROOT).free < 5 * 1024**3:
        raise RuntimeError("less than 5 GiB free space")
    audit = load_json(AUDIT_PATH)
    reset = audit.get("reset_audit") or {}
    pd = audit.get("controllability_audit") or {}
    if (
        audit.get("environment_profile") != "rationalized_l2_compatible_physical_v4"
        or reset.get("samples") != 2000
        or not reset.get("all_initial_states_inside_distance_limit")
        or pd.get("episodes") != 20
        or pd.get("success_rate") != 1.0
        or pd.get("distance_failure_rate") != 0.0
    ):
        raise RuntimeError("L2 reset/controllability audit did not pass")
    smoke = load_json(SMOKE_MANIFEST)
    initialization = smoke.get("initialization") or {}
    if (
        smoke.get("status") != "completed"
        or smoke.get("actual_model_timesteps") != 64
        or smoke.get("fresh_initialization") is not False
        or smoke.get("external_artifacts_loaded") is not True
        or smoke.get("initial_replay_buffer_size") != 0
        or initialization.get("model_sha256") != SOURCE_MODEL_SHA256
        or initialization.get("replay_buffer_loaded") is not False
        or initialization.get("reset_num_timesteps") is not True
    ):
        raise RuntimeError("initialized-policy smoke chain did not pass")


def validate_training_manifest() -> dict[str, Any]:
    manifest_path = RUN_LOG_DIR / "manifest.json"
    model_path = RUN_MODEL_DIR / "final_model.zip"
    if not manifest_path.is_file() or not model_path.is_file():
        raise RuntimeError("completed training artifacts are incomplete")
    manifest = load_json(manifest_path)
    initialization = manifest.get("initialization") or {}
    expected = {
        "status": "completed",
        "run_name": RUN_NAME,
        "algorithm": "sac",
        "profile": "sac_rationalized_adapt",
        "environment_profile": "rationalized_l2_compatible_physical_v4",
        "requested_steps": 200000,
        "actual_model_timesteps": 200000,
        "seed": 20260814,
        "fresh_initialization": False,
        "external_artifacts_loaded": True,
        "initial_replay_buffer_size": 0,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"training manifest {key} drifted")
    if (
        initialization.get("model_sha256") != SOURCE_MODEL_SHA256
        or initialization.get("replay_buffer_loaded") is not False
        or initialization.get("reset_num_timesteps") is not True
    ):
        raise RuntimeError("training initialization provenance drifted")
    monitor = (RUN_LOG_DIR / "train_monitor.csv").read_text(
        encoding="utf-8", errors="replace"
    )
    if NUMERICAL_ANOMALY.search(monitor):
        raise RuntimeError("NaN found in completed training monitor")
    return manifest


def evaluate_candidate(label: str, model: Path) -> dict[str, Any]:
    write_status("evaluating_candidate", candidate=label)
    overrides = None
    if model == SOURCE_MODEL:
        overrides = {
            "max_time": 160.0,
            "max_distance": 24.0,
            "position_shell_inner_m": 2.0,
            "position_shell_outer_m": 8.0,
            "inertial_relative_velocity_component_limit_m_s": 0.1,
            "max_relative_attitude_rad": 1.0471975511965976,
            "target_tumble_scale": 0.5,
        }
    return evaluate(
        "sac",
        model,
        episodes=30,
        seed=20273300,
        device="cpu",
        deterministic=True,
        external_max_steps=None,
        attitude_threshold_deg=15.0,
        environment_overrides=overrides,
    )


def select_checkpoint(results: dict[str, dict[str, Any]]) -> str:
    def key(item: tuple[str, dict[str, Any]]) -> tuple[int, float, float, float]:
        label, result = item
        rates = result["rates"]
        safe = rates["distance_failure"] <= 0.05 and rates["position_success"] >= 0.90
        step = 0 if label == "initial_l0" else int(label.removesuffix("k"))
        return (
            int(safe),
            rates["joint_success"] - 2.0 * rates["distance_failure"],
            rates["position_success"],
            -float(step),
        )

    return max(results.items(), key=key)[0]


def main() -> None:
    validate_preflight()
    AUTOMATION_DIR.mkdir(parents=True)
    initial_source = source_snapshot()
    write_json_atomic(SNAPSHOT_PATH, initial_source)
    write_status("testing")
    run_checked(
        [str(PYTHON_EXE), "-B", "-m", "pytest", "-q"],
        AUTOMATION_DIR / "pytest.log",
        timeout_seconds=20 * 60,
        stop_on_nan=False,
    )
    assert_source_unchanged(initial_source)
    write_status("validating_training_request")
    run_checked(
        training_command(validate_only=True),
        AUTOMATION_DIR / "validate.log",
        timeout_seconds=10 * 60,
        stop_on_nan=False,
    )
    write_status("training", run_name=RUN_NAME, requested_steps=200000)
    run_checked(
        training_command(validate_only=False),
        AUTOMATION_DIR / "training_stdout.log",
        timeout_seconds=5 * 60 * 60,
        stop_on_nan=True,
    )
    manifest = validate_training_manifest()
    assert_source_unchanged(initial_source)

    candidates = {
        "initial_l0": SOURCE_MODEL,
        "50k": RUN_MODEL_DIR / "checkpoints" / "sac_50000_steps.zip",
        "100k": RUN_MODEL_DIR / "checkpoints" / "sac_100000_steps.zip",
        "150k": RUN_MODEL_DIR / "checkpoints" / "sac_150000_steps.zip",
        "200k": RUN_MODEL_DIR / "checkpoints" / "sac_200000_steps.zip",
    }
    for path in candidates.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    results = {label: evaluate_candidate(label, model) for label, model in candidates.items()}
    selected = select_checkpoint(results)
    selection = {
        "schema_version": 1,
        "paired_seed": 20273300,
        "episodes_per_candidate": 30,
        "selection_rule": (
            "prefer distance_failure<=5% and position_success>=90%; then maximize "
            "joint_success-2*distance_failure, position_success, and earlier step"
        ),
        "selected": selected,
        "candidate_models": {key: str(value) for key, value in candidates.items()},
        "candidate_rates": {key: value["rates"] for key, value in results.items()},
    }
    write_json_atomic(SELECTION_PATH, selection)
    selected_model = candidates[selected]
    write_status("evaluating_official_gate", selected=selected)
    official = evaluate(
        "sac",
        selected_model,
        episodes=100,
        seed=20273400,
        device="cpu",
        deterministic=True,
        external_max_steps=None,
        attitude_threshold_deg=15.0,
        environment_overrides=(
            {
                "max_time": 160.0,
                "max_distance": 24.0,
                "position_shell_inner_m": 2.0,
                "position_shell_outer_m": 8.0,
                "inertial_relative_velocity_component_limit_m_s": 0.1,
                "max_relative_attitude_rad": 1.0471975511965976,
                "target_tumble_scale": 0.5,
            }
            if selected == "initial_l0"
            else None
        ),
    )
    write_json_atomic(OFFICIAL_PATH, official)
    rates = official["rates"]
    gate_passed = bool(
        rates["joint_success"] >= 0.60
        and rates["position_success"] >= 0.95
        and rates["distance_failure"] <= 0.05
    )
    assert_source_unchanged(initial_source)
    write_status(
        "completed",
        run_name=RUN_NAME,
        actual_model_timesteps=manifest["actual_model_timesteps"],
        selected=selected,
        selected_model=str(selected_model),
        official_evaluation=str(OFFICIAL_PATH),
        rates=rates,
        gate={
            "joint_success_min": 0.60,
            "position_success_min": 0.95,
            "distance_failure_max": 0.05,
            "passed": gate_passed,
        },
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            write_status("failed", **failure)
        finally:
            print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        raise
