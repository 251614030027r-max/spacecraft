"""Run the remaining two L0 SAC seeds and the fixed three-model gate.

This is deliberately fail-closed: any configuration drift, numerical anomaly,
timeout, non-zero child exit, or incomplete artifact stops the sequence.
"""

from __future__ import annotations

import hashlib
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(
    r"C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\python\python.exe"
)
AUTOMATION_DIR = PROJECT_ROOT / "logs" / "l0_repro_gate_automation_v3"
STATUS_PATH = AUTOMATION_DIR / "status.json"
SOURCE_SNAPSHOT_PATH = AUTOMATION_DIR / "source_snapshot_sha256.json"
AGGREGATE_PATH = PROJECT_ROOT / "logs" / "pure_sac_l0_v1_matrix_3seeds.json"
MINIMUM_FREE_BYTES = 5 * 1024**3
TRAINING_TIMEOUT_SECONDS = 4 * 60 * 60
ADOPTED_CHECK_INTERVAL_SECONDS = 5 * 60
ADOPTED_STALE_SECONDS = 20 * 60
SOURCE_DIRS = ("dynamics", "env", "eval", "train")
NUMERICAL_ANOMALY = re.compile(r"(?<![A-Za-z])(nan)(?![A-Za-z])", re.IGNORECASE)

RUNS = (
    (20260810, "sac_l0_v4_300k_s20260810"),
    (20260811, "sac_l0_v4_300k_s20260811"),
)

MANIFEST_INVARIANT_KEYS = (
    "algorithm",
    "formal_training_enabled",
    "paper_algorithm_profile",
    "paper_environment_profile",
    "requested_steps",
    "requested_device",
    "resolved_device",
    "fresh_initialization",
    "external_artifacts_loaded",
    "environment",
    "environment_profile",
    "algorithm_spec",
    "profile",
    "effective_hyperparameters",
    "profile_note",
)


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


def source_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for directory_name in SOURCE_DIRS:
        directory = PROJECT_ROOT / directory_name
        for path in sorted(directory.rglob("*.py")):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def assert_source_unchanged(initial: dict[str, str]) -> None:
    current = source_snapshot()
    if current != initial:
        changed = sorted(set(initial) | set(current))
        changed = [name for name in changed if initial.get(name) != current.get(name)]
        raise RuntimeError(f"source code changed during gate: {changed}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def invariant_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: manifest.get(key) for key in MANIFEST_INVARIANT_KEYS}


def validate_completed_run(
    seed: int,
    run_name: str,
    anchor_signature: dict[str, Any],
) -> None:
    manifest_path = PROJECT_ROOT / "logs" / run_name / "manifest.json"
    model_path = PROJECT_ROOT / "models" / run_name / "final_model.zip"
    if not manifest_path.is_file() or not model_path.is_file():
        raise RuntimeError(f"missing completed artifacts for {run_name}")
    manifest = load_json(manifest_path)
    expected = {
        "status": "completed",
        "run_name": run_name,
        "seed": seed,
        "actual_model_timesteps": 300_000,
        "checkpoint_frequency": 50_000,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"{run_name}: manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    if invariant_signature(manifest) != anchor_signature:
        raise RuntimeError(f"{run_name}: manifest configuration drifted from seed 20260805")


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(PROJECT_ROOT / ".venv" / "Lib" / "site-packages"),
            str(PROJECT_ROOT),
        )
    )
    return environment


def process_exists(pid: int) -> bool:
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ exit 0 }} else {{ exit 1 }}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    return result.returncode == 0


def stop_process(pid: int) -> None:
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )


def monitor_steps(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        data_lines = (line for line in handle if not line.startswith("#"))
        return sum(
            int(float(row["l"]))
            for row in csv.DictReader(data_lines)
            if row.get("l")
        )


def wait_for_adopted_training(
    seed: int,
    run_name: str,
    pid: int,
    anchor_signature: dict[str, Any],
    initial_source: dict[str, str],
) -> None:
    manifest_path = PROJECT_ROOT / "logs" / run_name / "manifest.json"
    monitor_path = PROJECT_ROOT / "logs" / run_name / "train_monitor.csv"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "running":
        raise RuntimeError(f"{run_name}: adopted manifest is not running")
    if manifest.get("seed") != seed or invariant_signature(manifest) != anchor_signature:
        raise RuntimeError(f"{run_name}: adopted run identity or configuration mismatch")
    started_at = datetime.fromisoformat(str(manifest["started_at_utc"]))
    try:
        while True:
            assert_source_unchanged(initial_source)
            manifest = load_json(manifest_path)
            status = manifest.get("status")
            if status == "completed":
                validate_completed_run(seed, run_name, anchor_signature)
                return
            if status != "running":
                raise RuntimeError(f"{run_name}: adopted training status became {status!r}")
            elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
            if elapsed > TRAINING_TIMEOUT_SECONDS:
                raise RuntimeError(f"{run_name}: adopted training exceeded four hours")
            if not process_exists(pid):
                raise RuntimeError(f"{run_name}: adopted process {pid} exited before completion")
            if monitor_path.is_file():
                text = monitor_path.read_text(encoding="utf-8", errors="replace")
                if NUMERICAL_ANOMALY.search(text):
                    raise RuntimeError(f"{run_name}: numerical NaN detected in monitor")
                stale_seconds = time.time() - monitor_path.stat().st_mtime
                if stale_seconds > ADOPTED_STALE_SECONDS:
                    raise RuntimeError(
                        f"{run_name}: monitor has not advanced for {stale_seconds / 60:.1f} minutes"
                    )
            write_status(
                "waiting_for_adopted_training",
                seed=seed,
                run_name=run_name,
                adopted_pid=pid,
                observed_episode_steps=monitor_steps(monitor_path),
                elapsed_minutes=elapsed / 60,
            )
            time.sleep(ADOPTED_CHECK_INTERVAL_SECONDS)
    except BaseException:
        if process_exists(pid):
            stop_process(pid)
        raise


def run_checked(
    command: list[str],
    log_path: Path,
    *,
    timeout_seconds: int,
    stop_on_nan: bool,
) -> None:
    anomaly = threading.Event()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
            creationflags=creationflags,
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
            raise RuntimeError(
                f"child process exceeded {timeout_seconds / 3600:.1f} hours: {command}"
            )
        reader.join(timeout=10)
    if anomaly.is_set():
        raise RuntimeError(f"numerical NaN detected; stopped child: {command}")
    if return_code != 0:
        raise RuntimeError(f"child exited with code {return_code}: {command}")


def training_command(seed: int, run_name: str, *, validate_only: bool) -> list[str]:
    command = [
        str(PYTHON_EXE),
        "-B",
        "-m",
        "train.train",
        "--algorithm",
        "sac",
        "--profile",
        "sac_rationalized_l0",
        "--env-profile",
        "rationalized_l0_physical_v4",
        "--steps",
        "300000",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--run-name",
        run_name,
        "--checkpoint-freq",
        "50000",
    ]
    if validate_only:
        command.append("--validate-only")
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-seed-20260810-pid", type=int)
    parser.add_argument("--adopt-seed-20260811-pid", type=int)
    args = parser.parse_args()
    if not PYTHON_EXE.is_file():
        raise FileNotFoundError(PYTHON_EXE)
    if AUTOMATION_DIR.exists():
        raise FileExistsError(f"automation directory already exists: {AUTOMATION_DIR}")
    if AGGREGATE_PATH.exists():
        raise FileExistsError(f"aggregate output already exists: {AGGREGATE_PATH}")
    for seed, run_name in RUNS:
        log_dir = PROJECT_ROOT / "logs" / run_name
        model_dir = PROJECT_ROOT / "models" / run_name
        if not log_dir.exists() and not model_dir.exists():
            continue
        manifest_path = log_dir / "manifest.json"
        if not manifest_path.is_file() or not model_dir.is_dir():
            raise FileExistsError(f"incomplete or conflicting existing run: {run_name}")
        manifest = load_json(manifest_path)
        if manifest.get("status") not in {"running", "completed"}:
            raise FileExistsError(f"existing run is not safely adoptable: {run_name}")
        if manifest.get("seed") != seed:
            raise FileExistsError(f"existing run seed mismatch: {run_name}")
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < MINIMUM_FREE_BYTES:
        raise RuntimeError(f"insufficient free space: {free_bytes / 1024**3:.2f} GiB")

    anchor = load_json(
        PROJECT_ROOT / "logs" / "sac_l0_v4_300k_s20260809" / "manifest.json"
    )
    anchor_signature = invariant_signature(anchor)
    initial_source = source_snapshot()
    write_json_atomic(SOURCE_SNAPSHOT_PATH, initial_source)
    write_status("preflight", free_gib=free_bytes / 1024**3)

    for seed, run_name in RUNS:
        assert_source_unchanged(initial_source)
        manifest_path = PROJECT_ROOT / "logs" / run_name / "manifest.json"
        if manifest_path.is_file():
            status = load_json(manifest_path).get("status")
            if status == "completed":
                validate_completed_run(seed, run_name, anchor_signature)
                write_status("training_completed", seed=seed, run_name=run_name)
                continue
            adopted_pids = {
                20260810: args.adopt_seed_20260810_pid,
                20260811: args.adopt_seed_20260811_pid,
            }
            adopted_pid = adopted_pids[seed]
            if status == "running" and adopted_pid is not None:
                wait_for_adopted_training(
                    seed,
                    run_name,
                    adopted_pid,
                    anchor_signature,
                    initial_source,
                )
                write_status("training_completed", seed=seed, run_name=run_name)
                continue
            raise RuntimeError(f"{run_name}: running artifact requires an explicitly adopted PID")
        write_status("validating_training_request", seed=seed, run_name=run_name)
        run_checked(
            training_command(seed, run_name, validate_only=True),
            AUTOMATION_DIR / f"{run_name}_validate.log",
            timeout_seconds=10 * 60,
            stop_on_nan=False,
        )
        assert_source_unchanged(initial_source)
        write_status("training", seed=seed, run_name=run_name)
        run_checked(
            training_command(seed, run_name, validate_only=False),
            AUTOMATION_DIR / f"{run_name}_stdout.log",
            timeout_seconds=TRAINING_TIMEOUT_SECONDS,
            stop_on_nan=True,
        )
        validate_completed_run(seed, run_name, anchor_signature)
        assert_source_unchanged(initial_source)
        write_status("training_completed", seed=seed, run_name=run_name)

    write_status("evaluating_three_model_gate")
    run_checked(
        [
            str(PYTHON_EXE),
            "-B",
            "-m",
            "eval.evaluate_l0_matrix",
            "--episodes",
            "100",
            "--device",
            "cpu",
        ],
        AUTOMATION_DIR / "evaluate_l0_matrix_stdout.log",
        timeout_seconds=2 * 60 * 60,
        stop_on_nan=True,
    )
    assert_source_unchanged(initial_source)
    aggregate = load_json(AGGREGATE_PATH)
    acceptance = aggregate.get("acceptance") or {}
    write_status(
        "completed",
        gate_passed=bool(acceptance.get("passed", False)),
        acceptance=acceptance,
        aggregate=str(AGGREGATE_PATH),
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
