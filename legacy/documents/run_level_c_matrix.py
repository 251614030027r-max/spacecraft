"""Train and evaluate three independent Level-C SAC adaptation lineages."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.evaluate_policy import evaluate


ROOT = Path(__file__).resolve().parent
PYTHON = Path(
    r"C:\Users\35884\.cache\codex-runtimes\codex-primary-runtime"
    r"\dependencies\python\python.exe"
)
AUTOMATION = ROOT / "logs" / "level_c_matrix_automation"
STATUS = AUTOMATION / "status.json"
AGGREGATE = ROOT / "logs" / "pure_sac_level_c_matrix_3seeds.json"
INITIAL = (
    ROOT
    / "models"
    / "sac_l2_stable_v4_300k_s20260816"
    / "checkpoints"
    / "sac_300000_steps.zip"
)
RUNS = tuple(
    (seed, f"sac_level_c_v4_300k_s{seed}")
    for seed in (20260817, 20260818, 20260819)
)
NAN = re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])", re.IGNORECASE)


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def set_status(phase: str, **details: Any) -> None:
    write(
        STATUS,
        {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            **details,
        },
    )


def environment() -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / ".venv" / "Lib" / "site-packages"), str(ROOT))
    )
    return values


def train_command(seed: int, run: str, validate_only: bool = False) -> list[str]:
    command = [
        str(PYTHON),
        "-B",
        "-m",
        "train.train",
        "--algorithm",
        "sac",
        "--profile",
        "sac_rationalized_adapt_stable",
        "--env-profile",
        "rationalized_level_c_curriculum_physical_v4",
        "--steps",
        "300000",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--run-name",
        run,
        "--checkpoint-freq",
        "25000",
        "--init-model",
        str(INITIAL),
    ]
    if validate_only:
        command.append("--validate-only")
    return command


def run_child(command: list[str], log: Path, timeout: int) -> None:
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    if NAN.search(text):
        raise RuntimeError(f"NaN detected in {log}")
    if result.returncode:
        raise RuntimeError(f"child exit {result.returncode}: {command}")


def validate_run(seed: int, run: str) -> None:
    manifest_path = ROOT / "logs" / run / "manifest.json"
    model_path = ROOT / "models" / run / "final_model.zip"
    if not manifest_path.is_file() or not model_path.is_file():
        raise RuntimeError(f"incomplete training artifacts: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("seed") != seed
        or manifest.get("actual_model_timesteps") != 300000
        or manifest.get("profile") != "sac_rationalized_adapt_stable"
        or manifest.get("environment_profile")
        != "rationalized_level_c_curriculum_physical_v4"
        or manifest.get("initial_replay_buffer_size") != 0
    ):
        raise RuntimeError(f"training manifest mismatch: {run}")
    monitor = (ROOT / "logs" / run / "train_monitor.csv").read_text(
        encoding="utf-8", errors="replace"
    )
    if NAN.search(monitor):
        raise RuntimeError(f"NaN detected in monitor: {run}")


def evaluate_model(model: Path, episodes: int, seed: int) -> dict[str, Any]:
    return evaluate(
        "sac",
        model,
        episodes=episodes,
        seed=seed,
        device="cpu",
        deterministic=True,
        external_max_steps=None,
        attitude_threshold_deg=15.0,
        environment_overrides=None,
    )


def select_checkpoint(run: str, eval_seed: int) -> tuple[str, Path, dict[str, Any]]:
    candidates = {
        f"{step // 1000}k": ROOT
        / "models"
        / run
        / "checkpoints"
        / f"sac_{step}_steps.zip"
        for step in (150000, 200000, 250000, 300000)
    }
    results = {}
    for label, model in candidates.items():
        if not model.is_file():
            raise FileNotFoundError(model)
        set_status("evaluating_candidate", run_name=run, candidate=label)
        results[label] = evaluate_model(model, 30, eval_seed)
    eligible = [
        label
        for label, result in results.items()
        if result["rates"]["position_success"] >= 0.90
        and result["rates"]["distance_failure"] <= 0.05
    ]
    pool = eligible or list(results)
    selected = max(
        pool,
        key=lambda label: (
            results[label]["rates"]["joint_success"],
            -results[label]["rates"]["distance_failure"],
            results[label]["rates"]["position_success"],
        ),
    )
    matrix = {
        "paired_seed": eval_seed,
        "episodes_per_candidate": 30,
        "eligible": eligible,
        "selected": selected,
        "candidate_models": {key: str(value) for key, value in candidates.items()},
        "candidate_rates": {key: value["rates"] for key, value in results.items()},
    }
    write(ROOT / "logs" / run / f"checkpoint_matrix_30_s{eval_seed}.json", matrix)
    return selected, candidates[selected], matrix


def main() -> None:
    if AUTOMATION.exists() or AGGREGATE.exists():
        raise FileExistsError("Level-C matrix outputs already exist")
    if not INITIAL.is_file():
        raise FileNotFoundError(INITIAL)
    for _, run in RUNS:
        if (ROOT / "logs" / run).exists() or (ROOT / "models" / run).exists():
            raise FileExistsError(f"target run already exists: {run}")
    AUTOMATION.mkdir(parents=True)
    evaluations = []
    for index, (seed, run) in enumerate(RUNS):
        set_status("validating_training", run_name=run, lineage=index + 1)
        run_child(
            train_command(seed, run, validate_only=True),
            AUTOMATION / f"{run}_validate.log",
            10 * 60,
        )
        set_status("training", run_name=run, lineage=index + 1, requested_steps=300000)
        run_child(
            train_command(seed, run),
            AUTOMATION / f"{run}_training.log",
            5 * 60 * 60,
        )
        validate_run(seed, run)
        selected, model, matrix = select_checkpoint(run, 20273900 + 100 * index)
        set_status(
            "evaluating_official", run_name=run, lineage=index + 1, selected=selected
        )
        official_seed = 20274000 + 100 * index
        official = evaluate_model(model, 100, official_seed)
        official_path = ROOT / "logs" / run / f"evaluation_level_c_100_s{official_seed}.json"
        write(official_path, official)
        evaluations.append(
            {
                "training_seed": seed,
                "run_name": run,
                "selected_checkpoint": selected,
                "selected_model": str(model),
                "selection": matrix,
                "official_seed": official_seed,
                "official_evaluation": str(official_path),
                "rates": official["rates"],
            }
        )

    mean_rates = {
        key: sum(item["rates"][key] for item in evaluations) / len(evaluations)
        for key in evaluations[0]["rates"]
    }
    passing_lineages = sum(
        item["rates"]["joint_success"] >= 0.50 for item in evaluations
    )
    passed = (
        mean_rates["joint_success"] >= 0.60
        and mean_rates["position_success"] >= 0.90
        and mean_rates["distance_failure"] <= 0.05
        and passing_lineages >= 2
    )
    aggregate = {
        "schema_version": 1,
        "task": "rationalized_level_c_physical_v4",
        "lineages": evaluations,
        "mean_rates": mean_rates,
        "acceptance": {
            "mean_joint_success_min": 0.60,
            "mean_position_success_min": 0.90,
            "mean_distance_failure_max": 0.05,
            "lineages_joint_success_at_least_0p50_min": 2,
            "passing_lineages": passing_lineages,
            "passed": passed,
        },
    }
    write(AGGREGATE, aggregate)
    set_status(
        "completed",
        aggregate=str(AGGREGATE),
        mean_rates=mean_rates,
        acceptance=aggregate["acceptance"],
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        set_status(
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
