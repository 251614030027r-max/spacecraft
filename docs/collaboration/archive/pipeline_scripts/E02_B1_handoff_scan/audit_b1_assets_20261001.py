"""Inventory the frozen B1 files without reading handoff outcomes or window statistics."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("D:/py/DRL2")
RECORDS = Path("C:/Users/35884/Documents/Spacecraft/过程文件/阶段B1/记录")
OUTPUT = RECORDS / "B1_ASSET_INVENTORY_PRE_READOUT_20261001.json"
EXPECTED_COMMIT = "a714c61cc3327525ef6c103ecfcb9dc46e552a07"
MODELS = ("262420", "262421", "262422")
EXPECTED_BLOCK = set(range(262000, 262048))


def file_info(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    head = subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    tracked_status = subprocess.check_output(["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    problems: list[str] = []
    if head != EXPECTED_COMMIT:
        problems.append(f"HEAD {head} != {EXPECTED_COMMIT}")
    if tracked_status:
        problems.append("tracked worktree has changes")

    scan_rows: dict[str, object] = {}
    for model in MODELS:
        directory = REPO / "eval" / "v3e" / "stage_b" / model
        block_files = sorted(directory.glob("seed_262*.json"))
        block_ids = {int(path.stem.split("_")[1]) for path in block_files}
        missing = sorted(EXPECTED_BLOCK - block_ids)
        extra = sorted(block_ids - EXPECTED_BLOCK)
        locks = sorted(path.name for path in directory.glob("*.lock"))
        temps = sorted(path.name for path in directory.glob("*.tmp"))
        if missing or extra or len(block_files) != 48:
            problems.append(f"{model}: block count={len(block_files)} missing={missing} extra={extra}")
        if locks or temps:
            problems.append(f"{model}: locks={locks} temps={temps}")
        scan_rows[model] = {
            "judgment_block": [file_info(path) for path in block_files],
            "out_of_block_preserved": [file_info(path) for path in sorted(directory.glob("seed_270*.json"))],
            "locks": locks,
            "temp_files": temps,
        }

    inputs = [REPO / "eval" / "v3e" / "pure_mpc.json"]
    for model in MODELS:
        inputs += [REPO / "logs" / f"v3e_{model}" / "final_model.zip", REPO / "eval" / "v3e" / model / "learned_only.json"]
        inputs += [REPO / "eval" / "v3e" / model / f"m2_{part}.json" for part in "abcd"]
    for path in inputs:
        if not path.is_file():
            problems.append(f"missing input: {path}")
    logs = sorted(RECORDS.glob("b1_*.stdout.log")) + sorted(RECORDS.glob("b1_*.stderr.log"))
    logs += sorted(RECORDS.glob("smoke*.log"))
    nonempty_stderr = [str(path) for path in logs if path.name.endswith(".stderr.log") and path.stat().st_size]
    if nonempty_stderr:
        problems.append(f"nonempty stderr: {nonempty_stderr}")

    report = {
        "schema": "stage_b1_asset_inventory_pre_readout_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO),
        "code_commit": head,
        "tracked_worktree_clean": not bool(tracked_status),
        "judgment_block": "262000-262047",
        "not_used_for_judgment": "seed_270*.json; preserved without opening scientific content",
        "scan_files": scan_rows,
        "inputs": [file_info(path) for path in inputs if path.is_file()],
        "logs": [file_info(path) for path in logs],
        "problems": problems,
    }
    RECORDS.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "counts": {m: len(scan_rows[m]["judgment_block"]) for m in MODELS}, "problems": problems}, ensure_ascii=False))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
