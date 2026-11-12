"""Publish one independently reviewable Stage B1 evidence package.

The experiment files are read only. The package is built under 待发布 and
verified before the shared 最新 entry is replaced.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("C:/Users/35884/Documents/Spacecraft")
REPO = Path("D:/py/DRL2")
RECORDS = ROOT / "过程文件/阶段B1/记录"
DELIVERY = ROOT / "上层交付"
STAGING = DELIVERY / "待发布/阶段B1_20261001"
LATEST = DELIVERY / "最新"
HISTORY = DELIVERY / "历史/V3E_60K_VALUE_R2_MAIN_DELIVERY_20260929"
ZIP_NAME = "STAGE_B1_262000_BLOCK_20261001.zip"
RECEIPT_NAME = "STAGE_B1_DELIVERY_RECEIPT_20261001.json"
SCAN_COMMIT = "a714c61cc3327525ef6c103ecfcb9dc46e552a07"
READOUT_COMMIT = "d24b00a86d2320a3ff31ebe3841f536efdb566c4"
MODELS = ("262420", "262421", "262422")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def main() -> None:
    if STAGING.exists() or (LATEST / ZIP_NAME).exists() or (LATEST / RECEIPT_NAME).exists():
        raise RuntimeError("B1 staging or published package already exists; refusing overwrite")
    if git("rev-parse", "HEAD") != READOUT_COMMIT:
        raise RuntimeError("readout checkout differs from preregistered revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked checkout is dirty")
    inventory = json.loads((RECORDS / "B1_ASSET_INVENTORY_PRE_READOUT_20261001.json").read_text(encoding="utf-8"))
    if inventory["problems"] or inventory["code_commit"] != SCAN_COMMIT:
        raise RuntimeError("pre-readout inventory failed")
    readout_path = REPO / "eval/v3e/stage_b/readout_b1.json"
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    if readout["seeds"] != "262000-262047" or readout["verdict"] != "PROCEED" or readout["fidelity_problems"]:
        raise RuntimeError("unexpected official B1 readout")
    if any(not readout["gates"][m]["passes"] for m in MODELS):
        raise RuntimeError("model gates do not match the official verdict")

    state_path = RECORDS / "STAGE_B1_EXECUTION_20260930.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["status"] != "b1_judged_proceed" or state["handoff_4a"]["b1_scan_processes_remaining"] != 0:
        raise RuntimeError("B1 calculation has not been closed cleanly")

    sources: dict[str, Path] = {}
    def add(name: str, path: Path) -> None:
        if name in sources or not path.is_file():
            raise RuntimeError(f"duplicate or missing source: {name} -> {path}")
        sources[name] = path

    add("readout_b1.json", readout_path)
    add("inputs/pure_mpc.json", REPO / "eval/v3e/pure_mpc.json")
    for m in MODELS:
        directory = REPO / "eval/v3e/stage_b" / m
        block = sorted(directory.glob("seed_262*.json"))
        if {int(p.stem.split("_")[1]) for p in block} != set(range(262000, 262048)):
            raise RuntimeError(f"incomplete B1 block {m}")
        for path in sorted(directory.glob("seed_*.json")):
            if not (path.name.startswith("seed_262") or path.name.startswith("seed_270")):
                raise RuntimeError(f"unexpected scan seed {path}")
            add(f"scans/{m}/{path.name}", path)
        add(f"inputs/{m}/learned_only.json", REPO / f"eval/v3e/{m}/learned_only.json")
        for part in "abcd":
            add(f"inputs/{m}/m2_{part}.json", REPO / f"eval/v3e/{m}/m2_{part}.json")

    for name in (
        "B1_ASSET_INVENTORY_PRE_READOUT_20261001.json",
        "B1_INTERRUPTION_20260930.json",
        "B1_OFFICIAL_READOUT_20261001.log",
        "STAGE_B1_EXECUTION_20260930.json",
        "STAGE_B1_EXECUTION_PRE_4A_20261001.json",
        "STAGE_B1_EXECUTION_PRE_4B_20261001.json",
        "STAGE_B1_EXECUTION_PRE_READOUT_20261001.json",
        "pytest_stage_b1_20260930.log",
        "smoke_262006.stdout.log",
        "smoke_verification_262006.stdout.log",
    ):
        add(f"records/{name}", RECORDS / name)
    for path in sorted(RECORDS.glob("b1_*.stdout.log")) + sorted(RECORDS.glob("b1_*.stderr.log")):
        add(f"logs/{path.name}", path)
    for path in sorted(RECORDS.glob("*.stderr.log")):
        if path.stat().st_size:
            raise RuntimeError(f"nonempty stderr {path}")
        if f"logs/{path.name}" not in sources:
            add(f"logs/{path.name}", path)

    source_hashes = {name: sha256(path) for name, path in sorted(sources.items())}
    for m in MODELS:
        original = {Path(row["path"]).name: row["sha256"] for group in ("judgment_block", "out_of_block_preserved") for row in inventory["scan_files"][m][group]}
        for name, digest in original.items():
            if source_hashes[f"scans/{m}/{name}"] != digest:
                raise RuntimeError(f"B1 scan changed since pre-readout audit: {m}/{name}")
    model_info = []
    for m in MODELS:
        path = REPO / f"logs/v3e_{m}/final_model.zip"
        model_info.append({"model": m, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})

    gates = readout["gates"]
    rows = ["| 模型 | 失败轨迹 | 可救 | 非退化 W≥2 | 仅单点 | 独立窗口 | 原始可救状态 | 门 |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for m in MODELS:
        g = gates[m]
        rows.append(f"| {m} | {g['failures']} | {g['rescuable']} | {g['non_degenerate']} | {g['needle_only']} | {g['independent_windows']} | {g['critical_states_raw']} | {'PASS' if g['passes'] else 'FAIL'} |")
    ignored = readout["ignored_out_of_block_files"]
    report = f"""# Stage B1 修订 2 正式交付（2026-10-01）

**官方判定：PROCEED。** `readout_b1.json` 的精确性问题为 0；三个模型都通过预注册门。它只证明学习策略失败轨迹中存在稳定、可测的交接时段，允许进入 B2-lite 收集反方向样本；**不证明**现有价值仲裁安全、优于 Pure MPC，或阶段 C/D 已获准。B2 由用户看完本轮判定后手动决定并启动，本轮没有 B2 结果。

## 固定版本、样本与流程

- 正式工程：`D:/py/DRL2`，分支 `claude/sac-mpc-coupling-design-ns7g6i`；B1 扫描提交 `{SCAN_COMMIT}`，修订 2 判读提交 `{READOUT_COMMIT}`。两次提交间只改 `CLAUDE.md`、两份 Stage B 文档、判读脚本及其测试；扫描工具、环境、动力学、控制器、训练和模型未改。扫描时及判读时的已跟踪工作树均干净。
- 依据：`docs/STAGE_B_HANDOFF_WINDOW_PREREGISTRATION_20260930.md`、`docs/STAGE_B_RUN_ORDER_LOWER_20260930.md` 的修订 2；固定开局 `262000–262047`，三个模型各 48 个，合计 144 个完整扫描文件。未进行事后挑样本；已写出的 270000 块文件随包保留，但 **不参与本次判定**，官方 JSON 在 `ignored_out_of_block_files` 列明。
- 262420/262006 实机 smoke 为 `ALL_PASS`，含学习策略决策 26 失败、交接 k=0 与 Pure MPC 对齐及 3 点前缀核对。启动前指定测试 19/19，通过修订后的判读测试 10/10。B1 错误日志均为空。
- 扫描启动 2026-09-30 12:57 日本时间；9 月 30 日约 20:35 外部中断，原因未知且无 RuntimeError，20:48 在同提交、同参数安全续跑并复用已完成文件；10 月 1 日约 14:46 三个 262000 块全部完成，B1 扫描进程退出，残留 270 锁已备份。并行运行耗时不是正式单进程实时性数据。

## 官方门槛与结果

先验证 F1 正式 learned-only 一致、F2 第 0 步交接与 Pure MPC 一致、F3 样本/版本/前缀完整。任一失败为 `FIDELITY_FAIL`；本轮 `fidelity_problems=[]`。非退化窗口定义为连续可救长度 `W≥2` 决策；每模型至少 2 条非退化失败轨迹，且不少于仅单点可救轨迹，至少 2 模型通过才 `PROCEED`。

{chr(10).join(rows)}

表中原始可救状态彼此相关，不能当作独立样本；官方跨模型判定只看冻结门。参考基线在同 48 开局：Pure MPC 完成 **37/48**，三个 learned-only 分别 **42/48、33/48、38/48**。这些完成数为对照，B1 是交接时机机制诊断，不是正式性能结论。

## 证据与边界

- `readout_b1.json`：唯一正式 verdict、每模型门、逐失败轨迹与 270 块忽略清单；`scans/<模型>/seed_262*.json`：全部正式原始扫描；`scans/<模型>/seed_270*.json`：已产生但未判定的保留文件。
- `inputs/`：本轮精确性重算需要的 Pure、learned-only、M2 四块记录；`logs/` 和 `records/`：运行、smoke、中断、收口、审计及判读证据。`JSON_SHA256.txt` 与 `ALL_FILES_SHA256.txt` 给出包内哈希；交付回执另给 ZIP 哈希和 CRC 结果。
- 预判读资产审计核对 144 个正式 JSON、代码提交、干净工作树、无锁/临时文件和空错误日志；打包再次比对全部正式 JSON 的 SHA-256，原始科学 JSON 未修改。
- 模型 ZIP 不入包，路径和哈希见 `MODEL_FILES_SHA256.json`。不嵌旧交付、不包含 B2 文件；先前 V3e R2 的 `does not hold` 科学结论仍有效，不因本次 PROCEED 改写。B2 使用原扫描提交的路径需在用户决定启动前落实，不能用判读提交悄悄重扫。
"""
    STAGING.mkdir(parents=True, exist_ok=False)
    report_path = STAGING / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    models_path = STAGING / "MODEL_FILES_SHA256.json"
    models_path.write_text(json.dumps(model_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(readout_path, STAGING / "readout_b1.json")
    json_lines = [f"{digest}  {name}" for name, digest in source_hashes.items() if name.endswith(".json")]
    json_lines.append(f"{sha256(models_path)}  MODEL_FILES_SHA256.json")
    (STAGING / "JSON_SHA256.txt").write_text("\n".join(json_lines) + "\n", encoding="utf-8")

    package_items = {**sources, "REPORT.md": report_path, "MODEL_FILES_SHA256.json": models_path, "JSON_SHA256.txt": STAGING / "JSON_SHA256.txt"}
    # readout_b1.json already appears in sources; include it only once.
    manifest_lines = [f"{sha256(path)}  {name}" for name, path in sorted(package_items.items())]
    manifest_path = STAGING / "ALL_FILES_SHA256.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    package_items["ALL_FILES_SHA256.txt"] = manifest_path
    archive = STAGING / ZIP_NAME
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, path in sorted(package_items.items()):
            zf.write(path, arcname=name)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad or set(zf.namelist()) != set(package_items):
            raise RuntimeError(f"ZIP CRC or member-list failure: {bad}")
        for name, path in package_items.items():
            if hashlib.sha256(zf.read(name)).hexdigest() != sha256(path):
                raise RuntimeError(f"ZIP member hash mismatch: {name}")

    receipt = {
        "schema": "stage_b1_delivery_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "verdict": readout["verdict"],
        "scan_commit": SCAN_COMMIT,
        "readout_commit": READOUT_COMMIT,
        "judgment_block": "262000-262047",
        "zip_name": ZIP_NAME,
        "zip_bytes": archive.stat().st_size,
        "zip_sha256": sha256(archive),
        "zip_crc_ok": True,
        "all_members_sha256_ok": True,
        "member_count": len(package_items),
        "report_sha256": sha256(report_path),
        "readout_sha256": sha256(readout_path),
        "models": model_info,
    }
    receipt_path = STAGING / RECEIPT_NAME
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if HISTORY.exists():
        raise RuntimeError(f"history target exists: {HISTORY}")
    old_files = list(LATEST.iterdir())
    if not old_files or any(not p.is_file() for p in old_files):
        raise RuntimeError("latest delivery is missing or contains unexpected directories")
    HISTORY.mkdir(parents=True, exist_ok=False)
    for path in old_files:
        shutil.move(str(path), str(HISTORY / path.name))
    for name in ("REPORT.md", "readout_b1.json", ZIP_NAME, RECEIPT_NAME):
        shutil.move(str(STAGING / name), str(LATEST / name))
    (LATEST / "README.md").write_text(
        "# 当前上层交付：Stage B1 修订 2（2026-10-01）\n\n"
        "状态：正式判定 `PROCEED`；B2 尚未启动，由用户决定后手动运行。\n\n"
        "先看 `REPORT.md` 与 `readout_b1.json`；独立完整证据包为 `" + ZIP_NAME + "`，哈希和 CRC 见 `" + RECEIPT_NAME + "`。"
        "上轮 V3e 价值第二轮移入 `上层交付/历史/V3E_60K_VALUE_R2_MAIN_DELIVERY_20260929/`。\n",
        encoding="utf-8",
    )
    if sha256(LATEST / ZIP_NAME) != receipt["zip_sha256"]:
        raise RuntimeError("published archive differs from verified staging archive")
    print(json.dumps({"latest": str(LATEST), "zip": str(LATEST / ZIP_NAME), "zip_bytes": receipt["zip_bytes"], "zip_sha256": receipt["zip_sha256"], "verdict": receipt["verdict"], "members": receipt["member_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
