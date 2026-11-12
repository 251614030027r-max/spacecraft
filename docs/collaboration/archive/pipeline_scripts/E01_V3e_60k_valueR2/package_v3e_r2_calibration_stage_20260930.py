"""Publish a read-only, explicitly interim R2 calibration handoff."""

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(r"C:\Users\35884\Documents\Spacecraft")
PROJECT = Path(r"D:\py\DRL2_v3e")
PUBLIC = BASE / "上层交付"
NAME = "V3E_60K_R2_CALIBRATION_AND_50K_BRIDGE_20260930"
STATE = BASE / "过程文件/V3e价值第二轮/记录/V3E_VALUE_R2_EXECUTION_20260929.json"
sys.path.insert(0, str(BASE / "过程文件/V3e60k/脚本"))
from spacecraft_asset_layout import publish  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    state = read_json(STATE)
    seeds = (262420, 262421, 262422)
    if state["status"] != "running" or state["errors"] or state.get("package_status"):
        raise RuntimeError("The stage handoff applies only while R2 runs without errors")
    if any(state["jobs"][f"m6r2_{seed}"]["status"] != "completed" for seed in seeds):
        raise RuntimeError("All three M6 calibration jobs must be completed")
    if [state["models"][str(seed)]["m6_gate"] for seed in seeds] != ["PASS", "STOP", "STOP"]:
        raise RuntimeError("Unexpected M6 gates; inspect the live run")
    latest = PUBLIC / "最新"
    if list(latest.glob("*.zip")) != [latest / "V3E_60K_R2_CALIBRATION_STAGE_20260930.zip"]:
        raise RuntimeError("Latest release changed; avoid replacing a newer delivery")
    pending = PUBLIC / "待发布"
    bundle = pending / NAME
    archive = pending / f"{NAME}.zip"
    receipt_path = pending / f"{NAME}_RECEIPT.json"
    if any(path.exists() for path in (bundle, archive, receipt_path)):
        raise FileExistsError("Stage build already exists; inspect it before retrying")
    bundle.mkdir(parents=True)
    shutil.copy2(STATE, bundle / "execution_snapshot.json")
    comparison_src = BASE / "过程文件/V3e60k/记录/V3E_50K_VS_60K_LEARNED_READOUT_20260929.json"
    comparison = read_json(comparison_src)
    shutil.copy2(comparison_src, bundle / "50k_vs_60k_learned_readout.json")
    rows = []
    for seed in seeds:
        src = PROJECT / f"eval/v3e/{seed}"
        m3_src = src / "values_r2/m3_report.json"
        m6_src = src / "m6_r2.json"
        m3 = read_json(m3_src)
        m6 = read_json(m6_src)
        if m6["summary"]["gate"] != state["models"][str(seed)]["m6_gate"]:
            raise RuntimeError(f"M6 state and source mismatch: {seed}")
        shutil.copy2(m3_src, bundle / f"m3_report_{seed}.json")
        shutil.copy2(m6_src, bundle / f"m6_r2_{seed}.json")
        rows.append((seed, m3, m6))
    generated = datetime.now(timezone.utc).astimezone().isoformat()
    report = [
        "# V3e 60k 第二轮价值校准阶段反馈（供上层先审）",
        "",
        f"生成时间：{generated}。这是**阶段结果**；262420 的正式48回合仲裁仍在运行，不能把本报告当作最终仲裁或安全判定。",
        "",
        "## 当前结论",
        "",
        "三模型均按同一第二轮方案补齐192个独立 learned_full 开局并完成价值拟合；独立校准门为 **262420 PASS、262421 STOP、262422 STOP**。按预先固定的‘至少两个模型均通过两层仲裁’规则，本轮整体成立条件已无法达到；262420仍继续正式仲裁，以取得单模型实测效果和安全数据。STOP模型不进入仲裁，不能拿第一轮结果替代第二轮主结果。",
        "",
        "## 上层此前50k之后的已知进展",
        "",
        "三个60k冻结模型均完成训练与同48开局 learned-only 测试；Pure MPC为37/48。相同48开局的50k→60k对照如下，均为学习策略单独运行，不是仲裁效果。",
        "",
        "|模型|50k完成|60k完成|净变化|50k真值违规|60k真值违规|60k QP零回退步|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparison["rows"]:
        report.append(
            f"|{item['seed']}|{item['completed_50k']}/48|{item['completed_60k']}/48|"
            f"{item['completed_60k'] - item['completed_50k']:+d}|"
            f"{item['truth_violation_episodes_50k']}|{item['truth_violation_episodes_60k']}|"
            f"{item['qp_zero_fallback_steps_60k']}|"
        )
    report += [
        "",
        "50k→60k三种子完成数均提高，但60k learned-only仍有真值违规；这些违规属于结果，最终安全门检查仲裁后的结果。第一轮262420仲裁为42/48、真值违规1回合、QP零回退35步、救回7/丢失2，L1未通过、L2通过，仅作补充。第一轮M6也是PASS/STOP/STOP；第二轮扩大数据后仍为一过两停，但两轮独立校准开局不同，不能把一致率差当作同样本配对提升或证明样本量是唯一原因。",
        "",
        "## 第二轮固定设计与已完成的校准",
        "",
        "SAC策略、任务效用、V_B掩码、网络/超参数/拟合种子、仲裁规则均未改。每模型新补271000–271191共192个开局；旧M2复用。V_L用200个独立开局拟合、40个留出（270040–270047及271160–271191），校准开局262112–262123与拟合/正式48开局分开。M6决定性点少于5为INCONCLUSIVE；不少于5且符号一致率至少80%才PASS，否则STOP。",
        "",
        "|模型|M6决定性点|符号一致率|错误选择点|门|V_L留出MAE / R²|V_B留出MAE / R²|M5入口|",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for seed, m3, m6 in rows:
        summary = m6["summary"]
        entry = "正式48回合运行中" if seed == 262420 else "STOP，按规则跳过"
        report.append(
            f"|{seed}|{summary['decisive_checkpoints']}|{summary['sign_agreement_decisive']:.1%}|"
            f"{summary['wrong_picks']}|{summary['gate']}|"
            f"{m3['V_L']['holdout']['mae']:.2f} / {m3['V_L']['holdout']['r2']:.3f}|"
            f"{m3['V_B']['holdout']['mae']:.2f} / {m3['V_B']['holdout']['r2']:.3f}|{entry}|"
        )
    report += [
        "",
        "留出MAE/R²是回报预测误差，不是预设M6门的替代指标；三模型门禁直接依据独立校准的符号一致率。262420即使M6 PASS，校准中的关键点准确率仍为4/7，不能先声称仲裁有效或安全。",
        "",
        "## 仍待完成与最终判定边界",
        "",
        "262420继续同262000–262047的48开局正式仲裁，随后由官方判读脚本给出完整策略表、L1/L2和QP/安全结果。L1要求完成≥37、丢失≤2、真值违规≤Pure的0、平均learned占比≥5%；L2要求救回>丢失。违规数是结果，不作为中途停止条件。若无程序故障，后台自动打包第二轮主结果并核验发布；本阶段报告不会替代最终包。并行总耗时不作为正式串行实时性结论。",
        "",
        "## 证据与版本",
        "",
        "执行工程：D:/py/DRL2_v3e；固定执行提交 c84ed7435f790198630ee7011fc7fd74b7307b48；SAC训练提交 a03632a304bf8af50c4be42cc9d3e450fb23d5ad。模型为 logs/v3e_<seed>/final_model.zip，第二轮价值为 eval/v3e/<seed>/values_r2/values_L.pt 与 values_B.pt。所有拟合、校准原始JSON及执行状态快照在本包内；FILE_SHA256.json记录逐文件哈希，ZIP和回执记录包哈希。模型、NPZ、PT大文件仍在原工程，未入本阶段包，未提交或推送Git。",
        "",
    ]
    (bundle / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    hashes = {p.name: sha(p) for p in sorted(bundle.iterdir()) if p.is_file()}
    (bundle / "FILE_SHA256.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(bundle.iterdir()):
            z.write(path, path.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC failed")
        if set(z.namelist()) != {p.name for p in bundle.iterdir()}:
            raise RuntimeError("ZIP entry mismatch")
        for filename, digest in hashes.items():
            if hashlib.sha256(z.read(filename)).hexdigest() != digest:
                raise RuntimeError(f"ZIP content hash mismatch: {filename}")
    if read_json(STATE)["status"] != "running":
        raise RuntimeError("R2 reached a final state during stage packaging; leave staged files unpublished")
    receipt = {
        "role": "round-two calibration stage, not final arbitration",
        "generated_at": generated,
        "status": "calibration_completed_m5_running",
        "report": str(latest / NAME / "REPORT.md"),
        "archive": {"path": str(latest / archive.name), "bytes": archive.stat().st_size, "sha256": sha(archive)},
        "file_sha256": hashes,
        "zip_crc_and_file_hashes_verified": True,
        "m6_gates": {str(seed): m6["summary"]["gate"] for seed, _, m6 in rows},
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    final_bundle, final_archive, final_receipt = publish(bundle, archive, receipt_path)
    with (latest / "README.md").open("a", encoding="utf-8") as stream:
        stream.write("\n本包是第二轮校准阶段反馈，正式48回合仲裁和最终判定尚未完成。\n")
    print(json.dumps({"archive": str(final_archive), "report": str(final_bundle / "REPORT.md"),
                      "receipt": str(final_receipt), "sha256": sha(final_archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
