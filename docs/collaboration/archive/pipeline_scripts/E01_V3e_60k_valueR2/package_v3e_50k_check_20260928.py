"""Package completed 50k verification evidence without adding anything to Git."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
from datetime import datetime

ROOT = Path(r"D:\py\DRL2_v3e")
DELIVERY = Path(r"C:\Users\35884\Documents\Spacecraft")
OUT = ROOT / "eval" / "v3e"
execution = json.loads((DELIVERY / "V3E_50K_EXECUTION_20260928.json").read_text())
assert "completed_at" in execution, "verification is still running"
sys.path.insert(0, str(ROOT))
from eval.adaptive_pairing import compare_payloads

labels = ["Pure MPC", "v3_nominal"] + [f"v3d@50k {s}" for s in (262420,262421,262422)] + [f"v3e@50k {s}" for s in (262420,262421,262422)]
readout = json.loads((OUT / "early_50k_readout.json").read_text())
assert len(readout["rows"]) == len(labels)
rows = [dict(row, display_label=label) for row, label in zip(readout["rows"], labels)]
assert all(row["episodes"] == 48 for row in rows)
paths = [OUT/'pure_mpc.json', OUT/'nominal.json'] + [Path(r'D:\py\DRL2\eval\v3d')/f'learned_only_50k_{seed}.json' for seed in (262420,262421,262422)] + [OUT/str(seed)/'learned_only_50k.json' for seed in (262420,262421,262422)]
truth_details = {}
for label,p in zip(labels,paths):
    raw=json.loads(p.read_text())
    truth_details[label]=[{"seed":r["seed"],"completed":r["completed"],"survival_s":r["survival_s"],"minimum_truth_normalized_margin":r.get("minimum_truth_normalized_margin"),"violation_counts":{k:v for k,v in r.items() if "violation" in k and isinstance(v,(dict,int,float)) and not isinstance(v,bool)}} for r in raw["records"] if r["constraint_violated"]]
pure, nominal = rows[:2]
defects = []
review = []
if pure["completed"] != 37:
    defects.append(f"Pure MPC {pure['completed']}/48, expected 37/48")
for row in rows[5:]:
    if row["qp_zero_fallback_steps"] > 50:
        defects.append(f"{row['display_label']}: QP zero fallback {row['qp_zero_fallback_steps']} > 50")
    if row["episodes_with_truth_violation"] > pure["episodes_with_truth_violation"]:
        review.append(f"{row['display_label']}: truth-violation episodes {row['episodes_with_truth_violation']} vs Pure MPC {pure['episodes_with_truth_violation']}; order defines 'obviously more' qualitatively, no invented numeric threshold")
for r in execution["integrity"]:
    if r["health"]["status"] != "OK":
        defects.append(f"seed {r['seed']}: training health STOP")
    band = next(b for b in r["bins_by_episode_start"] if b["start_decisions"] == 40000)
    if band["completed"] == 0:
        review.append(f"seed {r['seed']}: zero completions in 40-50k; check timeout trend")
dryrun = {}
for name in ("dryrun_m2", "dryrun_m3", "dryrun_m6", "dryrun_m5"):
    job = execution["jobs"].get(name, {})
    dryrun[name] = {"status":job.get("status", "not_run"), "exit_code":job.get("exit_code"),
                    "stdout_tail":Path(job["stdout"]).read_text(errors="replace").splitlines()[-8:] if "stdout" in job else [],
                    "stderr_tail":Path(job["stderr"]).read_text(errors="replace").splitlines()[-8:] if "stderr" in job else []}
    if job.get("exit_code") != 0:
        defects.append(f"{name}: command did not complete successfully")
m3 = json.loads((OUT / "dryrun_50k/values/m3_report.json").read_text())
m6 = json.loads((OUT / "dryrun_50k/m6.json").read_text())
m5 = json.loads((OUT / "dryrun_50k/m5.json").read_text())
dryrun["structural_checks"] = {"V_B_masked_blocks":m3["V_B"]["masked_blocks"],
                               "task_target":m3.get("target"), "M6_has_gate":"gate" in m6["summary"], "M5_arbiter":m5["arbiter"],
                               "scope":"pipeline smoke test only; metrics not used in efficacy decisions or committed to Git"}
assert set(m3["V_B"]["masked_blocks"]) == {"task_state","applied_direction"}
assert m3["target"] == "task" and "gate" in m6["summary"] and m5["arbiter"] == "one_way"
paired = []
for seed in (262420,262421,262422):
    old=json.loads((Path(r"D:\py\DRL2\eval\v3d")/f"learned_only_50k_{seed}.json").read_text())
    new=json.loads((OUT/str(seed)/"learned_only_50k.json").read_text())
    assert old["episode_seeds"] == new["episode_seeds"] == list(range(262000,262048))
    assert not old["stochastic_policy"] and not new["stochastic_policy"]
    pair=compare_payloads(old,new)
    paired.append({"seed":seed,"v3d_completed":sum(r["completed"] for r in old["records"]),"v3e_completed":sum(r["completed"] for r in new["records"]),"v3e_rescued_from_v3d":pair["quadrants"]["rescued"]["count"],"v3e_lost_from_v3d":pair["quadrants"]["destroyed"]["count"],"common_success_v3e_minus_v3d":pair["common_success_candidate_minus_baseline"]})
bundle=DELIVERY/"V3E_50K_DELIVERY_20260928"
bundle.mkdir(exist_ok=True)
checklist={"generated_at":datetime.now().astimezone().isoformat(),"defects":defects,"requires_upper_review":review,"decision":"STOP before full 60k evaluation" if defects or review else "no listed obvious defect triggered; 60k formal evaluation may follow original order", "dryrun":dryrun,"label_mapping_note":"Official readout repeats v3e file stems; display labels here follow the CLI input order.","rows":rows,"truth_violation_details":truth_details,"equal_budget_pairs":paired,"timing_boundary":"Concurrent runs do not provide formal controller compute-time evidence."}
(bundle/"verification_checklist.json").write_text(json.dumps(checklist,ensure_ascii=False,indent=2),encoding="utf-8")
files=[(OUT/"pure_mpc.json","formal/pure_mpc.json"),(OUT/"nominal.json","formal/nominal.json"),(OUT/"early_50k_readout.json","formal/early_50k_readout.json"),
       (DELIVERY/"V3E_50K_INTEGRITY_20260928.json","integrity.json"),(DELIVERY/"V3E_50K_EXECUTION_20260928.json","execution.json"),(DELIVERY/"V3E_50K_TRAINING_BINS_20260928.csv","training_bins.csv")]
for seed in (262420,262421,262422):
    files += [(OUT/str(seed)/"learned_only_50k.json",f"formal/v3e_{seed}_50k.json"),(Path(r"D:\py\DRL2\eval\v3d")/f"learned_only_50k_{seed}.json",f"formal/v3d_{seed}_50k.json")]
for p in (OUT/"dryrun_50k").rglob("*.json"):
    files.append((p,"dryrun_only/"+p.relative_to(OUT/"dryrun_50k").as_posix()))
for job in execution["jobs"].values():
    for key in ("stdout","stderr"):
        if key in job:
            p=Path(job[key]);files.append((p,"command_logs/"+p.name))
for source,relative in files:
    dest=bundle/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
large_files=[ROOT/'logs'/f'v3e_{seed}'/'checkpoints'/'sac_mpc_50000_steps.zip' for seed in (262420,262421,262422)]
large_files += [OUT/'dryrun_50k/m2.npz',OUT/'dryrun_50k/values/values_L.pt',OUT/'dryrun_50k/values/values_B.pt']
(bundle/'large_artifact_locations.json').write_text(json.dumps([{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in large_files],ensure_ascii=False,indent=2),encoding='utf-8')

report=["# V3e 50k 提前验证交付", "",f"生成时间：{checklist['generated_at']}。训练版本 a03632a；验证版本 {execution['evaluation_commit']}，更新仅增加验证汇总脚本和文档。", "", "## 执行结论", ""]
if defects or review:
    report.append("按检查单暂停进入 60k 完整评估，交回上层判断。训练未被本次验证中断。")
    report += ["", *[f"- 明确触发：{v}" for v in defects], *[f"- 待上层判断：{v}" for v in review]]
else:
    report.append("检查单列出的明显缺陷均未触发，可以按原执行单继续 60k 正式流程。此次结论不是完整方法有效性结论。训练未被本次验证中断。")
report += ["", "## 48 开局配对评估", "", "所有行使用 seed 262000–262047；学习模型均为预定 50k 存档，确定性动作。时间与 Δv 差只统计与 Pure MPC 共同完成的开局，正值表示比 Pure MPC 更慢/更耗燃料。", "", "|行|完成|零违规完成|真值违规回合|QP回退步|救回Pure失败|丢失Pure成功|耗时差(s)|Δv差(m/s)|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
def fmt(v):return "无共同完成" if v is None else f"{v:.3f}"
for r in rows:
    report.append(f"|{r['display_label']}|{r['completed']}/48|{r['zero_violation_completed']}/48|{r['episodes_with_truth_violation']}|{r['qp_zero_fallback_steps']}|{r['rescued']}|{r['destroyed']}|{fmt(r['common_success_time_delta_s_mean'])}|{fmt(r['common_success_dv_delta_mean'])}|")
report += ["", "真值违规开局及最小归一化真值余量："]
for label,details in truth_details.items():
    if details:
        report += ["", f"- {label}: " + "; ".join(f"seed {v['seed']}, margin={v['minimum_truth_normalized_margin']}" for v in details)]
report += ["", "## 同预算 V3d → V3e", "", "|种子|V3d完成|V3e完成|V3e救回旧失败|V3e丢失旧成功|", "|---|---:|---:|---:|---:|"]
for p in paired:
    report.append(f"|{p['seed']}|{p['v3d_completed']}/48|{p['v3e_completed']}/48|{p['v3e_rescued_from_v3d']}|{p['v3e_lost_from_v3d']}|")
report += ["", "## 训练完整性与分段统计", "", "三模型 ZIP CRC 正常，内部 num_timesteps=50000；42D/2D、dirty=false、timeout_is_terminal=true。健康检查与模型哈希见 integrity.json。分组按回合开始时的累计决策数；跨边界回合归入开始区间，故这些区间不是截断到恰好 50k 的步级统计。平均回报是 monitor 中含势能整形的 SAC 训练回报，不等同于后训练任务效用。", "", "Monitor 未保存 time_failure/truncated，超时比例按未完成且达到 150 决策步推定；不能区分恰好在最后一步发生的其他失败。", "", "|种子|开始步区间|回合数|完成率|平均回报|超时代理比例|平均回合决策数|", "|---|---|---:|---:|---:|---:|---:|"]
for r in execution["integrity"]:
    for b in r["bins_by_episode_start"]:
        if b["start_decisions"]>=50000:continue
        report.append(f"|{r['seed']}|{b['start_decisions']//1000}–{b['end_decisions']//1000}k|{b['episodes']}|{b['completion_rate']:.1%}|{b['mean_return']:.3f}|{b['timeout_proxy_fraction']:.1%}|{b['mean_episode_decisions']:.2f}|")
report += ["", "## M2 → M3 → M6 → M5 试运行", "", "仅为管线可运行性检查；5 个数据种子、2 个校准开局、2 个仲裁评估开局，所有质量数字不用于效果判断、不提交 Git。", ""]
for name in ("dryrun_m2","dryrun_m3","dryrun_m6","dryrun_m5"):
    d=dryrun[name];report += [f"### {name}: {d['status']}，退出码 {d['exit_code']}", "", "```text", *d["stdout_tail"], "```", ""]
report += ["结构检查：V_B masked_blocks 为 task_state、applied_direction；M3 target=task；M6 summary 含 gate；M5 arbiter=one_way。", "", "## 证据边界与文件", "", "原始官方汇总表见 command_logs/readout.stdout.log。官方脚本的三条 V3e 标签相同，本报告按 CLI 的 262420、262421、262422 顺序补充种子，未修改官方脚本。并行测量的计算耗时不作为正式实时性证据。", "", "所有本次交付 JSON 的 SHA-256 见 JSON_SHA256.txt；模型 ZIP、M2 NPZ、价值 PT 不在交付包内，保留原路径。未提交 Git。60k 主结果仍使用 final_model.zip，Pure MPC/nominal 复用本次行；完整价值链依原执行单另跑。"]
(bundle/"REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
jsons=sorted(bundle.rglob("*.json"))
(bundle/"JSON_SHA256.txt").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(bundle).as_posix()}" for p in jsons)+"\n",encoding="utf-8")
zip_path=DELIVERY/"V3E_50K_DELIVERY_20260928.zip"
with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(bundle.rglob("*")):
        if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
print(json.dumps({"report":str(bundle/"REPORT.md"),"bundle":str(zip_path),"json_files":len(jsons),"defects":defects,"requires_upper_review":review},ensure_ascii=False,indent=2))
