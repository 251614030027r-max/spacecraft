from pathlib import Path
import json,csv,hashlib,subprocess,shutil
BASE=Path(r'C:\Users\35884\Documents\Spacecraft');TOP=BASE/'过程文件/整理_20261010';REC=TOP/'记录/B123';R=BASE/'过程文件/协作/Git工作树';C=R/'docs/collaboration'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
package=C/'lower/handoffs/cleanup_b123_20261010';assert not package.exists();package.mkdir()
for name in ('CLEANUP_FIX_VERIFICATION.json','REMOTE_BEFORE.json','TAG_PLAN.json','REMOTE_TAGS_VERIFIED.json','REMOTE_BRANCHES_VERIFIED.json','PIPELINE_STATS.json'):shutil.copy2(REC/name,package/name)
stats=read(REC/'PIPELINE_STATS.json')
report='''# Git整理第1–3步完成回执

执行时间2026-10-09 JST，主题/执行单名称20261010。用户最新明确授权CLEANUP_B_LOWER_20261010第1–3步；科学整理提交60d768ab47a3236ebe4cf3fcd05553ed4652c792，实际实验仍固定f2f8169。全程未在D:/py/DRL2执行Git操作、未移动删除任何本地科学资产、未停止实验进程。

1. 60d768a修复核验：保留62个代码文件与f2f8169 blob一致；1618条产物索引与冻结记录一致；71条中文路径用冻结提交实际取回，内容blob一致；D01口径及缩略路径问题已修正。下层尚未运行整理后的测试或行为重放。
2. 七个附注archive标签原子推到origin；七个解引用SHA逐一核验，六个分支标签与各原分支末端相同。核验后原子删除六个旧远端分支，远端仅剩main、claude/sac-mpc-coupling-design-ns7g6i、collab/spacecraft。main和科学分支SHA保持不变；collab后续仅增加此交付。旧worktree及本地分支不清理，历史提交与原v3e旧标签保留。
3. 106份COMMIT_TO_GIT候选逐字节SHA去重：81份新内容收入lower/pipeline_scripts，3份与冻结Git文件一致只登记，22份是重复副本只登记。INDEX.csv保存全部106个原路径、SHA、实验归属、处理方式、出处及盘点不确定性。归档脚本不是当前执行授权，不运行历史启动/清理脚本。

**验收边界：第1–3步完成，第5步等待评估全部交付和进程退出。** 届时干净克隆明确固定60d768a，全部测试与271000前三个开局的Pure/262460 stopping逐字段一致后，才可宣布整个整理验收完成。本地DRL2和Spacecraft实际归档仍等待用户确认。
'''
(package/'README.md').write_text(report,encoding='utf-8')
for script in ('complete_b123.py','seal_b123.py'):shutil.copy2(TOP/'脚本'/script,package/script)
(package/'.gitattributes').write_text('* -text\n',encoding='utf-8')
manifest={p.name:sha(p) for p in package.iterdir() if p.is_file()};(package/'FILES_SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
current=read(C/'CURRENT.json');current['cleanup_b']={'phase':'B1_B2_BRANCHES_B3_PIPELINES_DONE_STEP5_DEFERRED','science_cleanup_commit':'60d768ab47a3236ebe4cf3fcd05553ed4652c792','archive_tags_verified':7,'old_remote_branches_deleted':6,'pipeline_candidates':106,'new_unique_copies':81,'identical_to_git':3,'duplicates':22,'handoff':'lower/handoffs/cleanup_b123_20261010/README.md','verification_target':'60d768ab47a3236ebe4cf3fcd05553ed4652c792','step5_condition':'All experimental evidence delivered and evaluation/replay processes exited','scientific_checkout_untouched':True}
(C/'CURRENT.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
with (C/'lower/README.md').open('a',encoding='utf-8') as f:f.write('\n\nGit整理第1–3步完成：[标签、分支及流水线去重回执](handoffs/cleanup_b123_20261010/README.md)；[流水线INDEX](pipeline_scripts/INDEX.csv)。第5步等待实验交付，目标60d768a；本地未整理。\n')
print('B123_DELIVERY_READY',stats)
