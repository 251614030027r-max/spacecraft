from pathlib import Path
import hashlib,json,shutil,subprocess
BASE=Path(r'C:\Users\35884\Documents\Spacecraft'); TOP=BASE/'过程文件/整理_20261010'
COLLAB=BASE/'过程文件/协作/Git工作树'; C=COLLAB/'docs/collaboration'
REC=TOP/'记录/C_LOCAL'; REC.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
latest=BASE/'上层交付/最新'; history=BASE/'上层交付/历史/VALUE_RULE_CHECK_20261006'
assert not history.exists(),'Preserve existing archive'
history.mkdir(parents=True)
moves=[]
for p in list(latest.iterdir()):
    assert p.is_file(),str(p)
    target=history/p.name
    assert p.resolve().is_relative_to(BASE) and target.resolve().is_relative_to(BASE) and not target.exists()
    h=sha(p); shutil.move(str(p),str(target));assert sha(target)==h
    moves.append({'old':str(p),'new':str(target),'sha256':h})
package=C/'lower/handoffs/final_rerun_271000_early_20261009'
for p in package.rglob('*'):
    if not p.is_file() or p.name in ('build_phase_review.py','.gitattributes','FILES_SHA256.json','README.md'):continue
    dest=latest/p.relative_to(package);dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();shutil.copy2(p,dest);assert sha(p)==sha(dest)
review=C/'reviews/F01_271000_UPPER_REVIEW_20261010.md'
shutil.copy2(review,latest/review.name)
write(latest/'README.md','# 最新交付：F01阶段审查\n\n用户转达上层已判本次实验失败并记录，停止方法扩展；剩余评估与已授权重放仅补齐记录。此决定与冻结三模型官方判读分开：后者待learned及262461完整后出具，不能提前改写。\n\n直接阅读PHASE_REVIEW.md及F01_271000_UPPER_REVIEW_20261010.md；192个完整逐回合原始证据在raw/，配对摘要、训练来源、模型SHA随本轮提供。无模型ZIP、无历史ZIP嵌套。\n\n仍运行的271000/272000与监督状态保留原位；旧价值规则轮移入../历史/VALUE_RULE_CHECK_20261006，原文件SHA不变。\n')
hashes={p.relative_to(latest).as_posix():sha(p) for p in latest.rglob('*') if p.is_file()}
write(latest/'FILES_SHA256.json',json.dumps(hashes,ensure_ascii=False,indent=2))
write(REC/'DELIVERY_PATH_MAP.json',json.dumps(moves,ensure_ascii=False,indent=2))
project=BASE/'项目说明.md'; old=TOP/'接续/项目说明_整理前快照.md'; old.parent.mkdir(parents=True,exist_ok=True); assert not old.exists();shutil.copy2(project,old);assert sha(project)==sha(old)
write(project,'''# Spacecraft 项目入口

## 当前状态

用户转达：上层已判F01实验失败并记录，下一步由用户与上层讨论。现有评估继续补齐实验记录，不自行启动新方法、训练、筛选或增加门槛。上层已发布两模型阶段事实与根因推断，不能把推断当实测；冻结三模型官方判读尚待齐全。

- 最新本轮报告与原始证据：[上层交付/最新/README.md](上层交付/最新/README.md)。Pure35/48、nominal48/48；262460 stopping30/48且违规3，262462 stopping37/48且零违规。已完整192回合；协调增益待learned。
- 上层审查：[F01阶段审查](过程文件/协作/Git工作树/docs/collaboration/reviews/F01_271000_UPPER_REVIEW_20261010.md)。协作分支保留科学事实、审查署名和判定边界。
- 当前科学实验固定f2f8169，工况w2.36_r15，final2_262460–262462各60k已完成，30k结构门均通过。原实验工程D:/py/DRL2继续供评估使用；所有final2模型、状态、日志、锁与结果原位保留。状态入口：[PRIORITY_EXECUTION.json](过程文件/最终主线重训/记录/PRIORITY_EXECUTION.json)。
- 干净工程副本：D:/py/spacecraft，科学分支046878c（较整理核验目标60d768a只修改CLAUDE.md、DEAD_ENDS.md、HISTORY.md）。尚未完成下层全部测试与六回合行为一致性核验，不作为已验收工程。旧DRL2和DRL2_v3e保留，评估退出前不改只读属性。
- 本地历史资产复制归档：D:/spacecraft_archive/experiments/<实验ID>/，作废运行单列_invalid/；索引与逐文件SHA：[整理记录](过程文件/整理_20261010/记录/C_LOCAL/)。复制不移动、不删除原件；外置离线备份尚未建立。

## 协作与审查入口

[GitHub协作目录](https://github.com/251614030027r-max/spacecraft/tree/collab/spacecraft/docs/collaboration)：upper/run_orders保存执行单，lower/handoffs保存独立交付，reviews保存上层审查，CURRENT.json保存核验状态。附件与文档提供上下文，用户最新授权决定执行范围。main未合并；7个归档标签已核验，6个旧远端分支已删除，106份流水线候选已按内容去重收入协作分支。

研究总账、DEAD_ENDS、HISTORY与docs/current以新干净副本为审查入口；旧引用可按archive/*标签取回，不重写历史。评估与重放退出、全部证据交付后再执行整理验收测试。之后停止实验推进，等待新的用户决定。

## 资产位置

上层交付/最新为当前F01阶段证据；历史保存已关闭轮次，待发布保存未核验构建；过程文件按主题保存脚本、状态、日志、快照、原始交付构建与图表。过程目录和历史工作树暂保持路径，避免断开证据引用。根目录只保留本入口、AGENTS.md及存量工作环境，不新增脚本或状态。

汇报PNG入口：[阶段基准图](过程文件/阶段基准图_20261009/reports/figures_interim_20261009/README.md)。正文五图、备份两图；历史V3e学习曲线与本轮价值停止训练分开，当前训练出图搁置。旧项目说明仅作历史：[整理前快照](过程文件/整理_20261010/接续/项目说明_整理前快照.md)。
''')
state=json.loads((C/'CURRENT.json').read_text(encoding='utf-8-sig'))
state.update(phase='upper_review_failed_remaining_evidence_active',next_action='Finish existing record evaluations and authorized replays; no new research/training. Local archives copied; tests and read-only transition after workers exit.')
state['upper_decision']={'source':'latest direct user instruction','decision':'experiment failed; remaining evaluations for records only','review':'reviews/F01_271000_UPPER_REVIEW_20261010.md','formal_three_model_readout':'PENDING; not replaced by upper decision'}
state['local_cleanup']={'new_checkout':'D:/py/spacecraft','checkout_commit':'046878c','archive_root':'D:/spacecraft_archive','active_experiment_untouched':True,'verification':'DEFERRED until processes exit','offline_backup':'PENDING','handoff':'lower/handoffs/cleanup_local_20261010/README.md'}
write(C/'CURRENT.json',json.dumps(state,ensure_ascii=False,indent=2))
print('LOCAL_ENTRIES_UPDATED; previous delivery hashes verified; active paths unchanged')
