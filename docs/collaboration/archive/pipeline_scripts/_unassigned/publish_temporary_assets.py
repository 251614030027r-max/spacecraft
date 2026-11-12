from pathlib import Path
import shutil,json,hashlib
from datetime import datetime
space=Path('C:/Users/35884/Documents/Spacecraft');source=space/'过程文件/汇报图表_20261009/reports/ppt_20261009';repo=space/'过程文件/协作/Git工作树'
rel=Path('docs/collaboration/lower/handoffs/ppt_temporary_20261009');dest=repo/rel
assert not dest.exists();dest.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_hash=json.loads((source/'FILES_SHA256.json').read_text(encoding='utf-8'))
included=[]
for name in ('fig_trajectory_3d','fig_state_response','fig_handoff_value'):
    included += [name+'.'+ext for ext in ('png','pdf','svg')]
for folder,names in [('诊断材料',('fig_training_return','fig_training_success')),('支撑材料',('fig_constraint_response',))]:
    included += [folder+'/'+name+'.'+ext for name in names for ext in ('png','pdf','svg')]
included += ['FIGURE_REPORT.md','FIGURE_AUDIT.md','TRAINING_AND_FIGURE_REVIEW.md','make_latest_figures.py','make_figures.py','finalize_latest_figures.py','audit_training_interpretation.py']
included += [p.relative_to(source).as_posix() for p in (source/'plot_data').glob('*') if p.is_file()]
for name in included:
    assert sha(source/name)==source_hash[name];p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p);assert sha(p)==source_hash[name]
(dest/'.gitattributes').write_text('* -text\n',encoding='ascii')
readme='''# 临时汇报资产：待上层审查，可替换

状态：TEMPORARY_REVIEW_ONLY / NOT_PAPER_PERFORMANCE。用户2026-10-09授权推送供上层判断产出价值；后续更好结果可取代本版，不据此改方法或追加训练。科学提交f2f8169acd580ea96a578d45022dd8952447eca5，与本协作提交用途不同。

## 三张演示图

- [三维轨迹](fig_trajectory_3d.png)
- [状态响应](fig_state_response.png)
- [价值差与实际交接](fig_handoff_value.png)

它们展示262460、262462的50k模型在固定开发开局266019的真实闭环，同初态Pure作为对照；真实0.1s物理状态，原科学加载、环境与决策逻辑，未筛选新案例。两个模型均完成且零违规，第21/39步交接；133.9/161.1s，Delta-v 1.7196/1.9805m/s。Pure为106.8s、1.3405m/s；该案例无效率优势，也没有同模型learned-only对照，不能证明协调贡献或方法优于Pure。

## 不利证据与支撑

- [完整训练与图表审核](TRAINING_AND_FIGURE_REVIEW.md)：三种子没有一致改善，262461未交接训练回合0/294完成，交接回合137/237完成；这是条件统计，不是正式learned-only性能或因果效应。
- [训练回报](诊断材料/fig_training_return.png) · [训练回合完成率](诊断材料/fig_training_success.png)：日志快照末步53388/49945/52348；50回合尾随均值。训练混合探索与随机MPC交接，不是确定性部署性能。
- [误差与约束裕度](支撑材料/fig_constraint_response.png)
- [模型SHA、配置与来源](FIGURE_REPORT.md) · [固定案例结果审核](FIGURE_AUDIT.md)
- [原始快照与0.1s导出](plot_data/) · [所有包内文件SHA256](FILES_SHA256.json)

PNG400dpi，PDF/SVG同名。仅足够独立审查的本次数据、模型哈希及绘图脚本入Git；模型不上传，重复组合图及旧图不上传。本目录.gitattributes禁止文本换行转换，确保公开包内文件字节SHA可复核。脚本依赖本机科学工程，固定模型、目录和提交均在脚本/来源报告中说明。

## 请上层判断

1. 三张演示图是否足以支持“价值判据执行与单案例闭环”的阶段汇报，哪些应剔除。
2. 三种子分化及262461未交接回合全失败是否需要在正式评估后的机制分析中重点审查；不据此修改当前预注册或提前宣布方法成立/失败。
3. 后续60k正式结果能否替换本临时入口；替换时注明新提交并保留本版固定提交供追溯，不将本版进入论文性能表。

主训练仍由用户独立启动，进程未改。30k结构门控三个模型已通过。60k正式评估/判读尚未交付，本包不替代FINAL_RERUN执行单或正式交付。
'''
(dest/'README.md').write_text(readme,encoding='utf-8')
files={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
(dest/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8')
assert all(sha(dest/name)==digest for name,digest in files.items())
handoff=repo/'docs/collaboration/lower/handoffs/ppt_temporary_20261009.md'
handoff.write_text('# 临时汇报资产交接（2026-10-09）\n\n状态：待上层审查、可被后续更好结果取代，非论文性能证据。用户直接授权本次推送。\n\n[完整临时图表与独立审查证据](ppt_temporary_20261009/README.md)，科学提交f2f8169，两个50k模型262460/262462、固定266019开局；三种子训练快照截至53388/49945/52348步。三张演示图、两张不利训练诊断图、约束支撑、原始CSV/JSON、模型SHA及文件SHA均提供。\n\n结论仅为原部署判据执行与单案例闭环完成，不证明收敛、价值校准、协调增益或优于Pure。该案例两个模型耗时和Delta-v均高于Pure；262461未交接训练回合0/294完成。请审查图表产出价值和解释边界；当前正式训练/预注册不变。后续替代保留本版固定提交引用。\n',encoding='utf-8')
index=repo/'docs/collaboration/lower/README.md';s=index.read_text(encoding='utf-8');s+='\n\n**最新临时汇报资产（2026-10-09，待上层审查，可替换）：** [交接](handoffs/ppt_temporary_20261009.md) · [图表及证据](handoffs/ppt_temporary_20261009/README.md)。不是正式交付或论文性能；含不利训练诊断和全部本轮复核数据。\n';index.write_text(s,encoding='utf-8')
current=repo/'docs/collaboration/CURRENT.json';v=json.loads(current.read_text(encoding='utf-8'))
v.update(verified_date_jst='2026-10-09',phase='formal_training_active_structure_passed',training_started=True,training_launcher='user_manual_independent_windows_powershell',training_progress_snapshot={'262460':54405,'262461':50737,'262462':53184},structure_gate_30k='three models D1-D4 passed; 144 openings completed',next_action='Complete existing 60k runs and prescribed formal evaluation; temporary PPT assets await upper review',temporary_asset_handoff='lower/handoffs/ppt_temporary_20261009.md',temporary_asset_status='TEMPORARY_REVIEW_ONLY_REPLACEABLE_NOT_PAPER_PERFORMANCE')
current.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt={'package':str(dest),'files':len(files),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'package_sha_manifest':sha(dest/'FILES_SHA256.json'),'source_unchanged':True}
(space/'过程文件/汇报图表_20261009/记录/TEMP_PUBLICATION_PREPARATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
