from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]/'reports'/'ppt_20261009'
p=root/'FIGURE_AUDIT.md';s=p.read_text(encoding='utf-8').replace('| Pure MPC | True | 0 |','| Pure MPC | True | — |');p.write_text(s,encoding='utf-8')
receipt=json.loads((root/'REPLACEMENT_RECEIPT.json').read_text(encoding='utf-8'))
p=root/'plot_data'/'pure_provenance.json';v=json.loads(p.read_text(encoding='utf-8'));v['source']=str(Path(receipt['old_evidence_preserved'])/'plot_data');p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
names=[('fig_training_return','三种子训练回报，原始波动和50回合均值'),('fig_training_success','三种子训练完成率，50回合滑动窗口'),('fig_training_combined','训练回报与完成率双图'),('fig_trajectory_3d','两个50k模型与Pure的目标系三维轨迹'),('fig_state_response','相对距离、速度和姿态误差'),('fig_constraint_response','位置误差、安全距离及视场裕度'),('fig_handoff_value','两个50k模型的价值差与实际交接')]
s='# 最新汇报图表\n\n三种子日志快照截至53388/49945/52348步；轨迹为262460和262462的50k模型，固定开发开局266019。所有PNG400dpi，附PDF/SVG。原30k版已从当前目录移除，历史位置见REPLACEMENT_RECEIPT.json。\n\n'
for name,label in names:s+=f'- [{label}]({name}.png) · [PDF]({name}.pdf) · [SVG]({name}.svg)\n'
s+='\n[科学支撑审核](FIGURE_AUDIT.md) · [资产来源与模型SHA](FIGURE_REPORT.md) · [文件哈希](FILES_SHA256.json)\n\n推荐展示训练双图、三维轨迹、状态响应和价值图；约束图作支撑页。本版能展示机制执行和真实闭环，不宣称收敛或优于Pure。当前案例两个协调模型都完成且零违规，但耗时和Delta-v均高于Pure。\n'
(root/'README.md').write_text(s,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
(root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(root/name)==digest for name,digest in files.items());print('FINAL_HASHES_VERIFIED',len(files))
