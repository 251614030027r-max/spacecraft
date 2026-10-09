from pathlib import Path
import json,hashlib,shutil
space=Path('C:/Users/35884/Documents/Spacecraft').resolve();root=Path(__file__).resolve().parents[1]/'reports'/'ppt_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
moves=[]
for name,folder in [('fig_training_return','诊断材料'),('fig_training_success','诊断材料'),('fig_training_combined','重复图归档'),('fig_constraint_response','支撑材料')]:
    for ext in ('png','pdf','svg'):
        source=root/f'{name}.{ext}';dest=root/folder/source.name
        assert source.resolve().is_relative_to(space) and dest.resolve().is_relative_to(space)
        assert source.is_file() and not dest.exists();digest=sha(source);dest.parent.mkdir(exist_ok=True);shutil.move(str(source),str(dest));assert sha(dest)==digest
        moves.append({'from':source.name,'to':dest.relative_to(root).as_posix(),'sha256':digest})
audit='''# 训练曲线与全部图表重新审核

曲线不是绘图错误：逐回合复算累计外层决策步数、回报与50回合滑动均值，均与原绘制CSV相同；l等于learned_decisions加交接决定的1步，MPC后段不混入横轴。completed是任务完成，不能读成干净完成或正式部署成功率。50回合趋势最早出现在第50回合，且当时已超过2000步预热；图上较高的第一段并非未训练策略的0步测试。

训练指标混合了随机任务动作、受[0.002,0.01]限制的随机交接及Pure MPC后段；Monitor回报包括该后段。部署则使用确定性动作和价值规则。因此这不是固定测试集上的确定性性能曲线。50回合窗口及变化的开局、交接与回合长度会带来波动，但不能据此把持续下降解释成无害。manifest明确curriculum_enabled=false，本轮没有证据支持“后期自动变难”解释。

## 现有数据有没有学好

下面按回合结束时的累计决策数分组，末组为40k至各日志快照末端。它是训练描述，非同开局配对评估，也不构成统计显著性结论。

| 种子 | 0–10k完成率 | 40k–快照末完成率 | 0–10k平均回报 | 40k–末平均回报 |
|---|---:|---:|---:|---:|
| 262460 | 41.6% | 43.1% | 1.04 | 2.57 |
| 262461 | 36.8% | 26.7% | −0.67 | −0.96 |
| 262462 | 57.9% | 54.9% | 3.26 | 3.28 |

没有三个种子一致改善的证据。262460仅有有限改善；262461较弱；262462整体接近早期水平。尤其262461快照531回合中，294个未交接回合全部未完成，237个交接回合里137个完成。早期36.8%来自43/117回合，43个完成全部属于交接回合；不是学习策略本身已很强。这是学习分支完成能力的风险信号。条件分组受随机停止与轨迹影响，不可当作正式learned-only成功率或交接因果增益。

## 每张图如何处理

| 图 | 能支持什么 | 不能支持什么 | 处理 |
|---|---|---|---|
| 训练回报 | 真实训练波动、种子分化 | 稳定收敛、性能单调改善 | 移入诊断材料 |
| 训练完成率 | 混合训练行为的回合完成记录 | 确定性部署成功率、学习分支单独效果 | 移入诊断材料 |
| 训练双图 | 上述两项并列 | 新增独立证据 | 重复图归档 |
| 50k三维轨迹 | 同初态两个模型的真实运动及交接位置 | 路径更优；目标旋转坐标弧线等于惯性轨迹 | 演示保留 |
| 状态响应 | 交接后的距离、速度与姿态响应，两个案例真实终止 | 优于Pure或交接救回失败开局 | 演示保留 |
| 价值差与交接 | 第21/39步首次满足规则，状态相关的实际触发 | 价值校准、阈值最优、科学贡献已成立 | 演示保留 |
| 位置误差及裕度 | 真实误差与距离/视场约束，补充闭环证据 | 全部约束仅由两条裕度覆盖 | 移入支撑材料 |

## 展示边界

固定266019案例中Pure为106.8s/1.3405m/s，262460为133.9s/1.7196m/s，262462为161.1s/1.9805m/s，三者均完成且零违规。现有演示图证明机制被执行并形成完成任务的闭环，不能证明协调收益或方法优于Pure。262460由30k到50k该单例有所改善，仍不能外推总体改善。尚缺同模型同块learned-only对照，正式贡献等待60k预注册评估。不会通过改变平滑窗口、剪掉下降段、只画最好种子或换开局制造上升曲线。

结论：删掉的是重复或不适合作为成果展示的图，不是删掉不利证据。原始快照、所有诊断图及本报告均保留。没有追加实验、改变训练或控制方法。
'''
(root/'TRAINING_AND_FIGURE_REVIEW.md').write_text(audit,encoding='utf-8')
index='''# 当前演示版：真实交接与闭环过程

本版仅保留三张演示图；它们验证机制执行与单案例闭环，不代表训练已收敛或优于Pure。两个50k模型、同一266019开局，真实0.1s物理数据。所有PNG400dpi，PDF/SVG同名。

- [三维轨迹](fig_trajectory_3d.png)：共同初态、两种子轨迹和交接位置。
- [状态响应](fig_state_response.png)：距离、速度、姿态误差与真实交接时刻。
- [价值触发](fig_handoff_value.png)：两种子首次价值越零及交接。

支撑页：[误差与约束裕度](支撑材料/fig_constraint_response.png)。

诊断材料：[训练回报](诊断材料/fig_training_return.png) · [训练完成率](诊断材料/fig_training_success.png)。完整双图移入重复图归档，避免PPT重复。

必须先看[训练与全部图表审核](TRAINING_AND_FIGURE_REVIEW.md)。[资产与模型SHA](FIGURE_REPORT.md) · [固定案例审核](FIGURE_AUDIT.md) · [文件SHA](FILES_SHA256.json)。

三个种子未表现出一致改善，262461的未交接训练回合0/294完成。图表不伪装正面性能结论；正式方法是否成立仍由60k同块评估决定。
'''
(root/'README.md').write_text(index,encoding='utf-8')
p=root/'FIGURE_REPORT.md';s=p.read_text(encoding='utf-8');s+='\n\n## 展示目录调整\n\n全部图表重新审核见[TRAINING_AND_FIGURE_REVIEW.md](TRAINING_AND_FIGURE_REVIEW.md)。训练曲线已移入诊断材料，约束图移入支撑材料，重复双图归档；当前入口只保留轨迹、状态响应和价值触发三张，不宣称已学好或性能优势。文件名称与哈希位置以README和FILES_SHA256为准。\n';p.write_text(s,encoding='utf-8')
(root/'FIGURE_SELECTION_RECEIPT.json').write_text(json.dumps({'moves':moves,'training_or_control_changed':False,'new_simulations':0,'diagnostic_data_preserved':True},ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__),root/Path(__file__).name)
files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='FILES_SHA256.json'}
(root/'FILES_SHA256.json').write_text(json.dumps(files,ensure_ascii=False,indent=2),encoding='utf-8');assert all(sha(root/name)==digest for name,digest in files.items())
project=space/'项目说明.md';s=project.read_text(encoding='utf-8');needle='**10月9日汇报图表：**';start=s.index(needle);end=s.index('\n\n',start)
s=s[:start]+needle+' 已重新审核并精简演示版，只保留两个50k模型固定266019开局的三维轨迹、状态响应、价值触发；训练曲线转入诊断材料，约束图作支撑，重复双图归档。三种子训练没有一致改善证据，262461未交接训练回合0/294完成；不能宣称收敛或优于Pure。入口：[演示与完整审核](C:/Users/35884/Documents/Spacecraft/过程文件/汇报图表_20261009/reports/ppt_20261009/README.md)。原始不利证据完整保留，未追加实验或改训练。'+s[end:];project.write_text(s,encoding='utf-8')
print('CURATED_3_DEMONSTRATION_FIGURES',len(moves),'FILES_MOVED',len(files),'HASHES_VERIFIED')
