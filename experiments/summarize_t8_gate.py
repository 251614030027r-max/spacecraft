import json, shutil
from pathlib import Path
from statistics import mean
from collections import Counter

root=Path('D:/py/DRL2/logs/t8_eval')
out=Path('C:/Users/35884/Documents/Spacecraft/交付资料/T8_GATE1_DELIVERY_20260912')
out.mkdir(exist_ok=True)
summary={'gate':'S4_STOP','S5':'not_started','S3':'not_started','evaluations':{},'replays':{}}
lines=['# T8 阶段交付：第一道闸门触发（2026-09-12）','','## 结论与进度','','S1已完成；S2五个12场景评估与S4四个4场景回放全部串行完成，原始JSON齐全，额度中断没有丢失这些产物。**S4等待假设不成立，按用户闸门停止，S5和S3尚未启动。** 本轮没有优化、改奖励/MPC/任务/反馈或选择新机制。', '', '死种子主要在正半轴动作下耗尽时限。负半轴不是本批死种子长期不完成的共同解释；这不等于否定所有形式的暂存或拖延行为，也不能自动确立探索温度为下一因子。', '', '## S2：正式评估汇总', '', '四模型固定final_model、deterministic=True；共同场景262000–262011；h20；无并行训练。时间与力冲量按各自成功回合取均值，成功集合不同，不能解释成逐场景配对效应。', '', '| 模型/对照 | 完成 | 成功平均时间s | 成功平均力冲量N·s | 真值违约回合比例 | 控制p95 ms / 100ms预算 |', '|---|---:|---:|---:|---:|---:|']
def fmt(x): return '—' if x is None else f'{x:.3f}'
for p in sorted(root.glob('eval_*.json')):
    d=json.loads(p.read_text()); m=d['main_table']; assert len(d['records'])==12
    row={'completed':m['completed_episodes'],'episodes':12,'time_s':None if m['completion_time_s'] is None else m['completion_time_s']['mean'],'force_n_s':None if m['force_impulse_n_s']['completed_only'] is None else m['force_impulse_n_s']['completed_only']['mean'],'all_force_n_s':m['force_impulse_n_s']['all_episodes']['mean'],'violation_rate':m['constraint_violation_rate'],'p95_ms':m['per_step_compute_s']['controller']['p95']*1000,'p95_over_budget':m['per_step_compute_s']['controller_p95_over_budget'],'worst_truth_normalized_margin':None}
    summary['evaluations'][p.stem]=row
    lines.append(f"| {p.stem} | {row['completed']}/12 | {fmt(row['time_s'])} | {fmt(row['force_n_s'])} | {row['violation_rate']:.1%} | {row['p95_ms']:.2f} / {row['p95_over_budget']:.3f}× |")
    shutil.copy2(p,out/p.name)
lines += ['', '两版本均为7/12与0/12，按两个训练种子等权平均均为29.17%，没有反馈提升完成率的证据。h20同场景对照8/12；不能仅据各自成功均值下严格配对燃料结论。零成功V3/V2的全场景平均力冲量分别1250.789、1472.593 N·s，未完成并不意味着省燃料。', '', '**裕度列修正：** 这些JSON在T8B之前生成，旧worst_constraint_margin是未区域门控、不同单位的逐字段原始最小值，不作为安全性或主表归一化裕度列；遵从上层说明，本轮保留正确门控的constraint_violation_rate。minimum_truth_normalized_margin统一记缺失，不从回放低频采样拼补。本轮S3被闸门停止，因此“随第三种子评估补新裕度”的计划也未执行。', '', 'Pure MPC主表按上层选择h35：9/12、88.5s、167.2 N·s，旧300步样本p95 77.44ms，超预算2/300；h50为269/300。S2新h20对照补出了自己的力冲量208.181 N·s（8个成功回合），仅补h20，不跨时域填h50。h35、h20质量和compute来源范围不同，必须分别标注。旧h50燃料仍缺，不能写h35比h50省燃料。样本max在第0步不等于全程永不发生冷启动。', '', '## S4问题1：动作负半轴占比、均值与极值', '', '总体占比按记录的逐决策动作数加权；另报4场景等权平均，防止短暂失败场景影响解释。JSON动作保留5位小数，极近零动作存在舍入精度限制。', '', '| 策略 | 负动作数/决策数 | 决策加权比例 | 场景等权比例 | a_commit均值 | 最小 / 最大 |', '|---|---:|---:|---:|---:|---:|']
detail=[]
for p in sorted(root.glob('replay_*.json')):
    d=json.loads(p.read_text()); eps=d['episodes']; assert len(eps)==4
    trace=[r for e in eps for r in e['trace']]; actions=[r['action'][0] for r in trace]
    feedback={k:{'min':min(r[k] for r in trace),'max':max(r[k] for r in trace)} for k in ('hybrid_feedback_fallback_fraction','hybrid_feedback_solved_peak_slack','hybrid_feedback_mean_actuator_usage')}
    row={'decisions':len(trace),'negative':sum(x<0 for x in actions),'fraction_negative':mean(x<0 for x in actions),'scene_mean_fraction':mean(e['fraction_commit_negative'] for e in eps),'action_mean':mean(actions),'action_min':min(actions),'action_max':max(actions),'true_range_min':min(r['true_range_m'] for r in trace),'true_range_max':max(r['true_range_m'] for r in trace),'feedback':feedback,'ended_by':dict(Counter(x for e in eps for x in e['ended_by'])),'scenes':[]}
    lines.append(f"| {p.stem} | {row['negative']}/{len(trace)} | {row['fraction_negative']:.2%} | {row['scene_mean_fraction']:.2%} | {mean(actions):.5f} | {min(actions):.5f} / {max(actions):.5f} |")
    for e in eps:
        es={'seed':e['seed'],'completed':e['completed'],'ended_by':e['ended_by'],'fraction_commit_negative':e['fraction_commit_negative'],'action_mean':e['action_mean'][0],'action_min':e['action_min'][0],'action_max':e['action_max'][0],'range_min':min(r['true_range_m'] for r in e['trace']),'range_max':max(r['true_range_m'] for r in e['trace'])}
        row['scenes'].append(es)
        detail.append(f"| {p.stem} / {e['seed']} | {e['fraction_commit_negative']:.2%} | {es['action_mean']:.5f} | {es['action_min']:.5f} / {es['action_max']:.5f} | {','.join(e['ended_by'])} | {es['range_min']:.4f}–{es['range_max']:.4f} |")
    summary['replays'][p.stem]=row
    shutil.copy2(p,out/p.name)
lines += ['', '**触闸依据：** V2死种子的负半轴占比并不高于活种子。V3即使场景等权均值偏高，也由262001短暂视场失败的90%负动作拉高；它在三个超时场景的负动作比例是0%、0%、1.33%，不支持“靠负半轴长期等待而超时”。V2三个超时场景对应0%、0%、6.67%。不能用一个短暂失败场景替代长期等待机制。', '', '## S4问题2和3：终止原因与真实距离', '', '两个死种子在共同4场景均为3次time_failure、1次fov_failure。活种子各2次完成，另有V3一次fov_failure与一次distance_failure，V2两次fov_failure。完成回合ended_by=none_flagged表示成功不在失败标志列表中，不是未知失败。', '', '以下范围来自每决策结束时true_range_m，约2秒采样且不含初始reset，不是完整底层轨迹连续极值。', '', '| 策略/场景 | 负半轴占比 | 动作均值 | 动作最小/最大 | 终止标志 | 采样真实距离m |', '|---|---:|---:|---:|---|---:|']+detail
lines += ['', '## S4问题4：反馈是否近乎常量', '', '| 策略 | 回退比例范围 | 有效解归一化最大松弛范围 | 平均执行器使用率范围 |', '|---|---:|---:|---:|']
for name,row in summary['replays'].items():
    vals=[f"{v['min']:.5f}–{v['max']:.5f}" for v in row['feedback'].values()]
    lines.append('| '+name+' | '+' | '.join(vals)+' |')
lines += ['', '范围只回答是否有变化，不能凭极差衡量携带信息量、非平稳程度或策略利用程度；窄分布与少量离群值需要另行时序统计。V2虽未把反馈输入策略，环境仍计算这些info统计，因此可用于同口径诊断。', '', '## T8B补丁与下一步边界', '', 'T8B共6条，其中第一条Pure MPC文档已存在，git am -3识别为Patch already applied，其余5条正常应用。代码变更修复评估裕度记录，不改变已完成JSON，不重跑旧队列。新CLAUDE.md、单因子核对和姿态参考探针资料已纳入；姿态探针结论为上游提供的summary，本地未重跑。', '', '本次不执行S5，不启动S3补训。耗时分解和两个新训练目录因此不存在，不用旧训练冒充。上层需先裁决S4反证，再明确是否解除闸门及后续步骤；不会自行启动探索温度、姿态参考或预测反馈新机制。', '', '交付包包含本报告、summary.json、五评估JSON和四回放JSON；原运行日志仍在D:/py/DRL2/logs/t8_eval。复算脚本为summarize_t8_gate.py。']
(out/'上层交付报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
shutil.copy2(__file__,out/'summarize_t8_gate.py')
print(json.dumps({k:{f:v[f] for f in ('fraction_negative','scene_mean_fraction','true_range_min','true_range_max','feedback')} for k,v in summary['replays'].items()},ensure_ascii=False))
