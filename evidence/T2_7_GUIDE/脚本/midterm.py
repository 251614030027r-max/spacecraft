"""Read-only training-log snapshot and report. No model loading or scientific execution."""
from pathlib import Path
import json,statistics,hashlib,datetime,shutil
import psutil
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
R=Path(__file__).resolve().parents[1];BASE=R.parent;SPACE=R.parents[1]
STAMP=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
OUT=R/'发布快照'/('midterm_'+STAMP);OUT.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sample(rows):
 out={'n':len(rows),'mean_true_return':statistics.mean(x['return_true'] for x in rows) if rows else None,'capture':sum(x['success'] for x in rows),'step_range':[rows[0]['steps'],rows[-1]['steps']] if rows else None,'event_counts':{k:sum(x['event']==k for x in rows) for k in sorted({x['event'] for x in rows})}}
 for u in (True,False):
  v=[x for x in rows if x['mission']['urgent']==u];out['urgent' if u else 'economy']={'n':len(v),'capture':sum(x['success'] for x in v),'mean_true_return':statistics.mean(x['return_true'] for x in v) if v else None}
 return out
def main():
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 groups={};provenance={}
 for topic,arms in (('T2_7_GUIDE_20261031',('e_mve','e_sac')),('T2_6W1_CUTOFF_20261029',('e_mve',)),('T2_6W_WIDE_140_200',('e_sac',))):
  for arm in arms:
   for s in (0,1):
    tag=f'{arm}_s{s}';src=BASE/topic/f'B/TRAIN_{tag}';file=src/'live/episodes.jsonl';raw=file.read_bytes();raw=raw[:raw.rfind(b'\n')+1];rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
    manifest=read(src/'manifest.json');progress=read(src/'live/progress.json')
    key=topic+'/'+tag;v={'progress':progress,'manifest':manifest,'common2500_3000':sample([x for x in rows if 2500<x['steps']<=3000])}
    if topic=='T2_7_GUIDE_20261031':
     assert manifest['git_head']=='2208a5176e88718c2c1db3e4a19c58853da49739'
     d=OUT/'训练快照'/tag;d.mkdir(parents=True);(d/'episodes.jsonl').write_bytes(raw);dump(d/'manifest.json',manifest);dump(d/'progress.json',progress)
     v['latest50']=sample(rows[-50:]);v['by_window']={f'{lo}-{hi}':sample([x for x in rows if lo<x['steps']<=hi]) for lo,hi in ((0,1000),(1000,2000),(2000,2500),(2500,3000))}
     e=EventAccumulator(str(src/'live/tensorboard'),size_guidance={'scalars':0});e.Reload();v['mechanism_recent']={}
     for k in ('critic/grad_cos','model/two_stage_share','model/decisive_share','model/w2_mean','model/pi_gap_sigma_median','model/stage2_cutoff_share','model/entry_bias_now'):
      if k in e.Tags()['scalars']:
       z=e.Scalars(k)[-20:];v['mechanism_recent'][k]={'n':len(z),'mean':statistics.mean(x.value for x in z),'step_range':[z[0].step,z[-1].step]}
     dump(d/'mechanism_recent.json',v['mechanism_recent']);shutil.copy2(src/'live/TORCH_THREADS.json',d/'TORCH_THREADS.json')
    else:
     dump(OUT/'历史必要对照'/topic/(tag+'.json'),{'source':str(file),'source_manifest':manifest,'window':'2500<completed_decision_steps<=3000','rows':[x for x in rows if 2500<x['steps']<=3000]})
    provenance[key]={'source_file':str(file),'snapshot_prefix_bytes':len(raw),'snapshot_prefix_SHA256':hashlib.sha256(raw).hexdigest(),'manifest_SHA256':sha(src/'manifest.json')};groups[key]=v
 state=read(R/'RUN_STATE.json');dump(OUT/'RUN_STATE_SNAPSHOT.json',state);dump(OUT/'MIDTERM_DATA.json',groups);dump(OUT/'PROVENANCE.json',provenance)
 for n in ('EARLY1000.json','pytest.log'):shutil.copy2(R/n,OUT/n)
 # Full observed training curves; no simulation, interpolation or model inference.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for j,arm in enumerate(('e_mve','e_sac')):
  for s in (0,1):
   ax=axes[s,j]
   for topic,label,color in (('T2_7_GUIDE_20261031','T2.7 guided','#1565c0'),('T2_6W1_CUTOFF_20261029' if arm=='e_mve' else 'T2_6W_WIDE_140_200','prior unguided','#bb6818')):
    p=BASE/topic/f'B/TRAIN_{arm}_s{s}/live/episodes.jsonl';rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()];rows=[x for x in rows if x['steps']<=groups['T2_7_GUIDE_20261031/'+f'{arm}_s{s}']['progress']['steps']];ys=[x['return_true'] for x in rows]
    ax.plot([x['steps'] for x in rows],[statistics.mean(ys[max(0,i-49):i+1]) for i in range(len(ys))],label=label,color=color)
   ax.axvline(2000,ls='--',color='gray');ax.set(title=f'{arm} seed{s}',xlabel='Completed decision steps',ylabel='True return (50 episode mean)');ax.legend();ax.grid(alpha=.2)
 fig.savefig(OUT/'MIDTERM_CURVES.png',dpi=160);plt.close(fig)
 lines=['# T2.7 本机训练中期报告（供上层审查）','',f'快照时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}。状态：**训练进行中，尚无本轮部署评估或正式判定**。本报告只读取现有日志与历史固定结果；未运行新评估、真值分支或改变训练。','', '## 一、范围、版本及结论摘要','', '执行书collab `8708f39`；科学冻结 `2208a5176e88718c2c1db3e4a19c58853da49739`；环境t2-stage-0.5.0，学习器t2-learner-2.7。新训2×2中的引导列：EACC/SAC×seed0/1，均从零4000步、各2异步worker，p(t)=0.5 max(0,1−t/2000)。复用列：T2.6W SAC和T2.6W1 EACC。紧急U(140,200)s、经济U(280,340)s，各0.5，预算3m/s。四条Normal进程共享CPU0–11，OMP/MKL/OPENBLAS=1，实际PyTorch intra-op=1。52测试与两臂真实引导日志/事件读取验收通过，滚动约30秒，未重启、未热改观察器。','', '**阶段判断：存在局部改善，但未形成一致、跨种子的优势。** 相同2500–3000步窗口内，EACC seed0改善、seed1接近历史；SAC seed0退化、seed1改善。撤引导后没有一致性崩溃，但仍有种子差异与局部波动。训练分布和策略均随轮次改变，这些对照不能替代共同开局的确定性部署配对。','', '## 二、当前真实进度','', '|训练|完成步数/4000|优化更新|完成回合|墙钟小时|累计引导/已存转移|重启次数|','|---|---:|---:|---:|---:|---:|---:|']
 for arm in ('e_mve','e_sac'):
  for s in (0,1):
   tag=f'{arm}_s{s}';p=groups['T2_7_GUIDE_20261031/'+tag]['progress'];lines.append(f'|{tag}|{p["steps"]}/4000|{p["updates"]}|{p["episodes"]}|{p["wall_s"]/3600:.2f}|{p["replay"].get("guided",0)}/{p["steps"]}|{state.get("attempts",{}).get(tag,"MISSING")}|')
 lines+=['','上述计数取各自实际日志时间，彼此可能相差几十秒；不将预计值当完成数。四条均已超过2000步，累计引导计数随后保持不变，当前是策略自主采样阶段。此前16–18小时仅为上层预算估计；尚未完成，不承诺精确结束时刻。','', '## 三、同决策步数窗口的历史对照','', '统一窗口为完成决策步数2500<steps≤3000；不比较一轮当前尾部与另一轮最终尾部。样本为该窗口内完成的训练回合，紧急/经济数和轨迹不同，不是配对试验。','', '|方法|种子|轮次|回合数|平均真实回报|成功/全部|紧急成功/样本|经济成功/样本|','|---|---|---|---:|---:|---:|---:|---:|']
 for arm in ('e_mve','e_sac'):
  for s in (0,1):
   old='T2_6W1_CUTOFF_20261029' if arm=='e_mve' else 'T2_6W_WIDE_140_200'
   for topic,label in ((old,'历史无引导'),('T2_7_GUIDE_20261031','本轮有引导历史、2000后自主')):
    v=groups[topic+f'/{arm}_s{s}']['common2500_3000'];lines.append(f'|{arm}|{s}|{label}|{v["n"]}|{v["mean_true_return"]:.3f}|{v["capture"]}/{v["n"]}|{v["urgent"]["capture"]}/{v["urgent"]["n"]}|{v["economy"]["capture"]}/{v["economy"]["n"]}|')
 lines+=['','历史固定来源：SAC科学证据605f3e4；EACC科学证据7dac3dd（原生no_gain，四点EACC−SAC为−8.212150/+19.685933，紧急捕获差−7/+15）。必要原始窗口回合随本报告提供，位置与哈希见PROVENANCE。','', '## 四、当前尾部读数及波动','', '|训练|最近50回合步数范围|平均真实回报|成功|紧急成功/样本|经济成功/样本|','|---|---|---:|---:|---:|---:|']
 for arm in ('e_mve','e_sac'):
  for s in (0,1):
   v=groups[f'T2_7_GUIDE_20261031/{arm}_s{s}']['latest50'];lines.append(f'|{arm}_s{s}|{v["step_range"]}|{v["mean_true_return"]:.3f}|{v["capture"]}/{v["n"]}|{v["urgent"]["capture"]}/{v["urgent"]["n"]}|{v["economy"]["capture"]}/{v["economy"]["n"]}|')
 lines+=['','尾部窗口用于看现场波动，步数和任务样本各异，不用它给四臂排名。完整已观测曲线见MIDTERM_CURVES.png；曲线截至各自快照，不补点、不外推。回报是真实任务奖励累计，非塑形训练奖励。','', '## 五、引导与模型反馈实际运行证据','', '千步精确快照：四条引导比例25.5%、27.4%、23.6%、31.4%；EACC两种子first_model_step均300。千步早报已于collab9965b39推送。最近20条实际机制探针如下，列样本数和步数范围；它们只证明反馈参与学习，不证明标签增量或部署收益。','', '```json',json.dumps({f'{arm}_s{s}':groups[f'T2_7_GUIDE_20261031/{arm}_s{s}']['mechanism_recent'] for arm in ('e_mve',) for s in (0,1)},ensure_ascii=False,indent=2),'```','', '## 六、本轮必须回答的问题、尚缺证据及建议','', '1. **完整方案能否两种子重复获益：尚不能判。** 等4000完成及1000/2000/3000/4000×48共同开局部署评估，以t27原生判定为准。E1要求每个种子四点均差>0且种子均值≥+5；U要求紧急捕获累计总增≥4且每种子不为负；E2终点种子平均≥−3；S每种子unsafe总计≤SAC+1且终点捕获≥SAC−1；M保持原机制阈值并要求guided_share≥0.03；C0每点48/48、无ERROR。阈值不变。','', '2. **收益来自引导还是标签：尚不能判。** 主比较EACC+引导−SAC只回答整体组合；SAC+引导−SAC回答引导作用；EACC+引导−SAC+引导回答标签在引导之上的增量；EACC+引导−原EACC为辅助参照。当前SAC两个种子反向变化，不能把个别改善直接归给引导或标签。','', '3. **2000步撤引导后能否自主保持：已有训练线索，尚缺部署验证。** 累计guided计数不再增长，2500–3000训练仍有较高捕获，但本轮局部尾部回落及seed差异说明不能仅凭训练成功率宣布保持成立。需按原计划的3000/4000部署与late_*读数回答。','', '建议：继续原冻结训练至4000，不提前停止、不加训、不调p或标签/网络；完成后按四配对与紧急/经济、压力分档、共同成功燃料/时间、逐开局中位差及剔除最大两例报告。若最后只有引导有效，明确写“标签没有额外增量”；若种子仍分化，保留no_gain/partial等原生判定，不挑种子、检查点或截止档。中期现象不足以授权下一轮机制修改。','', '## 七、交付与缺项说明','', '**MISSING：本轮部署评估、四个最终READOUT、DECISION、最终模型及最终墙钟。** 尚未生成即保持缺项，不借历史结果代替。附件含四条原始训练回合快照、manifest、progress、实际线程回执、机制读数、必要历史原始窗口、RUN_STATE快照、EARLY1000、pytest、源文件固定路径及SHA256；SHA256SUMS覆盖本期文件。当前运行日志仍会增长，PROVENANCE记录已保存前缀的字节数与SHA，不能把它误认为日后完整活动文件的哈希。','', '过程目录：`'+str(R)+'`。本中期快照独立于最终交付，最新入口更新为本期中期报告；训练继续，最终交付完成后按共用入口规则归档本期。']
 (OUT/'RESULT_MIDTERM.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 (OUT/'SHA256SUMS').write_text('\n'.join(sha(p)+'  '+p.relative_to(OUT).as_posix() for p in sorted(OUT.rglob('*')) if p.is_file())+'\n',encoding='utf-8')
 dump(R/'记录/MIDTERM_BUILD.json',{'snapshot':str(OUT),'report_SHA256':sha(OUT/'RESULT_MIDTERM.md'),'timestamp':STAMP});print(str(OUT))
if __name__=='__main__':main()
