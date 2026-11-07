"""Evidence transport/reporting only. No simulation or scientific code changes."""
from pathlib import Path
import json, os, time, subprocess, shutil, math
import runtime as a
R=a.R; HANDOFF=a.COLLAB/'docs/collaboration/lower/handoffs/t2_6w1_cutoff_20261029'
def collab_commit(files,message,receipt):
 try:
  assert not a.git('diff','--cached','--name-only',repo=a.COLLAB),'unrelated staged files'
  a.git('fetch','origin','collab/spacecraft',repo=a.COLLAB)
  a.git('merge','--ff-only','origin/collab/spacecraft',repo=a.COLLAB)
  rel=[str(Path(p).relative_to(a.COLLAB)).replace('\\','/') for p in files]
  a.git('add','--',*rel,repo=a.COLLAB)
  if a.git('diff','--cached','--name-only',repo=a.COLLAB):a.git('commit','-m',message,repo=a.COLLAB)
  head=a.git('rev-parse','HEAD',repo=a.COLLAB)
  a.git('push','origin','HEAD:refs/heads/collab/spacecraft',repo=a.COLLAB)
  remote=a.git('ls-remote','origin','refs/heads/collab/spacecraft',repo=a.COLLAB).split()[0];assert remote==head
  out={'jst':a.now(),'status':'pushed_verified','commit':head}
 except Exception as e:out={'jst':a.now(),'status':'publication_pending','error':str(e)}
 a.dump(R/'记录'/receipt,out);return out
def early():
 HANDOFF.mkdir(parents=True,exist_ok=True)
 raw=R/'MECHANISM1000.json';dest=HANDOFF/'MECHANISM1000.json';shutil.copy2(raw,dest)
 text='# T2.6W1 1000步机制早报\n\n冻结科学 `'+a.SHA+'`，执行书 `185765b`。两条EACC训练继续，不按早报读数停训或调整。以下均为实际记录且 steps<=1000 的反馈探针；每项列均值和样本数，首个模型步单独列。stage2_cutoff_share>0 仅表示截止修正在样本中实际触发，不等于性能增益。\n\n```json\n'+raw.read_text(encoding='utf-8')+'\n```\n'
 report=HANDOFF/'RESULT_1000.md';report.write_text(text,encoding='utf-8')
 return collab_commit([dest,report,a.CURRENT],'evidence: T2.6W1 1000-step mechanism readings; training continues','EARLY_PUSH_RECEIPT.json')
def targets():
 rank=a.read(R/'D_RANK.json');out={'readings':rank,'decision_rules':'Pre-registered section3; diagnostic only; not the t26w verdict','matches':[]}
 both=all(rank[f'e_mve_s{s}_step_4000']['critic']['accuracy'] is not None and rank[f'e_sac_s{s}_step_4000']['critic']['accuracy'] is not None and rank[f'e_mve_s{s}_step_4000']['critic']['accuracy']>rank[f'e_sac_s{s}_step_4000']['critic']['accuracy'] for s in (0,1))
 if both:out['matches'].append('两个种子的EACC critic关键对准确率均高于SAC；若性能未转正，下一步对象为Actor更新/探索。')
 for s in (0,1):
  r=rank[f'e_mve_s{s}_step_4000'];c=r['critic']['accuracy']
  better=[k for k in ('label1','label2') if c is not None and r[k]['accuracy'] is not None and r[k]['accuracy']>c]
  if better:out['matches'].append(f'种子{s}：{better}准确率高于自身critic，符合核查融合方式的分支。')
  for k in ('label1','label2'):
   v=r[k]
   if v['accuracy'] is not None and (v['accuracy']<=.5 or (v['decisive_pairs']>0 and v['tie']>.5*v['decisive_pairs'])):out['matches'].append(f'种子{s} {k}：准确率<=0.5或多数关键对并列，符合核查候选信息构造的分支。')
  base=r['label1_abs_error']['bootstrap'];diff=r['diff_abs_error']['bootstrap']
  out[f'seed{s}_bootstrap_error']={'absolute':base,'difference':diff,'diff_over_label1':diff['median']/base['median'] if base['median'] not in (None,0) and diff['median'] is not None else None}
 out['limits']=['执行书没有给明显更小的数值门槛，报告误差和比率，不事后造门槛。','各检查点审计沿各自策略访问状态，策略后续真值也各自不同；四个检查点汇总不是跨策略共同状态的配对因果估计。每个决策内部四种打分在同一候选组上对照相同真值。','label2为审计工具定义的两阶段目标，非完整生产软Retrace融合目标。审计不训练、不实施候选差替换。','多个预登记解释可以同时匹配；不足以唯一归因时保留不确定性，不强行归到一行。']
 a.dump(R/'MECHANISM_INTERPRETATION.json',out);return out
def plots():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 import numpy as np
 out=R/'图';out.mkdir(exist_ok=True);readout=a.read(R/'C_READOUT.json');old=a.read(a.W/'C_READOUT.json')
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for s in (0,1):
  for arm,root,color in (('e_mve',R,'#1565c0'),('e_sac',a.W,'#c55b12')):
   e=EventAccumulator(str(root/f'B/TRAIN_{arm}_s{s}/live/tensorboard'),size_guidance={'scalars':0});e.Reload()
   rows=e.Scalars('rollout/return_true');x=[v.step for v in rows];y=[v.value for v in rows]
   assert rows and all(math.isfinite(v) for v in y)
   axes[s,0].plot(x,y,color=color,alpha=.15,lw=.4)
   sm=[sum(y[max(0,i-49):i+1])/min(i+1,50) for i in range(len(y))]
   axes[s,0].plot(x,sm,color=color,label=arm+' (50 episode mean)')
  axes[s,0].set(xlabel='Completed decision steps',ylabel='True episode return',title=f'Seed {s}: complete training curve');axes[s,0].legend()
  for field,color in (('mean','#1565c0'),('urgent','#bd2828'),('economy','#2e7d32')):
   ys=[readout['paired'][f's{s}/step_{k}'][field] for k in a.MARKS];axes[s,1].plot(a.MARKS,ys,'o-',label=field,color=color)
  axes[s,1].plot(a.MARKS,[old['paired'][f's{s}/step_{k}']['mean'] for k in a.MARKS],'--',color='gray',label='T2.6W before fix')
  axes[s,1].axhline(0,color='black',lw=.7);axes[s,1].set(xlabel='Decision steps',ylabel='Paired EACC - SAC return',title=f'Seed {s}: registered evaluations');axes[s,1].legend()
 for ax in axes.flat:ax.grid(alpha=.2)
 fig.savefig(out/'CURVES.png',dpi=180);fig.savefig(out/'CURVES.pdf');plt.close(fig)
 return list(out.iterdir())
def verify_logs():
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 out={}
 for s in (0,1):
  root=R/f'B/TRAIN_e_mve_s{s}';e=EventAccumulator(str(root/'live/tensorboard'),size_guidance={'scalars':0});e.Reload();tags=e.Tags()['scalars']
  assert not (root/'live/LOGGER_ERROR.json').exists()
  required=['progress/steps','progress/updates','progress/episodes','rollout/return_true','critic/real_td','critic/q_pi_mean','critic/model_td','model/stage2_cutoff_share']
  assert all(k in tags and e.Scalars(k) for k in required)
  for t in tags:
   v=e.Scalars(t);assert all(math.isfinite(x.value) for x in v) and all(y.step>=x.step for x,y in zip(v,v[1:]))
  assert e.Scalars('progress/steps')[-1].value==4000
  eps=a.read(root/'episodes.json');logged=[json.loads(x) for x in (root/'live/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
  assert len(eps)==len(logged)>0 and [(x['steps'],x['worker'],x['event']) for x in eps]==[(x['steps'],x['worker'],x['event']) for x in logged]
  out[f'e_mve_s{s}']={'completed_episodes':len(eps),'all_tags':{t:len(e.Scalars(t)) for t in tags},'events':[{'path':str(p),'SHA256':a.sha(p)} for p in (root/'live/tensorboard').glob('events.out*')],'finite_and_nondecreasing':True,'source_local_only':True}
 a.dump(R/'记录/FINAL_EVENT_VERIFICATION.json',out)
def report(interp):
 dec=a.read(R/'DECISION.json');rd=a.read(R/'C_READOUT.json');old=a.read(a.W/'C_READOUT.json')
 lines=['# T2.6W1 截止终止修复及动作排序核验', '',f'完成时间：{a.now()}。原生判定：**{dec["verdict"]}**。执行书 collab `185765b`；训练/评估冻结 `{a.SHA}`，机制审计 `{a.AUDIT}`。主张仍为执行反馈独立改善SAC长期决策；实际结论以如下判定与证据为限。','', '## 版本、配置与覆盖','', '环境 t2-stage-0.4.3、学习器 t2-learner-2.6.1；EACC两种子各从零4000步、2个异步worker。紧急U(140,200)s、经济U(280,340)s各0.5。只改已冻结的截止/预算终止逻辑；未加预测深度、直接下降、入口偏差修正或候选差训练。每种子4检查点×48开局；新EACC384例、原样复用SAC384例，总768例，294000–294047，原始JSON随附。SAC无重训、无重评。系统参照为T2.6W A_ref 336例（固定位置/哈希见PROVENANCE）。','', '## 原生判定（不改门槛）','','```json',json.dumps(dec,ensure_ascii=False,indent=2),'```','', '## 配对曲线与历史对照','','|种子|步数|本轮EACC−SAC|剔除最大两例|紧急|经济|捕获EACC/SAC|unsafe EACC/SAC|修复前EACC−SAC|','|---|---:|---:|---:|---:|---:|---|---|---:|']
 for s in (0,1):
  for k in a.MARKS:
   p=rd['paired'][f's{s}/step_{k}'];o=old['paired'][f's{s}/step_{k}'];lines.append(f'|{s}|{k}|{p["mean"]:.6f}|{p["without_top2"]:.6f}|{p["urgent"]:.6f}|{p["economy"]:.6f}|{p["capture_mve"]}/{p["capture_sac"]}|{p["unsafe_mve"]}/{p["unsafe_sac"]}|{o["mean"]:.6f}|')
 for s in (0,1):
  new=sum(rd['paired'][f's{s}/step_{k}']['mean'] for k in a.MARKS)/4;before=sum(old['paired'][f's{s}/step_{k}']['mean'] for k in a.MARKS)/4
  lines+=['',f'种子{s}四点均值：修复前{before:.6f}，本轮{new:.6f}；历史对照本轮重述，不能由此单独推断标签修复的因果贡献。']
 lines+=['','完整训练曲线和四点评估见 `图/CURVES.png` / PDF；原始TensorBoard事件保留本机并列SHA256，训练逐回合曲线无补点、无插值。','', '## 机制、分档与墙钟','','```json',json.dumps({'training':rd.get('training'),'urgent_bins_and_both_success':{k:{'urgent_bins':p.get('urgent_bins'),'both_success':p.get('both_success')} for k,p in rd['paired'].items()}},ensure_ascii=False,indent=2),'```','', '## 动作价值排序审计','', '下表以原生D_RANK逐项读数为准。准确率中并列计0.5；关键对按捕获翻转或真值回报差≥50定义。top1分母为真值可区分决策。','', '|检查点|评分|关键对数|准确率|Kendall tau|top1|','|---|---|---:|---:|---:|---:|']
 for label,r in interp['readings'].items():
  for scorer in ('critic','label1','label2','diff'):
   v=r[scorer];fmt=lambda x:'MISSING' if x is None else f'{x:.6f}'
   lines.append(f'|{label}|{scorer}|{v["decisive_pairs"]}|{fmt(v["accuracy"])}|{fmt(v["kendall_tau_mean"])}|{fmt(v["top1_on_separating"])}|')
 lines+=['','审计解释和误差（不改变性能判定，不自动安排下一轮）：','','```json',json.dumps(interp,ensure_ascii=False,indent=2),'```','', '## 证据与交付边界','','本轮原始评估、审计JSON、训练manifest/finished/episodes/model_stats/heartbeat、启动/日志验收、模型SHA256、原生读出和判定一并提供；SHA256SUMS验证本轮文件。历史基线不复制整套目录，仅保留固定引用、哈希及本轮复用所必需的SAC评估。模型ZIP和TensorBoard原始事件留本机。所有状态、日志、脚本在本轮过程目录；OS_ANCESTRY证明任务不依赖Codex进程。','',f'过程目录：`{R}`。科学交付路径：`evidence/T2_6W1_CUTOFF/`；collab交付：`docs/collaboration/lower/handoffs/t2_6w1_cutoff_20261029/`。训练/评估/审计均已结束；下一版方案及执行书由上层审读后决定，本轮不擅改机制。']
 return '\n'.join(lines)+'\n'
def science_commit(assets):
 try:
  a.git('fetch','origin',a.SCIENCE);parent=a.git('rev-parse','origin/'+a.SCIENCE)
  assert a.git('merge-base',a.AUDIT,parent)==a.AUDIT,'remote science history diverged from audit source'
  index=R/'记录/science_publication.index';assert not index.exists();env=os.environ.copy();env['GIT_INDEX_FILE']=str(index)
  a.git('read-tree',parent,env=env)
  for p in sorted(assets.rglob('*')):
   if not p.is_file():continue
   rel='evidence/T2_6W1_CUTOFF/'+p.relative_to(assets).as_posix();blob=a.git('hash-object','-w',p)
   a.git('update-index','--add','--cacheinfo',f'100644,{blob},{rel}',env=env)
  tree=a.git('write-tree',env=env)
  commit=a.git('commit-tree',tree,'-p',parent,'-m','evidence: T2.6W1 cutoff training, registered verdict and rank audit')
  a.git('push','origin',commit+':refs/heads/'+a.SCIENCE)
  assert a.git('ls-remote','origin','refs/heads/'+a.SCIENCE).split()[0]==commit
  out={'status':'pushed_verified','commit':commit,'parent':parent,'jst':a.now(),'science_checkout_unchanged':a.git('rev-parse','HEAD')}
 except Exception as e:out={'status':'publication_pending','jst':a.now(),'error':str(e)}
 a.dump(R/'记录/SCIENCE_PUSH_RECEIPT.json',out);return out
def final():
 a.frozen(a.AUDIT);verify_logs();interp=targets();figs=plots()
 assets=R/'发布快照/final';assets.mkdir(parents=True,exist_ok=False)
 files=[R/n for n in ('C_READOUT.json','DECISION.json','D_RANK.json','MECHANISM_INTERPRETATION.json','MECHANISM1000.json','pytest.log','D_pytest.log')]
 files+=list((R/'C_eval').rglob('*.json'))+list((R/'D_rank').rglob('*.json'))+[R/'D_rank/run.log']+figs+list((R/'输入').glob('*'))+list((R/'脚本').glob('*.py'))
 for name in ('PREFLIGHT_PASS.json','PYTEST_VERIFIED.json','LOGGING_VERIFIED.json','ADAPTER_SHA256.json','SAC_EVAL_REUSE_SHA256.json','AUDIT_CHECKPOINT_SHA256.json','OS_ANCESTRY.json','FINAL_EVENT_VERIFICATION.json','PERFORMANCE_FROZEN.json','EARLY_PUSH_RECEIPT.json','RESOURCE_POLICY.json','RESOURCE_CHANGES.jsonl','FIRST_LIVE_GROWTH_VERIFIED.json','TORCH_RESOURCE_VERIFIED.json','THROUGHPUT_ESTIMATE.json','NO_HOT_OBSERVER_CHANGE_VERIFIED.json'):
  p=R/'记录'/name
  if p.exists():files.append(p)
 for s in (0,1):
  root=R/f'B/TRAIN_e_mve_s{s}'
  files += [root/n for n in ('manifest.json','finished.json','episodes.json','model_stats.json','heartbeat.json')]
  files += [root/'live'/n for n in ('progress.json','progress.jsonl','episodes.jsonl','ADAPTER_MANIFEST.json')]
 for p in (R/'记录').glob('*EXIT.json'):files.append(p)
 for src in files:
  dest=assets/src.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);assert a.sha(src)==a.sha(dest)
 provenance={'jst':a.now(),'train_eval_SHA':a.SHA,'audit_SHA':a.AUDIT,'order_collab':'185765b','baseline_science_evidence':'605f3e49ef358418c2a7ffc35a2ce013986c50ea','baseline_collab_delivery':'4ad969ae4ff67ff906f2b4e6e1ef29d91a946304','baseline_readout':{'path':str(a.W/'C_READOUT.json'),'SHA256':a.sha(a.W/'C_READOUT.json')},'reference':{'path':str(a.W/'A_ref'),'cases':336,'file_hashes':{p.name:a.sha(p) for p in (a.W/'A_ref').glob('*.json')}},'complete_original_results':768,'training_telemetry':'raw events retained locally; per-file hashes in FINAL_EVENT_VERIFICATION','no_new_training_after_this_order':True}
 a.dump(assets/'PROVENANCE.json',provenance)
 text=report(interp);(assets/'RESULT_FINAL.md').write_text(text,encoding='utf-8')
 sums='\n'.join(a.sha(p)+'  '+p.relative_to(assets).as_posix() for p in sorted(assets.rglob('*')) if p.is_file())+'\n';(assets/'SHA256SUMS').write_text(sums,encoding='utf-8')
 sci=science_commit(assets)
 latest=a.SPACE/'上层交付/最新';history=a.SPACE/'上层交付/历史'/('T26W1_PREVIOUS_DELIVERY_'+time.strftime('%Y%m%d_%H%M%S'))
 assert latest.resolve().is_relative_to(a.SPACE.resolve()) and history.resolve().is_relative_to(a.SPACE.resolve()) and not history.exists()
 if latest.exists() and any(latest.iterdir()):
  before={str(p.relative_to(latest)):a.sha(p) for p in latest.rglob('*') if p.is_file()};shutil.move(str(latest),str(history));assert before=={str(p.relative_to(history)):a.sha(p) for p in history.rglob('*') if p.is_file()}
 latest.mkdir(parents=True,exist_ok=True)
 shutil.copytree(assets,latest,dirs_exist_ok=True);shutil.copy2(R/'记录/SCIENCE_PUSH_RECEIPT.json',latest/'SCIENCE_PUSH_RECEIPT.json')
 (latest/'README.md').write_text('# T2.6W1 最终交付\n\n状态：训练、评估、机制审计均已结束。报告见 RESULT_FINAL.md；原生判定见 DECISION.json；本轮原始JSON和SHA256SUMS随附。科学推送状态见SCIENCE_PUSH_RECEIPT.json。\n',encoding='utf-8')
 HANDOFF.mkdir(parents=True,exist_ok=True)
 for name in ('RESULT_FINAL.md','C_READOUT.json','DECISION.json','D_RANK.json','MECHANISM_INTERPRETATION.json','PROVENANCE.json','SHA256SUMS'):shutil.copy2(assets/name,HANDOFF/name)
 shutil.copy2(R/'记录/SCIENCE_PUSH_RECEIPT.json',HANDOFF/'SCIENCE_PUSH_RECEIPT.json')
 a.current('complete_delivered');cur=a.read(a.CURRENT);cur.update(evidence_commit=sci.get('commit'),training_active=False,evaluation_active=False,mechanism_audit_active=False,result={'verdict':a.read(R/'DECISION.json')['verdict'],'evaluation_cases':768,'audit_runs':4,'performance_SHA':a.SHA,'audit_SHA':a.AUDIT},publication_status=sci['status']);a.dump(a.CURRENT,cur)
 receipt=collab_commit(list(HANDOFF.iterdir())+[a.CURRENT],'evidence: final T2.6W1 cutoff verdict and trained-checkpoint ranking audit','FINAL_PUSH_RECEIPT.json')
 shutil.copy2(R/'记录/FINAL_PUSH_RECEIPT.json',latest/'COLLAB_PUSH_RECEIPT.json')
 # Final transport receipts are external to the content hash manifest to avoid self-reference.
 a.dump(R/'DELIVERY_RECEIPT.json',{'jst':a.now(),'local_delivery':str(latest),'report_SHA256':a.sha(latest/'RESULT_FINAL.md'),'science':sci,'collab':receipt,'raw_files':len(list(assets.rglob('*'))),'all_science_finished':True})
 # Verify both the published copy and the complete content manifest, then stop.
 for line in sums.splitlines():
  h,rel=line.split('  ',1);assert a.sha(latest/rel)==h
