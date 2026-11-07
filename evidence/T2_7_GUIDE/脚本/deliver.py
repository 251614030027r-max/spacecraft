from pathlib import Path
import json,shutil,math,time,statistics
import runtime as a
import transport as t
R=a.R;H=t.HANDOFF;ARMS=('e_mve','e_sac');PAIRS=('C_eval','C_guide_only','C_labels','C_vs_eacc')
def early_snapshots():
 out={}
 for arm in ARMS:
  for s in (0,1):
   d=R/f'记录/early1000/{arm}_s{s}';hb=a.read(d/'heartbeat.json');eps=[x for x in a.read(d/'episodes.json') if x['steps']<=1000];ms=[x for x in a.read(d/'model_stats.json') if x['steps']<=1000 and x.get('model_rows')]
   assert hb['steps']==1000
   v={'steps':1000,'guided':hb['replay'].get('guided',0),'guided_share':hb['replay'].get('guided',0)/1000,'episodes':len(eps),'guided_per_episode_mean':sum(x['guided'] for x in eps)/len(eps),'wall_seconds_per_decision':hb['wall_s']/1000,'first_model_step':ms[0]['steps'] if ms else None}
   for u in (True,False):
    rows=[x for x in eps if x['mission']['urgent']==u];v['urgent' if u else 'economy']={'n':len(rows),'capture':sum(x['event']=='success' for x in rows),'capture_rate':sum(x['event']=='success' for x in rows)/len(rows) if rows else None}
   for k in ('two_stage_share','decisive_share','w2_mean','pi_gap_sigma_median','grad_cos','stage2_cutoff_share','entry_bias_now'):
    vals=[x[k] for x in ms if x.get(k) is not None];v[k]={'n':len(vals),'mean':statistics.mean(vals) if vals else None}
   out[f'{arm}_s{s}']=v
 a.dump(R/'EARLY1000.json',{'jst':a.now(),'window':'exact preserved1000 snapshots; no later probes','not_a_stop_gate':True,'readings':out});H.mkdir(parents=True,exist_ok=True)
 shutil.copy2(R/'EARLY1000.json',H/'EARLY1000.json');(H/'RESULT_1000.md').write_text('# T2.7 四条引导训练千步早报\n\n冻结2208a51，四条训练继续，不按早报停止或调参。\n\n```json\n'+json.dumps(out,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
 t.collab_commit([H/'EARLY1000.json',H/'RESULT_1000.md',a.CURRENT],'evidence: T2.7 four guided runs at1000; training continues','EARLY_PUSH_RECEIPT.json')
def checks_and_supplement():
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 telemetry={};models=[];history={};medians={}
 for arm in ARMS:
  for s in (0,1):
   d=R/f'B/TRAIN_{arm}_s{s}';fin=a.read(d/'finished.json');m=a.read(d/'manifest.json');assert fin['steps']==4000 and m['guide'] and m['git_head']==a.SHA
   assert a.read(d/'live/TORCH_THREADS.json')['intra_op']==1 and not (d/'live/LOGGER_ERROR.json').exists()
   native=a.read(d/'episodes.json');logged=[json.loads(x) for x in (d/'live/episodes.jsonl').read_text(encoding='utf-8').splitlines()];assert len(native)==len(logged)>0
   assert [(x['steps'],x['worker'],x['event'],x['guided']) for x in native]==[(x['steps'],x['worker'],x['event'],x['guided']) for x in logged]
   e=EventAccumulator(str(d/'live/tensorboard'),size_guidance={'scalars':0});e.Reload();tags=e.Tags()['scalars'];assert all(k in tags for k in ('progress/steps','rollout/return_true','rollout/guided','critic/real_td','critic/q_pi_mean'))
   for k in tags:
    vals=e.Scalars(k);assert all(math.isfinite(x.value) for x in vals) and all(y.step>=x.step for x,y in zip(vals,vals[1:]))
   assert e.Scalars('progress/steps')[-1].value==4000
   telemetry[f'{arm}_s{s}']={'episodes':len(native),'tags':{k:len(e.Scalars(k)) for k in tags},'raw_event_SHA256':[{'path':str(p),'SHA256':a.sha(p)} for p in (d/'live/tensorboard').glob('events.out*')],'bitwise_observer_gate':a.read(R/'记录/LOGGING_VERIFIED.json')['native_updates_bitwise_equal']}
   history[f'{arm}_s{s}']={}
   for lo,hi in ((0,1000),(1000,2000),(2000,3000),(3000,4000)):
    rows=[x for x in logged if lo<x['steps']<=hi];v={'episode_count':len(rows),'mean_true_return':statistics.mean(x['return_true'] for x in rows) if rows else None,'mean_guided_per_completed_episode':statistics.mean(x['guided'] for x in rows) if rows else None,'warning':'Episode ending in this interval may span earlier decisions; guided totals can carry over the2000 boundary.'}
    for u in (True,False):
     z=[x for x in rows if x['mission']['urgent']==u];v['urgent' if u else 'economy']={'n':len(z),'captures':sum(x['success'] for x in z),'capture_rate':statistics.mean(x['success'] for x in z) if z else None}
    history[f'{arm}_s{s}'][f'{lo}-{hi}']=v
   for p in (d/'checkpoints').glob('*.zip'):models.append({'path':str(p),'SHA256':a.sha(p)})
 for pair in PAIRS:
  medians[pair]={}
  for s in (0,1):
   for k in a.MARKS:
    load=lambda arm:{x['opening']:x for x in (a.read(p) for p in (R/f'{pair}/{arm}_s{s}/step_{k}').glob('*.json'))}
    mv,sc=load('e_mve'),load('e_sac');assert len(mv)==len(sc)==48
    diffs=[mv[o]['return']-sc[o]['return'] for o in sorted(mv)];medians[pair][f's{s}/step_{k}']={'n':48,'median_paired_difference':statistics.median(diffs),'per_opening':{str(o):mv[o]['return']-sc[o]['return'] for o in sorted(mv)}}
 a.dump(R/'记录/FINAL_EVENT_VERIFICATION.json',telemetry);a.dump(R/'LOCAL_MODEL_SHA256.json',models);a.dump(R/'TRAINING_HISTORY.json',history);a.dump(R/'PAIR_SUPPLEMENT.json',medians)
 return history,medians
def figures():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 out=R/'图';out.mkdir(exist_ok=True);fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
 for s in (0,1):
  for arm,root,label in (('e_mve',R,'EACC+guide'),('e_sac',R,'SAC+guide'),('e_mve',a.W1,'EACC'),('e_sac',a.W,'SAC')):
   e=EventAccumulator(str(root/f'B/TRAIN_{arm}_s{s}/live/tensorboard'),size_guidance={'scalars':0});e.Reload();vals=e.Scalars('rollout/return_true');y=[x.value for x in vals];x=[v.step for v in vals];mean=[statistics.mean(y[max(0,i-49):i+1]) for i in range(len(y))];axes[s,0].plot(x,mean,label=label)
  axes[s,0].axvline(2000,color='gray',ls='--');axes[s,0].set(title=f'Seed{s}:50-episode true return',xlabel='Decision steps',ylabel='True return');axes[s,0].legend()
  for pair in PAIRS:
   r=a.read(R/(pair+'_READOUT.json'));axes[s,1].plot(a.MARKS,[r['paired'][f's{s}/step_{k}']['mean'] for k in a.MARKS],'o-',label=pair)
  axes[s,1].axhline(0,color='gray');axes[s,1].set(title=f'Seed{s}: four paired contrasts',xlabel='Decision steps',ylabel='Paired return');axes[s,1].legend()
  r=a.read(R/'C_eval_READOUT.json')
  for field in ('urgent','economy'):
   axes[s,2].plot(a.MARKS,[r['paired'][f's{s}/step_{k}'][field] for k in a.MARKS],'o-',label=field)
  axes[s,2].axhline(0,color='gray');axes[s,2].set(title=f'Seed{s}:primary contrast',xlabel='Decision steps',ylabel='Paired return');axes[s,2].legend()
 for ax in axes.flat:ax.grid(alpha=.2)
 fig.savefig(out/'CURVES.png',dpi=180);fig.savefig(out/'CURVES.pdf');plt.close(fig)
 return list(out.iterdir())
def final():
 a.frozen();history,medians=checks_and_supplement();figs=figures();dec=a.read(R/'DECISION.json');readouts={n:a.read(R/(n+'_READOUT.json')) for n in PAIRS}
 assets=R/'发布快照/final';assets.mkdir(parents=True,exist_ok=False)
 files=[R/(n+'_READOUT.json') for n in PAIRS]+[R/n for n in ('DECISION.json','pytest.log','EARLY1000.json','LOCAL_MODEL_SHA256.json','TRAINING_HISTORY.json','PAIR_SUPPLEMENT.json')]+figs
 files+=list((R/'E_guided').rglob('*.json'))+list((R/'输入').glob('*'))+list((R/'脚本').glob('*.py'))
 for root,arm in ((a.W,'e_sac'),(a.W1,'e_mve')):
  for s in (0,1):
   for k in a.MARKS:
    for p in (root/f'C_eval/{arm}_s{s}/step_{k}').glob('*.json'):
     d=assets/'复用评估'/root.name/p.relative_to(root);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
 for arm in ARMS:
  for s in (0,1):
   root=R/f'B/TRAIN_{arm}_s{s}';files += [root/n for n in ('manifest.json','finished.json','episodes.json','model_stats.json','heartbeat.json')]+[root/'live'/n for n in ('progress.json','progress.jsonl','episodes.jsonl','TORCH_THREADS.json','ADAPTER_MANIFEST.json')]
 for n in ('OS_ANCESTRY.json','PREFLIGHT_PASS.json','LOGGING_VERIFIED.json','REFERENCE_SHA256.json','OBSERVER_SHA256.json','FINAL_EVENT_VERIFICATION.json','EARLY_PUSH_RECEIPT.json'):
  p=R/'记录'/n
  if p.exists():files.append(p)
 for src in files:
  d=assets/src.relative_to(R);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,d);assert a.sha(d)==a.sha(src)
 provenance={'science_SHA':a.SHA,'order_collab':a.ORDER,'new_training_runs':4,'new_evaluations':768,'reused_SAC_evaluations':384,'reference_unguided_EACC_evaluations':384,'SAC_evidence_commit':'605f3e49ef358418c2a7ffc35a2ce013986c50ea','EACC_evidence_commit':'7dac3ddfab51e0f8e2429475b3fbab026eeadf08','references':[{'path':str(root/'C_READOUT.json'),'SHA256':a.sha(root/'C_READOUT.json')} for root in (a.W,a.W1)],'prior_round_CURRENT_backup_SHA256':a.sha(R/'记录/CURRENT_BEFORE.json'),'model_and_raw_event_files':'Local only, SHA256 indexed','no_posthoc_thresholds':True}
 a.dump(assets/'PROVENANCE.json',provenance)
 report=['# T2.7 执行器引导2×2验证最终交付','',f'生成{a.now()}。科学冻结`{a.SHA}`，执行书collab`8708f39`。原生主判定 **{dec["verdict"]}**。四条从零训练、各4000步/2异步worker，guide_p0=0.5、guide_until=2000，环境0.5.0/学习器2.7；全部部署评估关闭引导，仅策略独立决策。','', '## 主判定与证据边界','','```json',json.dumps(dec,ensure_ascii=False,indent=2),'```','', '主判定仅EACC+引导对复用SAC；三个辅助配对仅报告，不另造通过条件。引导收益不能自动归给标签；标签独立增量以C_labels为准。两个种子、开发块结果不等于正式泛化证明。','', '## 四种配对曲线','', '|对照|种子|步数|均值|中位差|剔除最大两例|紧急|经济|捕获第一/第二|unsafe第一/第二|','|---|---|---:|---:|---:|---:|---:|---:|---|---|']
 for pair,r in readouts.items():
  for s in (0,1):
   for k in a.MARKS:
    key=f's{s}/step_{k}';p=r['paired'][key];report.append(f'|{pair}|{s}|{k}|{p["mean"]:.6f}|{medians[pair][key]["median_paired_difference"]:.6f}|{p["without_top2"]:.6f}|{p["urgent"]:.6f}|{p["economy"]:.6f}|{p["capture_mve"]}/{p["capture_sac"]}|{p["unsafe_mve"]}/{p["unsafe_sac"]}|')
 report+=['','C_eval=EACC+引导−SAC；C_guide_only=SAC+引导−SAC；C_labels=EACC+引导−SAC+引导；C_vs_eacc=EACC+引导−EACC。图/CURVES展示完整训练回报和四种配对曲线。原始逐开局结果、分档、双方成功燃料/时间随附，未选择性删样本。','', '## 三个问题的事实读数','', '完整方案跨种子可重复性按DECISION中E1/U/E2/S回答。引导作用与标签增量按如下四点均值各自报告：']
 for pair,r in readouts.items():report.append(f'- {pair}：'+str({s:statistics.mean(r['paired'][f's{s}/step_{k}']['mean'] for k in a.MARKS) for s in (0,1)}))
 report+=['','2000步撤除引导后的自主保持：3000/4000步部署仍没有引导，见完整四点曲线和原生reported.late_*；训练回合历史下列分段仅描述，跨边界完成的回合可能含前段guided动作，不把回合guided数误认为2000步后仍在引导。','', '```json',json.dumps(history,ensure_ascii=False,indent=2),'```','', '## 机制、压力分档、共同成功燃料/时间与墙钟','', '```json',json.dumps({'training':readouts['C_eval'].get('training'),'pressure_and_both_success':{n:{k:{'urgent_bins':v.get('urgent_bins'),'both_success':v.get('both_success')} for k,v in r['paired'].items()} for n,r in readouts.items()}},ensure_ascii=False,indent=2),'```','', '## 对照与交付位置','', 'SAC复用T2.6W，未重训/重评；无引导EACC复用T2.6W1，仅辅助参照。旧轮数值在四种配对读出中重述，原始复用JSON作为本轮判断必需证据随附，其余历史只固定引用及SHA。52项测试、真实guidance-on两臂日志冒烟、TensorBoard实际解析及增长验收、操作系统祖先链、完整训练meta/回合/模型探针均附。实际PyTorch线程数在每条训练live/TORCH_THREADS.json，启动时Normal优先级；11月1日11:50按用户要求将现有训练树提升High（非Realtime），不据此声称已证明加速。四条共享CPU0–11，OMP/MKL/OPENBLAS=1。','',f'完整过程目录`{R}`；科学证据`evidence/T2_7_GUIDE/`。模型ZIP及原始TensorBoard事件留本机，仅记SHA。报告、原始结果与SHA256SUMS足够上层独立判定。下一轮实验或机制修改未授权。']
 (assets/'RESULT_FINAL.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 sums='\n'.join(a.sha(p)+'  '+p.relative_to(assets).as_posix() for p in sorted(assets.rglob('*')) if p.is_file())+'\n';(assets/'SHA256SUMS').write_text(sums,encoding='utf-8')
 sci=t.science_commit(assets)
 latest=a.SPACE/'上层交付/最新';hist=a.SPACE/'上层交付/历史'/('T27_previous_'+time.strftime('%Y%m%d_%H%M%S'))
 assert latest.resolve().is_relative_to(a.SPACE.resolve()) and hist.resolve().is_relative_to(a.SPACE.resolve()) and not hist.exists()
 if latest.exists() and any(latest.iterdir()):
  h={p.relative_to(latest).as_posix():a.sha(p) for p in latest.rglob('*') if p.is_file()};shutil.move(str(latest),str(hist));assert h=={p.relative_to(hist).as_posix():a.sha(p) for p in hist.rglob('*') if p.is_file()}
 latest.mkdir(parents=True,exist_ok=True);shutil.copytree(assets,latest,dirs_exist_ok=True)
 H.mkdir(parents=True,exist_ok=True)
 for n in ('RESULT_FINAL.md','DECISION.json','PROVENANCE.json','SHA256SUMS','TRAINING_HISTORY.json','PAIR_SUPPLEMENT.json')+tuple(x+'_READOUT.json' for x in PAIRS):shutil.copy2(assets/n,H/n)
 a.current('complete_delivered');cur=a.read(a.CURRENT);cur.update(result={'verdict':dec['verdict'],'new_evaluation_cases':768,'reused_reference_cases':768,'training_runs':4},evidence_commit=sci.get('commit'),next_action='Wait for upper review and a new execution order. No additional science authorized.');a.dump(a.CURRENT,cur)
 receipt=t.collab_commit(list(H.iterdir())+[a.CURRENT],'evidence: T2.7 guided2x2 complete verdict and attribution','FINAL_PUSH_RECEIPT.json')
 for n in ('SCIENCE_PUSH_RECEIPT.json','FINAL_PUSH_RECEIPT.json'):shutil.copy2(R/'记录'/n,latest/n)
 (latest/'README.md').write_text('# T2.7 最终交付\n\n报告RESULT_FINAL.md；主判定DECISION.json；四种配对读出、原始评估、完整训练日志及SHA256SUMS随附。推送状态见两份PUSH_RECEIPT。\n',encoding='utf-8')
 for line in sums.splitlines():
  h,rel=line.split('  ',1);assert a.sha(latest/rel)==h
 a.dump(R/'DELIVERY_RECEIPT.json',{'jst':a.now(),'science':sci,'collab':receipt,'local_delivery':str(latest),'report_SHA256':a.sha(latest/'RESULT_FINAL.md'),'all_science_finished':True})

