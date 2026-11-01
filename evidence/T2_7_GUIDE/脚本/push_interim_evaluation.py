"""Immediate evidence delivery from existing results; all evaluation processes continue unchanged."""
import os,json,shutil,subprocess,time
from pathlib import Path
import psutil
import runtime as a
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
R=a.R;tag=time.strftime('%Y%m%d_%H%M%S');OUT=R/'发布快照'/('evaluation_interim_'+tag);OUT.mkdir(parents=True,exist_ok=False)
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert a.sha(src)==a.sha(dst)
def cmd(args,log):
 with (OUT/log).open('wb') as f:
  p=subprocess.run([a.PY,'-B',*map(str,args)],cwd=a.REPO,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
 assert p.returncode==0,log
def main():
 a.frozen();state=a.read(R/'RUN_STATE.json');a.dump(OUT/'RUN_STATE_SNAPSHOT.json',state);a.dump(OUT/'EVALUATION_POOL_SNAPSHOT.json',a.read(R/'记录/EARLY_EVAL_POOL_STATE.json'))
 provenance={};new=0;inflight=[]
 for p in sorted((R/'E_guided').rglob('*.json')):
  raw=p.read_bytes()
  try:x=json.loads(raw)
  except (json.JSONDecodeError,UnicodeDecodeError):inflight.append(str(p));continue
  assert not p.name.startswith('ERROR_')
  rel=p.relative_to(R/'E_guided');d=OUT/('C_primary' if rel.parts[0].startswith('e_mve') else 'SAC_guided_partial')/rel
  d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(raw);provenance[d.relative_to(OUT).as_posix()]={'source':str(p),'SHA256':a.sha(d)};new+=1
 for s in (0,1):
  for k in a.MARKS:
   src=a.W/f'C_eval/e_sac_s{s}/step_{k}';assert len(list(src.glob('*.json')))==48
   for p in src.glob('*.json'):copy(p,OUT/'C_primary'/p.relative_to(a.W/'C_eval'))
 for arm in ('e_mve','e_sac'):
  for s in (0,1):
   src=R/f'B/TRAIN_{arm}_s{s}';assert a.read(src/'finished.json')['steps']==4000
   for name in ('manifest.json','finished.json','episodes.json','model_stats.json','heartbeat.json'):copy(src/name,OUT/f'B/TRAIN_{arm}_s{s}'/name)
   if arm=='e_mve':
    d=R/f'记录/interim_pairing_{tag}/B_mve/TRAIN_{arm}_s{s}';d.mkdir(parents=True)
    for name in ('model_stats.json','finished.json'):copy(src/name,d/name)
 cmd(['-m','experiments.t2_readout','curve','--eval-root',OUT/'C_primary','--train-root',R/f'记录/interim_pairing_{tag}/B_mve','--screen',a.W/'A_ref','--expected','48','--output',OUT/'PRIMARY_READOUT.json'],'primary_readout.log')
 assert a.read(OUT/'PRIMARY_READOUT.json')['all_complete']
 cmd(['-m','experiments.t2_decide','--criteria','t27','--checkpoints','1000,2000,3000,4000','--readout',OUT/'PRIMARY_READOUT.json','--train-root',R/f'记录/interim_pairing_{tag}/B_mve','--output',OUT/'PRIMARY_NATIVE_PREVIEW_NOT_FINAL.json'],'primary_preview.log')
 dec=a.read(OUT/'PRIMARY_NATIVE_PREVIEW_NOT_FINAL.json');rd=a.read(OUT/'PRIMARY_READOUT.json')
 prior={'path':str(a.W1/'C_READOUT.json'),'SHA256':a.sha(a.W1/'C_READOUT.json'),'fixed_science_evidence':'7dac3ddfab51e0f8e2429475b3fbab026eeadf08','paired_auc':{'seed0':-8.212149946246909,'seed1':19.6859326220293}}
 a.dump(OUT/'PROVENANCE.json',{'generated_jst':a.now(),'frozen_SHA':a.SHA,'order_collab':'8708f39','new_complete_result_snapshot':new,'primary_new_EACC':384,'reused_SAC':384,'auxiliary_evaluation':'still running; partial cases cannot be treated as complete checkpoints','skipped_inflight_files':inflight,'source_file_hashes':provenance,'previous_EACC_reference':prior,'no_new_simulation_or_model_inference':True,'evaluation_not_stopped':True})
 report=['# T2.7 主比较已齐：即时上层问题判断材料','',f'生成{a.now()}。用户要求已有资产立即推送，评估不停。本包固定快照含{new}例本轮已完整写出的评估、384例复用SAC及四条完成训练的完整元数据/回合日志；辅助SAC+引导评估继续。科学冻结2208a51不变。','', '## 已经可以明确的问题','',f'**主比较原生判定预览：{dec["verdict"]}。** PRIMARY_NATIVE_PREVIEW_NOT_FINAL由未修改的t27原生工具对完整主比较输入生成；不是全轮交付完成声明，根目录正式DECISION和四配对最终读出仍待原队列完成。主比较已齐，不必等待辅助评估才看到主比较问题。','', '```json',json.dumps(dec,ensure_ascii=False,indent=2),'```','', '|种子|四点平均EACC+引导−SAC|上轮EACC−SAC|4000步差|四点unsafe EACC+引导/SAC|紧急捕获累计差|','|---|---:|---:|---:|---|---:|']
 for s in (0,1):
  ps=[rd['paired'][f's{s}/step_{k}'] for k in a.MARKS];mean=sum(p['mean'] for p in ps)/4;ug=sum(p['urgent_capture_mve']-p['urgent_capture_sac'] for p in ps);um=sum(p['unsafe_mve'] for p in ps);us=sum(p['unsafe_sac'] for p in ps)
  report.append(f'|{s}|{mean:.6f}|{prior["paired_auc"][f"seed{s}"]:.6f}|{ps[-1]["mean"]:.6f}|{um}/{us}|{ug}|')
 report+=['','效率未过：seed0四点差仍为负，两种子平均约+2.28也低于+5。安全未过：seed0四检查点累计4次unsafe对SAC的0，超过允许的+1；这些是评估计数，可能包含同一开局在不同检查点的重复违规，不冒称4个独立场景。4000步无unsafe不能抹掉此前记录。种子0略改善、种子1相对上轮回落，尚无稳定整体提升。','', '## 问题线索与上层审查建议','', '先核查seed0的unsafe原始开局、轨迹终止字段和检查点分布，并对照引导撤除前后的策略表现；原始评估JSON随附足够定位开局，本包没有新增回放或真值分支。不得据此直接把安全退化归因于某个机制。','', 'SAC+引导辅助对照未齐，暂不能区分引导本身与标签增量。下一步只需完成既定辅助评估并看C_guide_only/C_labels/C_vs_eacc；不等于批准扩深、加门、改奖励或新训练。所有四条训练均4000完成、无新增训练。本轮即使4000点局部转好，仍严格保留四点与安全门槛，不挑点。','', '## 附件与缺项','', 'C_primary提供完整主比较768个逐例JSON；SAC_guided_partial仅提供已完成的辅助结果，未完成标MISSING。B包含四条训练manifest、finished、episodes、model_stats、heartbeat与完整外部回合/滚动日志；模型ZIP不上传。PRIMARY_READOUT含逐开局配对、紧急/经济、分档、共同成功燃料时间。正式四配对最终读出与本轮最终交付：MISSING、仍在评估。SHA256SUMS固定本包，不能用于推断后续活动目录完整性。','', '本即时交付不停止评估、不改训练、不改变预登记门槛。正式最终报告将在全部辅助评估及四种配对完整后另外交付。']
 (OUT/'RESULT_INTERIM_EVAL.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 (OUT/'SHA256SUMS').write_text('\n'.join(a.sha(p)+'  '+p.relative_to(OUT).as_posix() for p in sorted(OUT.rglob('*')) if p.is_file())+'\n',encoding='utf-8')
 # Separate index and evidence path: never switch or alter the live frozen checkout or final publication index.
 a.git('fetch','origin',a.SCIENCE);parent=a.git('rev-parse','origin/'+a.SCIENCE);assert a.git('merge-base',a.SHA,parent)==a.SHA
 env=os.environ.copy();env['GIT_INDEX_FILE']=str(R/f'记录/interim_publication_{tag}.index');a.git('read-tree',parent,env=env)
 for p in sorted(OUT.rglob('*')):
  if p.is_file():
   blob=a.git('hash-object','-w',p);rel='evidence/T2_7_GUIDE/interim_20261101/'+p.relative_to(OUT).as_posix();a.git('update-index','--add','--cacheinfo',f'100644,{blob},{rel}',env=env)
 tree=a.git('write-tree',env=env);science=a.git('commit-tree',tree,'-p',parent,'-m','evidence: T2.7 complete primary comparison while auxiliary evaluation continues');a.git('push','origin',science+':refs/heads/'+a.SCIENCE);assert a.git('ls-remote','origin','refs/heads/'+a.SCIENCE).split()[0]==science
 # Local common delivery entry, preserving the prior midterm unchanged.
 latest=a.SPACE/'上层交付/最新';history=a.SPACE/'上层交付/历史'/('T27_midterm_before_eval_'+tag)
 assert latest.resolve().is_relative_to(a.SPACE.resolve()) and history.resolve().is_relative_to(a.SPACE.resolve()) and not history.exists()
 h={p.relative_to(latest).as_posix():a.sha(p) for p in latest.rglob('*') if p.is_file()};shutil.move(str(latest),str(history));assert h=={p.relative_to(history).as_posix():a.sha(p) for p in history.rglob('*') if p.is_file()}
 shutil.copytree(OUT,latest);(latest/'README.md').write_text('# T2.7 即时评估交付｜辅助评估不停\n\n先读RESULT_INTERIM_EVAL.md。完整主比较及原生判定预览已齐；辅助SAC+引导仍评估。非全轮最终交付，正式最终报告仍待完整四配对。前期中期报告保全在历史目录。\n',encoding='utf-8')
 dest=a.COLLAB/'docs/collaboration/lower/handoffs/t2_7_guide_20261031/interim_eval_20261101';dest.mkdir(parents=True,exist_ok=False)
 for n in ('RESULT_INTERIM_EVAL.md','PRIMARY_NATIVE_PREVIEW_NOT_FINAL.json','PRIMARY_READOUT.json','PROVENANCE.json','SHA256SUMS'):copy(OUT/n,dest/n)
 a.dump(dest/'SCIENCE_POINTER.json',{'commit':science,'path':'evidence/T2_7_GUIDE/interim_20261101/','raw_cases_in_science_branch':True})
 a.git('fetch','origin','collab/spacecraft',repo=a.COLLAB);a.git('merge','--ff-only','origin/collab/spacecraft',repo=a.COLLAB);assert not a.git('diff','--cached','--name-only',repo=a.COLLAB)
 cur=a.read(a.CURRENT);cur.update(updated_at=a.now(),local_report=str(latest/'RESULT_INTERIM_EVAL.md'),current_delivery_report=str((dest/'RESULT_INTERIM_EVAL.md').relative_to(a.COLLAB)).replace('\\','/'),current_delivery_phase='primary_complete_auxiliary_evaluation_running',interim_science_evidence_commit=science,interim_native_preview=dec['verdict']);a.dump(a.CURRENT,cur)
 a.git('add','--',str(dest.relative_to(a.COLLAB)),'docs/collaboration/CURRENT.json',repo=a.COLLAB);a.git('commit','-m','delivery: T2.7 primary safety and efficiency issues; auxiliary evaluation continues',repo=a.COLLAB);collab=a.git('rev-parse','HEAD',repo=a.COLLAB);a.git('push','origin','HEAD:refs/heads/collab/spacecraft',repo=a.COLLAB);assert a.git('ls-remote','origin','refs/heads/collab/spacecraft',repo=a.COLLAB).split()[0]==collab
 a.frozen();receipt={'jst':a.now(),'science_commit':science,'collab_commit':collab,'native_primary_preview':dec['verdict'],'new_completed_cases_in_snapshot':new,'evaluation_continues':True,'local_report':str(latest/'RESULT_INTERIM_EVAL.md'),'report_SHA256':a.sha(OUT/'RESULT_INTERIM_EVAL.md')};a.dump(R/'记录/INTERIM_EVAL_PUSH_RECEIPT.json',receipt);copy(R/'记录/INTERIM_EVAL_PUSH_RECEIPT.json',latest/'PUSH_RECEIPT.json');print(json.dumps(receipt,ensure_ascii=True))
if __name__=='__main__':main()
