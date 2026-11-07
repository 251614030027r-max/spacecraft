"""Read-only T2.5 validation evidence; verified temporary-index ordinary Git transport."""
from pathlib import Path
import importlib.util,inspect,json,math,statistics,os,argparse,time,traceback
R=Path(__file__).resolve().parents[1];OLD=R.parent/'T2_3_TEST_20261025'
spec=importlib.util.spec_from_file_location('transport',OLD/'脚本/publish_t23.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.R=R;p.SHA='34f877e933dc78740674fc82ee0e36f918c541f4'
def build(stage,assets):
 assert p.git(p.REPO,'rev-parse','HEAD')==p.SHA
 assert p.read(R/'记录/PYTEST_VERIFIED.json')['status']=='passed'
 sources=[R/'T25_STOP_MECHANISM_READOUT.json',R/'T25_STOP_RECORD.json',R/'记录/BASELINE_PROVENANCE.json',R/'记录/PYTEST_VERIFIED.json',R/'记录/AUDIT_REUSE.json',R/'pytest.log',R/'输入/T2_5_1_VALIDATE_02c697d.md',R/'输入/USER_ORDER.md',R/'记录/LOGGING_VERIFIED.json',R/'记录/EXTERNAL_ADAPTER_SHA256.json',R/'记录/SAC_REUSE.json',R/'记录/OLD_ASSETS_SHA256.json']
 for name,h in p.read(R/'记录/EXTERNAL_ADAPTER_SHA256.json').items():assert p.sha(R/name)==h
 for name in ('logging_adapter.py','train_with_logs.py','verify_logging.py','run_t251.py','prepare_t251.py','publish_t251.py'):sources.append(R/'脚本'/name)
 state=p.read(R/'RUN_STATE.json');p.dump(assets/'RUN_STATUS_SNAPSHOT.json',state)
 p.dump(assets/'PROVENANCE.json',{'science_SHA':p.SHA,'order_collab':'02c697d','mode':'validation only; two EACC seeds4000/3workers; SAChistorical192; no branch','path_corrections':'T23actual20261025;T24actual20261026','logging':'User-authorized external observer; actual native update/RNG and event growth smoke; no science source changes','optional_T24_curve':'Deferred: requires e13; never switch sole science checkout during34f877e training','early_mechanism':'Provisional actual native probes after1000,source TensorBoard; final nativev251 reads full model_stats'})
 for name in ('A1_CAL.json','A1_LABEL_AUDIT.json'):
  if (R/name).exists():sources.append(R/name)
 sources+=sorted((R/'A1_audit').glob('audit_*.json'))
 report=['# T2.5.1验证：'+stage,'',f'生成{p.now()}；科学`{p.SHA}`；执行书collab`02c697d`；37tests入场门本机通过（实际数见PYTEST_VERIFIED）。双EACC(e_mve)种子0/1各4000、各3异步worker；SAC复用T2.3及T2.4补评估共192，不重训。只验证并交付，不运行任何后续分支。','滚动30秒准确步数/更新/回合，真实与训练回报、成功率、critic/min-Q、反馈探针；外部观察器不新增科学forward/backward/RNG，入场验收见LOGGING_VERIFIED。独立隐藏Windows后台、全CPU/BLAS1。Q1/Q2分别统计冻结源未提供，标MISSING，不捏造。']
 if stage=='mechanism1000':
  d=p.read(R/'MECHANISM1000.json');sources.append(R/'MECHANISM1000.json')
  report+=['','## 1000步机制早报（非最终判定）','统计窗口严格在1000之后，原生探针经事件文件读取，各字段列明实际样本数和首末步数，不以目录存在或合成数据代替。w2_mean低于0.5、pi_gap_sigma_median高于0.25、grad_cos不大于0或字段缺失即列警报；按书只报告、不停止训练。','```json',json.dumps(d,ensure_ascii=False,indent=2),'```']
 else:
  d=p.read(R/'B_READOUT.json');decision=p.read(R/'DECISION.json');fs=sorted((R/'B_eval').rglob('*.json'))
  assert len(fs)==432 and not any(f.name.startswith('ERROR_') for f in fs) and d['all_complete'] and decision['verdict']!='incomplete'
  sources+=fs+[R/'B_READOUT.json',R/'DECISION.json'];audits=sorted((R/'A1_audit').glob('audit_*.json'));assert len(audits)==6
  assert p.read(R/'A1_LABEL_AUDIT.json')['errors']==0
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  events={};modelhash=[]
  for seed in (0,1):
   tag=R/f'TRAIN_e_mve_s{seed}';assert p.read(tag/'finished.json')['steps']==4000
   for name in ('manifest.json','finished.json','episodes.json','model_stats.json','heartbeat.json'):sources.append(tag/name)
   for name in ('progress.json','progress.jsonl','episodes.jsonl','ADAPTER_MANIFEST.json'):sources.append(tag/'live'/name)
   ea=EventAccumulator(str(tag/'live/tensorboard'),size_guidance={'scalars':0});ea.Reload();tags=ea.Tags()['scalars'];summary={}
   for t in tags:
    vals=ea.Scalars(t);assert vals and all(math.isfinite(v.value) for v in vals) and all(b.step>=a.step for a,b in zip(vals,vals[1:]));summary[t]={'n':len(vals),'first_step':vals[0].step,'last_step':vals[-1].step,'last_value':vals[-1].value}
   for t in ('progress/steps','rollout/return_true','rollout/return_train','critic/real_td','critic/q_pi_mean','model/two_stage_share','model/decisive_share','model/c_mean','model/stage2_rows','model/w2_mean','model/pi_gap_sigma_median'):assert t in tags
   assert summary['progress/steps']['last_value']==4000;events[f'e_mve_s{seed}']=summary
   for f in sorted(tag.rglob('*.zip')):modelhash.append({'path':str(f),'SHA256':p.sha(f),'binary_not_published':True})
  p.dump(assets/'TENSORBOARD_EVENT_SUMMARY.json',events);p.dump(assets/'LOCAL_MODEL_SHA256.json',modelhash)
  report+=['','## 原生v251判定','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','pass仅本轮开发验证，不是论文正式证据或收敛定理。SAC种子1只到3k，MVE种子1的4k无匹配SAC，不捏造对照。','','|臂|步数|任务|n|回报|捕获|unsafe|','|---|---:|---|---:|---:|---:|---:|']
  for arm,per in d['arms'].items():
   for k,st in sorted(per.items(),key=lambda x:int(x[0])):
    for group,v in (('all',st),('urgent',st['urgent']),('economy',st['economy'])):report.append(f"|{arm}|{k}|{group}|{v['n']}|{v['mean_return']}|{v['capture']}|{v['unsafe']}|")
  report+=['','## 同开局配对（EACC−SAC）','```json',json.dumps(d['paired'],ensure_ascii=False,indent=2),'```','机制最终均值由原生v251按1000步之后model_stats计算，M/E1/E2/S逐项数值与verdict原件完整提供。two_stage_share、decisive_share、w2_mean、pi_gap_sigma_median、grad_cos正常只能说明机制工作；c_mean只报告不作门槛；是否有部署收益由匹配评估判定。六开局复用T2.5冻结84ee9c1审计包含阶段2标签、实际进入口与未进入口状态，不把缺匹配数据填0。','训练真实回报带探索，不能替代确定性部署曲线；两种子单独列报，不挑种子/检查点。']
 p.dump(assets/'T25_STOP_COMPARISON.json',p.read(R/'T25_STOP_MECHANISM_READOUT.json'))
 report+=['','## T2.5停止前机制对照（未完成、无部署判定）','```json',json.dumps(p.read(R/'T25_STOP_MECHANISM_READOUT.json'),ensure_ascii=False,indent=2),'```','旧T2.5已按用户停止，最后落盘2084/1888步，原目录保全；上表旧c值仅机制对照，不是最终性能失败结论。本轮只新learner训练；SAC与6例标签审计按书复用，不跑新的标签审计。']
 for f in sources:
  assert f.exists(),str(f);p.copy(f,assets)
 report+=['','## 历史对照、异常与墙钟','T2.4固定science632b633：MVE种子0的4k回报53.792454、种子1为55.238621，均捕获23/24、unsafe0；种子0相对SAC54.476604差−0.684149。T2.5.1原生v251按执行书阈值53.79，事后不改。T2.3对照来源、逐文件SHA和模型SHA随附，旧SAC恢复事件字段若缺失应标MISSING，不当0。','两个训练finished原件包含实际wall_s和worker_order，滚动吞吐与ETA区分；并行墙钟不相加。训练真实技术崩溃同配置从零最多一次且保全旧attempt，不拼Replay/RNG/优化器；纯发布失败只补报告，不重跑科学。可选T2.4早期曲线要求另一SHA，为不影响本轮先暂缓，不混版本执行。','模型ZIP/Replay/TB二进制本机保留，Git发布432必要原始评估、6审计、训练JSON与外部滚动日志/事件解析和验收。没有自动长训或后续分支，等待上层复核与用户批准。']
 text='\n'.join(report)+'\n';(assets/'RESULT.md').write_text(text,encoding='utf-8');(assets/('RESULT_FINAL.md' if stage=='final' else 'RESULT_MECHANISM1000.md')).write_text(text,encoding='utf-8');(assets/'README.md').write_text('T2.5.1验证/'+stage+'；入口RESULT.md；无后续分支。\n',encoding='utf-8')
p.build=build
source=inspect.getsource(p.main).replace("assert stage in ('A','B','P','N','K','L');","assert stage in ('final','mechanism1000');",1)
source=source.replace("prefix='evidence/T2_3_TEST/'+stage if key=='science' else 'lower/handoffs/t2_3_test_20261023'","prefix=('evidence/T2_5_1_VALIDATE' if stage=='final' else 'evidence/T2_5_1_VALIDATE/mechanism1000') if key=='science' else 'lower/handoffs/t2_5_1_validate_20261026'")
exec(compile(source,'verified_t25_transport','exec'),p.__dict__)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['final','mechanism1000'],required=True);arg=ap.parse_args();lock=R/'记录/PUBLICATION.lock';owned=False
 try:
  for _ in range(90):
   try:
    with lock.open('x') as f:f.write(str(os.getpid()))
    owned=True;break
   except FileExistsError:time.sleep(2)
  assert owned;p.main(arg.stage)
 except Exception:p.dump(R/f'记录/PUBLICATION_{arg.stage}_ERROR.json',{'jst':p.now(),'traceback':traceback.format_exc()});raise
 finally:
  if owned:lock.unlink()
