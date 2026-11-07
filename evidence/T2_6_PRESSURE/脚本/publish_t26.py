"""Read-only T2.6 evidence; ordinary Git with verified temporary-index byte-preserving transport."""
from pathlib import Path
import importlib.util,inspect,json,math,os,argparse,time,traceback
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transport',R.parent/'T2_3_TEST_20261025/脚本/publish_t23.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.R=R;p.SHA='f4d660aa820a2e3b08c8be310aeac7a55d641318'
def build(stage,assets):
 assert p.git(p.REPO,'rev-parse','HEAD')==p.SHA and p.read(R/'记录/PYTEST_VERIFIED.json')['status']=='passed'
 src=[R/'pytest.log',R/'记录/PYTEST_VERIFIED.json',R/'输入/T2_6_PRESSURE_c1a962c.md',R/'输入/USER_ORDER.md',R/'记录/PROBE_SOURCE_SHA256.json',R/'记录/STANDARD_REFERENCE.json',R/'记录/EXTERNAL_ADAPTER_SHA256.json',R/'A_PRESSURE.json']
 for name,h in p.read(R/'记录/EXTERNAL_ADAPTER_SHA256.json').items():assert p.sha(R/name)==h
 for name in ('logging_adapter.py','train_with_logs.py','verify_logging.py','runtime.py','run_t26.py','early_eval_spare.py','publish_t26.py','run_t26_parallel.py','parallel_A.py'):src.append(R/'脚本'/name)
 if (R/'记录/LOGGING_VERIFIED.json').exists():src.append(R/'记录/LOGGING_VERIFIED.json')
 fs=sorted((R/'A').rglob('*.json'));assert len(fs)==648 and not any(f.name.startswith('ERROR_') for f in fs);src+=fs
 selection=p.read(R/'A_PRESSURE.json');state=p.read(R/'RUN_STATE.json');p.dump(assets/'RUN_STATUS_SNAPSHOT.json',state)
 p.dump(assets/'PROVENANCE.json',{'science_SHA':p.SHA,'order_collab':'c1a962c','profiles':{'p170':[170,210],'p155':[155,195],'p140':[140,180]},'A_cases':648,'fixed_in_space':['rally','c_10_0','c_10_-5','c_6_5','c_10_5'],'gstar':'internal reference only; not baseline','selection':'native experiments.t2_readout pressure; no hand edits','training':'Only if native selected non-null, both arms seed0/1 each3000/2workers from zero on selected profile; no branches','logging':'External read-only adapter preserves native worker output; urgent_deadline passed unchanged; same-batch weights/RNG equality and actual EACC+SAC pressure smoke event growth required before training','standard_results':'Historical standard curves retained by fixed version/hash; no standard retraining; A p170 same-commit concurrent screen is explicitly required by order'})
 report=['# T2.6 时间压力测试：'+stage,'',f'生成{p.now()}；科学`{p.SHA}`；执行书collab`c1a962c`。本机测试结果见PYTEST_VERIFIED（预期41）。仅改变紧急任务截止时间区间，其余算法仍t2-learner-2.5.1；任务分配/分位、经济任务、动力学、执行器、动作、奖励和约束冻结。','', '## A 原生筛查与选档','三档各9策略×24，共648例无ERROR。5个阶段动作空间内固定策略、纯MPC、内部G★参照、标准任务SAC/EACC s0的4k探针。p170是执行书要求的本机同提交同期对照，未重训标准任务。','```json',json.dumps(selection,ensure_ascii=False,indent=2),'```','F1–F3/D1和selected/tightest_feasible以原生文件为准；没有合格档就停，不因收益或方便手选档。F2用阶段动作空间内固定策略或学习探针的可达并集，不用G★替代可达性。']
 if stage=='mechanism1000':
  src.append(R/'MECHANISM1000.json');report+=['','## 千步机制早报（非最终判定）','```json',json.dumps(p.read(R/'MECHANISM1000.json'),ensure_ascii=False,indent=2),'```','各字段列实际n/首末步数，严格after1000早期窗口；最终M由原生t26按完整model_stats的>=1000探针计算。首次model_td非空的决策步数按原生已有日志读取，不按墙钟估算。c只报告，不作门。早报异常仅报告，不改动训练。']
 if stage=='final':
  d=p.read(R/'C_READOUT.json');decision=p.read(R/'DECISION.json');assert d['all_complete'] and decision['verdict']!='incomplete' and selection['selected'] is not None
  cf=sorted((R/'C_eval').rglob('*.json'));assert len(cf)==672 and not any(f.name.startswith('ERROR_') for f in cf);src+=cf+[R/'C_READOUT.json',R/'DECISION.json']
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  modelsha=[];events={};activation={}
  for arm in ('e_sac','e_mve'):
   for seed in (0,1):
    tag=R/f'B/TRAIN_{arm}_s{seed}';fin=p.read(tag/'finished.json');manifest=p.read(tag/'manifest.json');assert fin['steps']==3000 and manifest['learner_version']=='t2-learner-2.5.1'
    for name in ('manifest.json','finished.json','episodes.json','model_stats.json','heartbeat.json'):src.append(tag/name)
    for name in ('progress.json','progress.jsonl','episodes.jsonl','ADAPTER_MANIFEST.json'):
     f=tag/'live'/name
     if f.exists():src.append(f)
    rows=p.read(tag/'model_stats.json');used=[x for x in rows if x.get('model_td') is not None];activation[f'{arm}_s{seed}']={'feedback_enabled':arm=='e_mve','first_native_model_row_step':used[0]['steps'] if used else None,'model_probe_count':len(used),'T251_first_model_row_reference':(512,537)[seed] if arm=='e_mve' else None}
    try:
     ea=EventAccumulator(str(tag/'live/tensorboard'),size_guidance={'scalars':0});ea.Reload();out={}
     for key in ea.Tags()['scalars']:
      v=ea.Scalars(key);assert v and all(math.isfinite(x.value) for x in v) and all(b.step>=a.step for a,b in zip(v,v[1:]));out[key]={'n':len(v),'first_step':v[0].step,'last_step':v[-1].step,'last_value':v[-1].value}
     events[f'{arm}_s{seed}']={'status':'parsed','tags':out,'last_progress_steps':p.read(tag/'live/progress.json').get('steps')}
    except Exception as e:events[f'{arm}_s{seed}']={'status':'MISSING_OR_INVALID','reason':repr(e),'scientific_finished3000':True,'not_retrained':True}
    for f in tag.rglob('*.zip'):modelsha.append({'path':str(f),'SHA256':p.sha(f),'binary_not_published':True})
  p.dump(assets/'MODEL_ACTIVATION.json',activation);p.dump(assets/'TENSORBOARD_EVENT_SUMMARY.json',events);p.dump(assets/'LOCAL_MODEL_SHA256.json',modelsha)
  report+=['','## 原生t26判定','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','', '|臂/种子|步数|任务|n|平均回报|捕获|unsafe|','|---|---:|---|---:|---:|---:|---:|']
  for arm,per in d['arms'].items():
   for k,st in sorted(per.items(),key=lambda x:int(x[0])):
    for group,v in (('all',st),('urgent',st['urgent']),('economy',st['economy'])):report.append(f"|{arm}|{k}|{group}|{v['n']}|{v['mean_return']}|{v['capture']}|{v['unsafe']}|")
  report+=['','## 同开局配对与紧急捕获曲线','```json',json.dumps(d['paired'],ensure_ascii=False,indent=2),'```','', '## 模型项实际启动','```json',json.dumps(activation,ensure_ascii=False,indent=2),'```','E1以500–3000六点等权均值、两个种子均正且平均>=5为准；U是预登记的跨检查点紧急捕获差合计，不是不同开局数或独立样本量。末点与安全另行判定，机制启用/梯度同向不等于收益或理论证明。0步照单评估但不纳入六点效率门。']
 elif stage=='A':
  report+=['','## 后续范围','原生selected为空则不启动B/C，报告tightest_feasible后停止；若非空，本次A推送成功且日志门通过后，已授权自动启动两臂×两种子各3000、各2worker，再7点×24共672评估与原生t26判定，不再另请批准。']
 report+=['','## 标准任务固定对照与日志边界','```json',json.dumps(p.read(R/'记录/STANDARD_REFERENCE.json'),ensure_ascii=False,indent=2),'```','滚动日志约30秒准确决策/更新/回合/吞吐（ETA另标估计），逐回合真实和训练回报、成功率及实际原生critic/min-Q和模型反馈。独立Q1/Q2未提供则MISSING，不额外forward/backward/RNG。外部适配器新增urgent_deadline参数只原值透传；入场双臂实际pressure smoke及native update/RNG验证见LOGGING_VERIFIED。SAC无候选/反馈模型项关闭，不伪造模型探针。','本轮Windows后台与Codex生命周期解耦，全部隐藏无密码无提权；实际墙钟见finished和LAUNCH/EXIT，不把并行墙钟相加。任何重启/错误原件保全，训练真崩溃同配置从零最多一次，纯日志/发布失败不重训。阶段齐备才判定，缺项MISSING；无后续分支或正式长训。','模型ZIP/Replay/TB二进制本机保留不入Git；发布必要原始A648与C672（若已完成）、4run训练5日志、原生读出/判定、机制及激活步数、滚动JSON/TB解析/外部源码验收/pytest/源SHA/异常与真实墙钟，普通push保护活动HEAD/index。']
 for name in ('A_PARALLEL_HANDOFF.json','A_TAIL_PARALLEL_HANDOFF.json','A_PARALLEL_EXIT.json'):
  f=R/'记录'/name
  if f.exists():src.append(f)
 if (R/'记录/A_TAIL_PARALLEL_HANDOFF.json').exists():report+=['','尾部加速偏离：原p170只读筛查2路成为瓶颈后，核对PID/创建时间并SHA保全全部完成JSON，受控停止该只读池；当时未完成案例由动态池按相同参数补齐。无训练停止或重启、没有更改样本和选择规则；完整记录见A_TAIL_PARALLEL_HANDOFF。']
 for f in src:assert f.exists(),str(f);p.copy(f,assets)
 text='\n'.join(report)+'\n';(assets/'RESULT.md').write_text(text,encoding='utf-8');(assets/('RESULT_A.md' if stage=='A' else 'RESULT_FINAL.md' if stage=='final' else 'RESULT_MECHANISM1000.md')).write_text(text,encoding='utf-8');(assets/'README.md').write_text('T2.6/'+stage+'；入口RESULT.md，选择/判据以原生文件为准，无后续分支。\n',encoding='utf-8')
p.build=build
src=inspect.getsource(p.main).replace("assert stage in ('A','B','P','N','K','L');","assert stage in ('A','mechanism1000','final');",1)
src=src.replace("prefix='evidence/T2_3_TEST/'+stage if key=='science' else 'lower/handoffs/t2_3_test_20261023'","prefix=('evidence/T2_6_PRESSURE/mechanism1000' if stage=='mechanism1000' else 'evidence/T2_6_PRESSURE') if key=='science' else 'lower/handoffs/t2_6_pressure_20261027'")
exec(compile(src,'verified_t26_transport','exec'),p.__dict__)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['A','mechanism1000','final'],required=True);args=ap.parse_args();lock=R/'记录/PUBLICATION.lock';owned=False
 try:
  for _ in range(90):
   try:
    with lock.open('x') as f:f.write(str(os.getpid()))
    owned=True;break
   except FileExistsError:time.sleep(2)
  assert owned;p.main(args.stage)
 except Exception:p.dump(R/f'记录/PUBLICATION_{args.stage}_ERROR.json',{'jst':p.now(),'traceback':traceback.format_exc()});raise
 finally:
  if owned:lock.unlink()
