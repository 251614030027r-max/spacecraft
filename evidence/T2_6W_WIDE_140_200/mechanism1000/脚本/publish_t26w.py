"""Independent T26W evidence and byte-preserving ordinary Git publication; no scientific execution."""
from pathlib import Path
import importlib.util,inspect,json,math,os,argparse,time,traceback,shutil,datetime
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('transport',R.parent/'T2_3_TEST_20261025/脚本/publish_t23.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.R=R;p.SHA='e1166eb5daee069e7f14b6f48d24dcf43c36d66f'
def build(stage,assets):
 assert p.git(p.REPO,'rev-parse','HEAD')==p.SHA and p.read(R/'记录/PYTEST_VERIFIED.json')['status']=='passed' and p.read(R/'记录/LOGGING_VERIFIED.json')['status']=='passed'
 src=[R/'pytest.log',R/'输入/T2_6W_WIDE_140_200_69d49fa.md',R/'输入/USER_ORDER.md',R/'记录/PYTEST_VERIFIED.json',R/'记录/PREPARE_VERIFIED.json',R/'记录/LOGGING_VERIFIED.json',R/'记录/EXTERNAL_ADAPTER_SHA256.json',R/'记录/STANDARD_REFERENCE.json',R/'记录/FOUR_TRAINING_LAUNCH.json']
 for name,h in p.read(R/'记录/EXTERNAL_ADAPTER_SHA256.json').items():assert p.sha(R/name)==h
 src+=list((R/'脚本').glob('*.py'))+list((R/'smoke').glob('*.json'))
 if (R/'记录/FIRST_LIVE_GROWTH_VERIFIED.json').exists():src.append(R/'记录/FIRST_LIVE_GROWTH_VERIFIED.json')
 if (R/'记录/DESKTOP_CPU_RESERVATION.json').exists():src.append(R/'记录/DESKTOP_CPU_RESERVATION.json')
 p.dump(assets/'RUN_STATUS_SNAPSHOT.json',p.read(R/'RUN_STATE.json'))
 p.dump(assets/'PROVENANCE.json',{'science_SHA':p.SHA,'order_collab':'69d49fa','deadline':[140.,200.],'learner':'t2-learner-2.6','pooled_calibration':'EACC true/SAC false','four_runs':'e_sac/e_mve x seeds0/1 each4000,2asyncworkers,from zero','evaluation':'48 openings294000-294047 x9marks0,500,...4000 x4=1728','reference':'7strategies x48=336; current SHA/current deadline only; no T26 p140/p155 reuse','criterion':'native t26w;500-3000 gate only; late3000/3500/4000 and pressure bins reported separately','no_branches':True,'limitations':'Task deadline and shared calibration both differ from historical standard training, so historical difference alone cannot identify a single cause. New24 openings share no old evaluation results. Sandbox calibration362 is external reference, not local observation. Nearfield direct-capture label omission not repaired in this freeze.'})
 report=['# T2.6W 固定140–200秒、共享校准与48开局验证：'+stage,'',f'生成{p.now()}，科学`{p.SHA}`，执行书collab`69d49fa`。本机44项测试/真实deadline冒烟及日志入场验收见随附回执。SAC/EACC各两个种子均从零4000、每组2个异步worker；没有预筛选档位。EACC共享阶段/入口校准，SAC模型项关闭且不共享校准。任务动力学、执行器、奖励、动作、K、β、候选集合与阈值冻结。', '判定窗口为500–3000；4000完成不等于收敛，不用末期最佳点替换预登记判据。原生pass仅本轮开发门通过，不能替代论文级证据。']
 if (R/'MECHANISM1000.json').exists():
  src.append(R/'MECHANISM1000.json');report+=['','## 千步机制早报（早期窗口）','```json',json.dumps(p.read(R/'MECHANISM1000.json'),ensure_ascii=False,indent=2),'```','严格after1000的原生TB探针，列样本数与首末步数；最终M按原生完整model_stats。c只报告，不作门。T251标准3worker的512/537和沙箱2worker共享362是参照，实际启动步数单独列；不能把不同任务/worker的历史差单独归因于共享校准。机制启用、w2高或梯度同向不等于部署收益。']
 if stage=='mechanism1000':assert (R/'MECHANISM1000.json').exists()
 if stage=='final':
  readout=p.read(R/'C_READOUT.json');decision=p.read(R/'DECISION.json');assert readout['all_complete'] and decision['verdict']!='incomplete'
  cases=list((R/'C_eval').rglob('*.json'));reference=list((R/'A_ref').glob('*.json'));assert len(cases)==1728 and len(reference)==336 and not any(f.name.startswith('ERROR_') for f in cases+reference)
  src+=cases+reference+[R/'C_READOUT.json',R/'DECISION.json',R/'记录/EVAL_POOL_EXIT.json',R/'记录/REFERENCE_EXIT.json']
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  activation={};events={};wall={};model_hashes=[]
  for arm in ('e_sac','e_mve'):
   for seed in (0,1):
    tag=f'{arm}_s{seed}';out=R/f'B/TRAIN_{tag}';manifest=p.read(out/'manifest.json');finished=p.read(out/'finished.json');assert finished['steps']==4000 and manifest['git_head']==p.SHA and manifest['learner_version']=='t2-learner-2.6' and manifest['pooled_calibration']==(arm=='e_mve') and manifest['urgent_deadline_s']==[140.,200.]
    src += [out/n for n in ('manifest.json','finished.json','episodes.json','model_stats.json','heartbeat.json')]
    for name in ('progress.json','progress.jsonl','episodes.jsonl','ADAPTER_MANIFEST.json','LOGGER_ERROR.json'):
     f=out/'live'/name
     if f.exists():src.append(f)
    rows=p.read(out/'model_stats.json');used=[x for x in rows if x.get('model_td') is not None];activation[tag]={'enabled':arm=='e_mve','pooled_calibration':manifest['pooled_calibration'],'first_model_step':used[0]['steps'] if used else None,'model_probe_count':len(used),'T251_reference_step':(512,537)[seed] if arm=='e_mve' else None,'sandbox_pooled_reference_step':362 if arm=='e_mve' else None}
    wall[tag]={'steps':finished['steps'],'wall_s':finished.get('wall_s'),'seconds_per_accepted_decision':finished.get('wall_s')/finished['steps'] if isinstance(finished.get('wall_s'),(float,int)) else 'MISSING','updates':finished.get('updates'),'elapsed_parallel_wall_not_sum':True}
    attempt=1 if (R/f'记录/RESTART_{arm}_s{seed}.json').exists() else 0
    wrapper_exit=R/f'记录/TRAIN_{arm}_s{seed}_attempt{attempt}_EXIT.json'
    wall[tag]['wrapper_wall_s']=p.read(wrapper_exit).get('wall_s') if wrapper_exit.exists() else 'MISSING'
    wall[tag]['wrapper_wait_polling_resolution_s']=15
    wall[tag]['attempt']=attempt
    try:
     ev=EventAccumulator(str(out/'live/tensorboard'),size_guidance={'scalars':0});ev.Reload();values={}
     required=('progress/steps','progress/updates','progress/episodes','rollout/return_true','rollout/return_train','rollout/success_rate_last50','critic/real_td','critic/q_pi_mean')
     assert all(k in ev.Tags()['scalars'] for k in required)
     for key in ev.Tags()['scalars']:
      seq=ev.Scalars(key);assert seq and all(math.isfinite(x.value) for x in seq) and all(y.step>=x.step for x,y in zip(seq,seq[1:]));values[key]={'n':len(seq),'first_step':seq[0].step,'last_step':seq[-1].step,'last_value':seq[-1].value}
     assert p.read(out/'live/progress.json')['steps']==4000 and values['progress/steps']['last_value']==4000
     episodes=[json.loads(s) for s in (out/'live/episodes.jsonl').read_text(encoding='utf-8').splitlines()];assert len(episodes)==len(p.read(out/'episodes.json'))
     events[tag]={'status':'parsed_verified','tags':values,'actual_logged_episodes':len(episodes),'Q1_Q2_separate':'MISSING; native min-Q q_pi_mean only'}
    except Exception as error:events[tag]={'status':'MISSING_OR_INVALID','error':repr(error),'science_finished4000':True,'not_retrained':True}
    for f in out.rglob('*.zip'):model_hashes.append({'path':str(f),'SHA256':p.sha(f),'binary_not_published':True})
  p.dump(assets/'MODEL_ACTIVATION.json',activation);p.dump(assets/'WALL_CLOCK.json',wall);p.dump(assets/'TENSORBOARD_EVENT_SUMMARY.json',events);p.dump(assets/'LOCAL_MODEL_SHA256.json',model_hashes)
  for pattern in ('RESTART_*.json','TRAIN_*_LAUNCH.json','TRAIN_*_EXIT.json'):
   for f in (R/'记录').glob(pattern):src.append(f)
  for f in (R/'记录').glob('TRAIN_*.err.log'):
   if f.stat().st_size:src.append(f)
  for f in (R/'记录').glob('*LOGGING_ONLY_FAILURE.json'):src.append(f)
  report+=['','## 原生t26w判定（500–3000）','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','C0为48/48。E1双seed都正且均值>=5；U跨检查点紧急捕获差双seed合计>=4且每seed非负，是重复检查点累计次数，不是独立开局数。E2/S取3000末点及登记窗口；不得用4000或最佳检查点替代。','', '|臂/种子|步数|任务|n|平均回报|捕获|unsafe|','|---|---:|---|---:|---:|---:|---:|']
  for arm,per in readout['arms'].items():
   for mark,stats in sorted(per.items(),key=lambda x:int(x[0])):
    for name,v in (('all',stats),('urgent',stats['urgent']),('economy',stats['economy'])):report.append(f"|{arm}|{mark}|{name}|{v['n']}|{v['mean_return']}|{v['capture']}|{v['unsafe']}|")
  report+=['','## 全检查点配对、紧急/经济拆分','```json',json.dumps(readout['paired'],ensure_ascii=False,indent=2),'```','', '## 末期稳定性、压力分箱、双方都成功的燃料/时间（只报告）','```json',json.dumps(decision.get('reported','MISSING'),ensure_ascii=False,indent=2),'```','只在双方均捕获的子集上比较Δv/时间，按紧急/经济区分；这是一项条件比较，不能外推为所有开局的经济优势。末期3000/3500/4000极差用于给定预算的稳定性描述，不宣称数学收敛。','', '## 反馈启动与实际计算开销','```json',json.dumps({'activation':activation,'wall':wall},ensure_ascii=False,indent=2),'```','每臂单列真实决策开销与完整wall，不把四个并行wall相加冒充整体用时。完整机制判据和各值在原生DECISION，非零模型损失和梯度不代替配对性能证据。G★仅内部系统参照，不作为阶段动作空间内的学习基线；参照结果不会用于调整本轮冻结训练。']
 report+=['','## 标准任务历史固定对照','```json',json.dumps(p.read(R/'记录/STANDARD_REFERENCE.json'),ensure_ascii=False,indent=2),'```','旧标准结果仅作固定SHA参照，本轮SAC和EACC均新训。任务deadline和校准共享两项相对历史变化，不能仅据历史差认定单因素因果收益。近场直接捕获标签漏判本次未修，模型完备性不作保证。','', '## 日志、异常与交付','外部只读适配器不改变原_slim返回、原生IPC消息或原replay.add已接受转换，不额外forward/backward/RNG；本SHA同批native update权重逐位相等、RNG检查、双臂真实48步异步事件及额外12步非smoke共享校准IPC都在独立目录验证，全部不作科研性能证据。原生smoke关闭候选/共享校准，故另做非smoke12步并检查pooled manifest，不把smoke假充共享校准实证。正式约30秒准确计数、逐回合true/train和原生模型字段，Q1/Q2分值缺失不编造。','训练真崩溃同配置同seed最多一次从零，先保全attempt，不拼Replay/优化器/RNG；纯日志/发布失败不重训。单例错误原件在记录/episode_errors保全后仅同例补一次。模型ZIP/Replay/TB二进制本机保留不入Git。普通Git临时index/--no-filters保护活动HEAD/index，不强推；必要原始1728+336+smoke与模型/源SHA、实际事件解析、原生判据随轮交付。无后续分支或长训授权。']
 if (R/'记录/DESKTOP_CPU_RESERVATION.json').exists():report+=['用户运行中要求桌面预留1个逻辑核：本轮进程限制CPU1–11，逻辑CPU0留桌面；未改科学或worker、未重启。实际墙钟包含该系统资源政策和用户并行工作，不按独占12核估算，详见DESKTOP_CPU_RESERVATION。']
 for f in src:assert f.exists(),str(f);p.copy(f,assets)
 text='\n'.join(report)+'\n';(assets/'RESULT.md').write_text(text,encoding='utf-8');(assets/('RESULT_FINAL.md' if stage=='final' else 'RESULT_MECHANISM1000.md')).write_text(text,encoding='utf-8');(assets/'README.md').write_text('T2.6W/'+stage+'，科学e1166eb，固定140,200与48开局。入口RESULT.md；早报非最终判定。\n',encoding='utf-8')
p.build=build
source=inspect.getsource(p.main).replace("assert stage in ('A','B','P','N','K','L');","assert stage in ('mechanism1000','final');",1)
source=source.replace("prefix='evidence/T2_3_TEST/'+stage if key=='science' else 'lower/handoffs/t2_3_test_20261023'","prefix=('evidence/T2_6W_WIDE_140_200/mechanism1000' if stage=='mechanism1000' else 'evidence/T2_6W_WIDE_140_200') if key=='science' else 'lower/handoffs/t2_6w_wide_20261027'")
source=source.replace('BEFORE_T23_','BEFORE_T26W_').replace('frozen T2.3 stage','frozen T2.6W stage')
exec(compile(source,'verified_t26w_transport','exec'),p.__dict__)
def archive_early_before_final():
 marker=R/'记录/FINAL_PRIOR_STAGE_ARCHIVED.json'
 if marker.exists() or not (R/'记录/LATEST_ARCHIVED.json').exists():return
 latest=p.SPACE/'上层交付/最新';history=p.SPACE/'上层交付/历史'/('T26W_prior_stage_'+datetime.datetime.now(p.TZ).strftime('%Y%m%d_%H%M%S'))
 assert latest.resolve().is_relative_to(p.SPACE.resolve()) and history.resolve().is_relative_to(p.SPACE.resolve()) and not history.exists();history.mkdir()
 hashes={f.relative_to(latest).as_posix():p.sha(f) for f in latest.rglob('*') if f.is_file()}
 for f in list(latest.iterdir()):assert f.resolve().is_relative_to(p.SPACE.resolve()) and not (history/f.name).exists();shutil.move(str(f),str(history/f.name))
 assert hashes=={f.relative_to(history).as_posix():p.sha(f) for f in history.rglob('*') if f.is_file()};p.dump(marker,{'history':str(history),'SHA256':hashes,'jst':p.now()})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['mechanism1000','final'],required=True);args=ap.parse_args();lock=R/'记录/PUBLICATION.lock';owned=False
 try:
  for _ in range(90):
   try:
    with lock.open('x') as f:f.write(str(os.getpid()))
    owned=True;break
   except FileExistsError:time.sleep(2)
  assert owned
  if args.stage=='final':archive_early_before_final()
  p.main(args.stage)
 except Exception:p.dump(R/f'记录/PUBLICATION_{args.stage}_ERROR.json',{'jst':p.now(),'traceback':traceback.format_exc()});raise
 finally:
  if owned:lock.unlink()
