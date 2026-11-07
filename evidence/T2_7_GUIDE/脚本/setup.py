from pathlib import Path
R=Path(__file__).resolve().parents[1];OLD=R.parent/'T2_6W1_CUTOFF_20261029'
src=(OLD/'脚本/runtime.py').read_text(encoding='utf-8')
prefix=src[:src.index('def current(')].replace("SHA='e2d4c14bc4d3bbe385c44e65f3c3dcdbd41372d5'","SHA='2208a5176e88718c2c1db3e4a19c58853da49739'").replace("AUDIT='91b346b664fe59e1f9c00af38f45e65735b27839'","AUDIT=SHA\nW1=R.parent/'T2_6W1_CUTOFF_20261029'").replace("ORDER='185765b'","ORDER='8708f39'")
helpers=src[src.index('def launch('):].replace('subprocess.BELOW_NORMAL_PRIORITY_CLASS','subprocess.NORMAL_PRIORITY_CLASS')
middle='''def current(phase,training=False,evaluation=False):
 old=read(CURRENT)
 x=dict(old,topic='t2_7_guide_20261031',updated_at=now(),scientific_commit=SHA,run_order='docs/collaboration/upper/run_orders/T2_7_GUIDE_20261031.md',order_collab_commit=ORDER,
 execution_state=phase,training_active=training,evaluation_active=evaluation,mechanism_audit_active=False,automatic_followup='DISABLED' if phase in ('complete_delivered','failed_stopped') else 'OS_PIPELINE_ACTIVE',native_state=str(R/'RUN_STATE.json'),
 handoff='docs/collaboration/lower/handoffs/t2_7_guide_20261031/RESULT_FINAL.md',local_report=str(SPACE/'上层交付/最新/RESULT_FINAL.md'),
 current_scope={'arms':['e_mve','e_sac'],'seeds':[0,1],'guide':True,'guide_p0':0.5,'guide_until':2000,'steps_each':4000,'workers_each':2,'cpu_affinity':'CPU0-11 shared','priority':'Normal','checkpoints':MARKS,'openings':'294000-294047','criterion':'t27'},
 pending_experiment={'status':'completed' if phase=='complete_delivered' else 'approved_executing','not_started':phase=='preflight'},
 conclusion_boundary='2x2: primary guided EACC vs reused SAC; guide-only, labels incremental and vs unguided EACC are reported separately. No presumed gain.',next_action='Execute authorized T2.7 only; no later experiment authorized.')
 if phase=='preflight':
  x.update(previous_round_snapshot={'path':str(R/'记录/CURRENT_BEFORE.json'),'SHA256':sha(R/'记录/CURRENT_BEFORE.json')},previous_completed_result=old.get('result'),evidence_commit=None,collab_delivery_commit=None)
  x.pop('result',None)
  for k in ('reference_review','reference_review_local','audit_parallel_resource_change','resource_policy_path','torch_resource_verification','stable_throughput_state'):x.pop(k,None)
 dump(CURRENT,x)
def state(**kw):
 S.update(kw,updated_jst=now(),pid=os.getpid(),processes={k:{'pid':p.pid,'running':p.poll() is None} for k,p in ACTIVE.items()},rolling={f'{arm}_s{s}':read(R/f'B/TRAIN_{arm}_s{s}/live/progress.json') for arm in ('e_mve','e_sac') for s in (0,1)})
 dump(R/'RUN_STATE.json',S)
'''
(R/'脚本/runtime.py').write_text(prefix+middle+helpers,encoding='utf-8')
p=R/'脚本/logging_adapter.py';s=p.read_text(encoding='utf-8').replace("'mission':info.get('mission')","'mission':info.get('mission'),'guided':bool(info.get('guided',False))")
s=s.replace("{'true':0.,'train':0.}","{'true':0.,'train':0.,'guided':0}").replace("state['true']+=v['true_reward'];state['train']+=d['reward_train']","state['true']+=v['true_reward'];state['train']+=d['reward_train'];state['guided']+=int(v.get('guided',False))")
s=s.replace("'episode_index':self.episodes","'guided':state['guided'],'episode_index':self.episodes").replace("'window_n'):self.writer","'window_n','guided'):self.writer")
s=s.replace("self.model=model;old=model.replay_buffer.add","self.model=model\n  import torch\n  dump(self.live/'TORCH_THREADS.json',{'intra_op':torch.get_num_threads(),'interop':torch.get_num_interop_threads(),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS'),'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS')})\n  old=model.replay_buffer.add")
s=s.replace('explicit user authorization2026-10-26','T2.7 authorized logging-only observer; fresh preflight required')
p.write_text(s,encoding='utf-8')
p=R/'脚本/verify_logging.py';s=p.read_text(encoding='utf-8').replace("'--smoke','--output'","'--smoke','--guide','--output'")
s=s.replace("assert before=={str(p):a.sha(p) for p in files}","assert before=={str(p):a.sha(p) for p in files}\n assert a.read(out/'manifest.json')['guide'] is True\n assert a.read(out/'live/TORCH_THREADS.json')['intra_op']==1\n assert [x['guided'] for x in eps]==[x['guided'] for x in logged]\n assert fin['replay'].get('guided',0)>0")
p.write_text(s,encoding='utf-8')
# Reuse transport only, not any old scientific decision or packaging logic.
pub=(OLD/'脚本/publish.py').read_text(encoding='utf-8')
transport=pub[:pub.index('def early():')]+pub[pub.index('def science_commit('):pub.index('def final():')]
transport=transport.replace('t2_6w1_cutoff_20261029','t2_7_guide_20261031').replace('evidence/T2_6W1_CUTOFF/','evidence/T2_7_GUIDE/').replace('T2.6W1 cutoff training, registered verdict and rank audit','T2.7 guided 2x2 training and registered verdict')
(R/'脚本/transport.py').write_text(transport,encoding='utf-8')
print('fresh T2.7 runtime, guided telemetry and transport prepared')
