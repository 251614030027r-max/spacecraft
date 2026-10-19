"""Bounded same-method continuation. Never overwrite the completed 5k experiment.

Administrative reset at process boundaries is disclosed; open intervals are dropped,
not terminated or joined to another opening. No new reward/algorithm/physics.
"""
import os,sys,json,pickle,random,hashlib,subprocess,time
from pathlib import Path
from functools import partial
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from stable_baselines3.common.vec_env import SubprocVecEnv
from experiments.t1b_repair_v1 import ROOT,make_env,RecordV1,dump,evaluate
from train.t1_b_feasible_sac import FeasibleSupportSAC
PY='D:/py/DRL2/.venv/Scripts/python.exe'
ARMS=['e_sac','e_tcpf']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def env_after_boundary(rank,feedback,counts):
    e=make_env(rank,feedback,0)
    e.episode=counts[rank]+1
    return e
class ContinueRecord(RecordV1):
    def __init__(self,out):
        super().__init__(out,10000);self.marks={7500,10000}
    def _on_training_start(self):
        self.handle=(self.out/'decisions.jsonl').open('x',encoding='utf-8')
        self.save('resume_start_5000')
    def _on_rollout_start(self):
        # Called after all previous gradient updates. Safe recovery artifacts.
        n=self.model.num_timesteps
        if n%1000==0 and not (self.out/f'recovery/{n}/COMPLETE.json').exists():
            p=self.out/f'recovery/{n}';p.mkdir(parents=True,exist_ok=False)
            self.model.save(p/'model');self.model.save_replay_buffer(p/'replay.pkl')
            with (p/'rng.pkl').open('wb') as f:pickle.dump({'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'python':random.getstate()},f)
            dump(p/'COMPLETE.json',{'global_steps':n,'critic_updates':self.model._n_updates,'actor_updates':self.model.actor_updates,'files':{k:sha(p/k) for k in ['model.zip','replay.pkl','rng.pkl']},'clock':'post-update; recovery needs same disclosed boundary reset, never claim uninterrupted trajectory'})
def train(arm):
    old=ROOT/f'结果/TRAIN_{arm}_seed0';out=ROOT/f'结果/CONTINUE_{arm}_seed0_10k'
    assert json.loads((old/'finished.json').read_text())['steps']==5000
    out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    counts=[0]*4
    for line in (old/'decisions.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line)
        if r['event']:counts[r['env']]+=1
    torch.set_num_threads(1)
    env=SubprocVecEnv([partial(env_after_boundary,k,arm=='e_tcpf',counts) for k in range(4)],start_method='spawn')
    model=FeasibleSupportSAC.load(old/'final.zip',env=env,device='cpu',force_reset=True)
    model.load_replay_buffer(old/'replay_final.pkl',truncate_last_traj=False)
    b=model.replay_buffer
    boundary={'open_intervals_dropped':sum(x is not None for x in b._open),'pending_model_prefixes_dropped':sum(x is not None for x in b._prefix),'unfinished_MC_lists_dropped':list(map(len,b._episode))}
    b._open=[None]*4;b._prefix=[None]*4;b._episode=[[] for _ in range(4)]
    b._pending={};b._release_queue=[]
    b.feedback_audit_path=out/'model_prefix_releases.jsonl'
    model.replay_buffer_kwargs['feedback_audit_path']=str(b.feedback_audit_path)
    # Original5k schedule has reached its floor: continuation holds that floor.
    # Do not silently restart the cosine or recompute its total as10k.
    model.learning_rate=1e-5;model.lr_schedule=lambda progress:1e-5
    assert model.num_timesteps==5000 and model._n_updates==4500 and model.actor_updates==1500
    assert b.demo_slots*4==1000 and b.pos*4==6000
    with (old/'rng_final.pkl').open('rb') as f:rng=pickle.load(f)
    np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);random.setstate(rng['python'])
    dump(out/'manifest.json',{'arm':arm,'seed':0,'scope':'global5k to10k same frozen method; no extra seeds','science_commit':'636b65517bf73985b4e544c67decf857b4fffc19','driver_sha256':sha(__file__),'source_models':{n:sha(old/n) for n in ['final.zip','replay_final.pkl','rng_final.pkl']},'source_steps':5000,'start_critic_updates':4500,'start_actor_updates':1500,'same_expert_fraction':.2,'BC_updates':0,'learning_rate_continuation':1e-5,'lr_reason':'original5k cosine floor, no optimizer reset','completed_episodes_before_boundary':counts,'next_openings':[1000000+k*100000+counts[k]+1 for k in range(4)],'boundary':boundary,'boundary_semantics':'Original5k closed process. Abandon remaining unfinished episodes without reward/done fabrication. Retain all old real transitions; unclosed review intervals remain ineligible. No cross-opening folding.','delta_calibration_preserved':model.delta_calibration,'evaluation_fixed_openings':list(range(292140,292148)),'expanded_development_openings':list(range(292120,292128)),'no_posthoc_checkpoint_selection':True})
    model.learn(total_timesteps=5000,reset_num_timesteps=False,callback=ContinueRecord(out))
    model.save(out/'final');model.save_replay_buffer(out/'replay_final.pkl')
    with (out/'rng_final.pkl').open('wb') as f:pickle.dump({'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'python':random.getstate(),'online_steps':model.num_timesteps},f)
    dump(out/'learning_audit.json',model.support_stats);dump(out/'feedback_audit.json',b.feedback_stats);dump(out/'delta_calibration.json',model.delta_calibration)
    dump(out/'finished.json',{'steps':model.num_timesteps,'segment_actual_real_steps':5000,'critic_updates':model._n_updates,'actor_updates':model.actor_updates,'finite':True,'expert_sampling_cumulative':b.sample_counts,'interval_counts':b.interval_counts})
    env.close()
def queue():
    state=json.loads((ROOT/'STATE.json').read_text(encoding='utf-8-sig'))
    assert state['phase']=='matched_5k_complete_readout_required' and not state['active']
    dump(ROOT/'记录/STATE_5K_CLOSED.json',state)
    state.update(phase='matched_seed0_continuation_10k',training_active=True,pid=os.getpid(),active=[],training_scope='same seed0 matched global5k→10k then fixed8 + expanded8 evaluation; no further automatic training')
    procs=[];handles=[]
    for arm in ARMS:
        cmd=[PY,'-B',str(Path(__file__).resolve()),'train',arm]
        stdout=(ROOT/f'记录/continue_{arm}.stdout.log').open('x',encoding='utf-8');stderr=(ROOT/f'记录/continue_{arm}.stderr.log').open('x',encoding='utf-8');handles.extend([stdout,stderr])
        en=os.environ.copy();en.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        p=subprocess.Popen(cmd,cwd='D:/py/DRL2',env=en,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        procs.append((arm,p));state['active'].append({'name':'continue_'+arm,'pid':p.pid,'command':cmd})
    dump(ROOT/'STATE.json',state)
    while any(p.poll() is None for _,p in procs):time.sleep(60)
    for arm,p in procs:dump(ROOT/f'记录/continue_{arm}.EXIT.json',{'exit_code':p.returncode,'pid':p.pid})
    for f in handles:f.close()
    state.update(active=[],training_active=False)
    if any(p.returncode for _,p in procs):
        state.update(phase='continuation_technical_failure_read_stderr');dump(ROOT/'STATE.json',state);return
    state['phase']='evaluate_predeclared_10k_and_expanded_dev';dump(ROOT/'STATE.json',state)
    for arm in ARMS:
        checkpoint=str(ROOT/f'结果/CONTINUE_{arm}_seed0_10k/checkpoints/step_10000.zip')
        evaluate(ROOT/f'结果/eval_{arm}_10000',checkpoint,f'{arm}_rl_10000')
        evaluate(ROOT/f'结果/eval_{arm}_10000_expanded',checkpoint,f'{arm}_rl_10000_expanded',range(292120,292128))
    state.update(phase='matched_10k_complete_readout_required',next_action='Publish complete10k and expandeddev paired evidence. No automatic new training or method changes.')
    dump(ROOT/'STATE.json',state)
if __name__=='__main__':
    if sys.argv[1]=='train':train(sys.argv[2])
    elif sys.argv[1]=='queue':queue()
