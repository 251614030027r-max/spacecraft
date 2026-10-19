"""Full true missions for a single exploratory interface; never train."""
import argparse,json,time,os,hashlib,subprocess
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

def run(job):
    opening,row,out,profile=job
    from env.t1_b_task import MissionDistribution,T1BReviewEnv
    from env.t1_b_admissible import T1BAdmissibleEnv,ADMISSIBLE_VERSION
    env=T1BAdmissibleEnv(openings=[opening],mission=MissionDistribution(**profile))
    started=time.perf_counter();obs,reset=env.reset();initial=obs.copy();model=None;policy_hash=None
    if row=='zero':
        import torch
        torch.set_num_threads(1)
        from train.t1_b_sac import T1BSAC
        model=T1BSAC('MlpPolicy',env,seed=0,device='cpu',buffer_size=100,learning_starts=2000,
            policy_kwargs={'net_arch':[128,128]},ent_coef=.005,mc_lambda=0.,eq_lambda=.1,delta_lambda=0.,verbose=0)
        policy_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.policy.parameters())).hexdigest()
    steps=[];done=False
    while not done:
        if not env._due or env.generator.remaining_s(env.env)>0 and row!='zero':a=np.zeros(1,np.float32)
        elif model is not None:a=np.asarray(model.predict(obs,deterministic=True)[0]).reshape(-1)
        else:a=np.array([{'early':-1.,'middle':0.,'late':1.}[row]],np.float32)
        obs,r,t,x,i=env.step(a);e=env.env
        steps.append({'t':e.time_s,'r':r,'dv':i['delta_v_step_m_s'],'a':a.tolist(),'review':i['review'],'execution':i['execution'],
            'observation':obs.tolist(),'control_trace':i['resource_control_trace'],
            'q':(e.chaser_state.position-e.target_state.position).tolist(),
            'v':(e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity).tolist()})
        done=t or x
    result={'opening':opening,'row':row,'interface':ADMISSIBLE_VERSION,'training':False,'cohort':'EXPLORATORY_SELECTION_NOT_INDEPENDENT_CONFIRMATION',
        'mission':reset['mission'],'profile':asdict(env.mission_distribution),'policy_sha256':policy_hash,'initial_observation':initial.tolist(),
        'event':i['event'],'discounted_return':sum(e.config.reward.gamma**k*s['r'] for k,s in enumerate(steps)),
        'return':sum(s['r'] for s in steps),'summary':e.summary(),'violations':i.get('violations',{}),'steps':steps,
        'catalogue_overhead_wall_s':env.catalogue_wall_s,'additional_catalogue_solves':env.catalogue_solves,'wall_s':time.perf_counter()-started}
    path=Path(out)/f'{row}_{opening}.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result),encoding='utf-8');temp.replace(path)
    return {k:result[k] for k in ['opening','row','event','discounted_return','wall_s']}

def contract(profile,out):
    from env.t1_b_task import MissionDistribution,T1BReviewEnv
    from env.t1_b_admissible import T1BAdmissibleEnv,CATALOGUE_OBSERVATIONS
    evidence=[]
    for a in [-1.,0.,1.]:
        env=T1BAdmissibleEnv(openings=[292106],mission=MissionDistribution(**profile));obs,_=env.reset()
        reference=T1BReviewEnv(openings=[292106],mission=MissionDistribution(**profile));reference.reset()
        encode=lambda x:json.dumps(asdict(x),sort_keys=True,default=lambda value:value.tolist())
        assert encode(env.env.config)==encode(reference.env.config),'Physics/reward/task changed'
        assert len(obs)==len(CATALOGUE_OBSERVATIONS)==52 and np.isfinite(obs).all()
        choices=env.catalogue();assert len(choices)>=2
        _,_,_,_,info=env.step(np.array([a],np.float32));selected=info['review']['catalogue_index'];tau=choices[selected]
        assert abs(info['execution']['tau_request_s']-tau)<1e-6
        assert info['review']['outcome']=='accepted' and info['execution']['new_plan']
        assert np.allclose(info['executed_action'],[a]),'Replay must retain the actual interface action'
        evidence.append({'a':a,'catalogue':choices,'selected_tau_s':tau,'accepted':True})
    (out/'CONTRACT.json').write_text(json.dumps({'passed':True,'real_first_steps':3,'physics_reward_equal_to_base':True,'evidence':evidence},indent=2),encoding='utf-8')
    print(json.dumps({'contract':'PASS','evidence':evidence}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--openings',default='292100-292107');p.add_argument('--workers',type=int,default=6);p.add_argument('--contract-only',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);profile=json.loads(Path(a.profile).read_text(encoding='utf-8-sig'))
    if a.contract_only:contract(profile,out);return
    from experiments.t1n_eval import parse_openings
    openings=parse_openings(a.openings);assert all(292000<=o<300000 for o in openings),'Formal block sealed'
    jobs=[(o,row,str(out),profile) for o in openings for row in ['early','middle','late','zero']]
    assert not any(out.glob('early_*.json')),'Never duplicate an existing batch'
    state={'pid':os.getpid(),'phase':'running','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'completed':[],'jobs':len(jobs),'args':vars(a),'training':False}
    def save():
        temp=out/'STATUS.tmp';temp.write_text(json.dumps(state,indent=2),encoding='utf-8');temp.replace(out/'STATUS.json')
    save()
    with ProcessPoolExecutor(a.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            r=future.result();state['completed'].append(r);save();print(json.dumps(r),flush=True)
    state['phase']='full_mission_diagnostic_complete';save();(out/'finished.json').write_text(json.dumps({'episodes':len(jobs),'training':False,'finished':time.time()}),encoding='utf-8')
if __name__=='__main__':main()
