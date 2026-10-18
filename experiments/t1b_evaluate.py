"""Plain-environment deterministic evaluation of zero/checkpoints/frozen rules."""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

def tree_choice(spec,obs):
    node=0
    while spec['children_left'][node]!=spec['children_right'][node]:
        node=spec['children_left'][node] if obs[spec['feature'][node]]<=spec['threshold'][node] else spec['children_right'][node]
    return spec['classes'][spec['class_index'][node]]

def run(job):
    row,opening,out,profile,seed,save_zero=job
    from env.t1_b_task import T1BReviewEnv,MissionDistribution
    from experiments.t1b_diagnostic import policy_action
    label=row.split(':')[0] if ':' in row else row
    if ':' in row:label+='_'+hashlib.sha256(row.encode()).hexdigest()[:8]
    path=Path(out)/f'{label}_{opening}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    started=time.perf_counter();mission=MissionDistribution(**profile) if profile else MissionDistribution()
    env=T1BReviewEnv(openings=[opening],mission=mission);obs,reset=env.reset()
    model=None;spec=None;weight_hash=None
    if row=='zero' or row.startswith('model:'):
        import torch
        torch.set_num_threads(1)
        from train.t1_b_sac import T1BSAC
        if row=='zero':
            model=T1BSAC('MlpPolicy',env,seed=seed,device='cpu',buffer_size=100,
                learning_starts=2000,policy_kwargs={'net_arch':[128,128]},ent_coef=.005,
                mc_lambda=0.,eq_lambda=.1,delta_lambda=0.,verbose=0)
            if save_zero:model.save(Path(out)/f'zero_seed_{seed}')
        else:model=T1BSAC.load(row.split(':',1)[1],device='cpu')
        weight_hash=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.policy.parameters())).hexdigest()
    elif row.startswith('tree:'):
        spec=json.loads(Path(row.split(':',1)[1]).read_text(encoding='utf-8'))
    steps=[];ended=False;initial_obs=obs.tolist()
    while not ended:
        if not env._due:a=env.continue_action()
        elif model is not None:a=np.asarray(model.predict(obs,deterministic=True)[0]).reshape(-1)
        else:a=policy_action(env,tree_choice(spec,obs) if spec else row,obs)
        obs,r,t,x,i=env.step(a);e=env.env
        steps.append({'t':e.time_s,'r':r,'dv':i['delta_v_step_m_s'],'a':a.tolist(),
            'review':i['review'],'execution':i['execution'],'observation':obs.tolist(),
            'control_trace':i['resource_control_trace'],
            'q':(e.chaser_state.position-e.target_state.position).tolist(),
            'v':(e.chaser_state.rotation@e.chaser_state.velocity-e.target_state.rotation@e.target_state.velocity).tolist()})
        ended=t or x
    result={'opening':opening,'row':row,'mission':reset['mission'],'profile':asdict(mission),
        'initial_observation':initial_obs,'policy_sha256':weight_hash,'event':i['event'],
        'return':sum(s['r'] for s in steps),'discounted_return':sum(e.config.reward.gamma**k*s['r'] for k,s in enumerate(steps)),
        'summary':e.summary(),'end_range_m':e.metrics().range_m,'violations':i.get('violations',{}),
        'steps':steps,'wall_s':time.perf_counter()-started}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,default=float),encoding='utf-8');tmp.replace(path)
    return {k:result[k] for k in ['row','opening','event','return','discounted_return','wall_s']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile')
    p.add_argument('--openings',required=True);p.add_argument('--rows',required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    from experiments.t1n_eval import parse_openings
    openings=parse_openings(a.openings)
    if any(300000<=o<1000000 for o in openings):raise ValueError('Formal block remains sealed in this overnight development')
    profile=json.loads(Path(a.profile).read_text(encoding='utf-8-sig')) if a.profile else None
    jobs=[(r,o,str(out),profile,a.seed,o==openings[0] and r=='zero') for r in a.rows.split(',') for o in openings]
    (out/'manifest.json').write_text(json.dumps({'args':vars(a),'jobs':jobs},indent=2),encoding='utf-8')
    with ProcessPoolExecutor(a.workers) as pool:
        fs=[pool.submit(run,j) for j in jobs]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    (out/'finished.json').write_text(json.dumps({'completed':True,'time':time.time()}))
if __name__=='__main__':main()
