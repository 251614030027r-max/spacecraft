"""Real matched-review interval validation; calibration and validation are disjoint."""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

def clean(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,list):return [clean(v) for v in x]
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    return x

def job(args):
    opening,out,profile=args;path=Path(out)/f'audit_{opening}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    from env.t1_b_task import T1BReviewEnv,MissionDistribution
    from env.t1_b_interval_cf import IntervalMissionPredictor,IntervalModelConfig,make_payload,calibrate_interval
    from env.t1_counterfactual import CounterfactualPredictor
    from experiments.t1b_cf_audit import truth_rollout
    from experiments.t1b_diagnostic import policy_action
    from dynamics.spacecraft import pack_state
    start=time.perf_counter();mission=MissionDistribution(**profile)
    e=T1BReviewEnv(openings=[opening],mission=mission);obs,reset=e.reset()
    cfg=IntervalModelConfig();pred=IntervalMissionPredictor(e,cfg)
    truth=CounterfactualPredictor(shadow_factory=lambda:T1BReviewEnv(openings=[opening],mission=mission))
    records=[];ended=False
    while not ended:
        a=policy_action(e,'tau110',obs) if e._due else e.continue_action()
        if e._due and e._review_index in [0,3,6,9,12,15]:
            payload=make_payload(e,pred,a,cfg)
            if payload is not None:
                alt=payload['a_tilde'];tf=truth_rollout(truth,e,a,cfg.maximum_macro_steps)
                ta=truth_rollout(truth,e,alt,cfg.maximum_macro_steps)
                gate=calibrate_interval(payload,tf['return'],tf['observation'],len(tf['rewards']),tf['event'])
                errors={}
                for label,pr,tr in [('factual',payload['factual'],tf),('alternative',payload['alternative'],ta)]:
                    errors[label]={'reward':abs(pr['return']-tr['return']),
                        'position_m':float(np.linalg.norm(pr['q']-tr['q'])),
                        'velocity_m_s':float(np.linalg.norm(pr['v']-tr['v'])),
                        'observation':float(np.linalg.norm(pr['observation']-tr['observation'])),
                        'terminal_match':pr['event']==tr['event']}
                records.append({'snapshot':{'target':pack_state(e.env.target_state).tolist(),
                    'chaser':pack_state(e.env.chaser_state).tolist(),'observation':obs.tolist(),
                    'a':a.tolist(),'alt':alt.tolist(),'t':e.env.time_s,'review_index':e._review_index},
                    'model_factual':clean(payload['factual']),'model_alternative':clean(payload['alternative']),
                    'truth_factual':clean(tf),'truth_alternative':clean(ta),'errors':errors,
                    'training_weight':gate['weight'],'delayed_gate':gate,
                    'true_local_difference':ta['return']-tf['return'],
                    'model_local_difference':payload['alternative']['return']-payload['factual']['return'],
                    'true_physical_difference_m':float(np.linalg.norm(ta['q']-tf['q']))})
        obs,r,t,x,info=e.step(a);ended=t or x
    result={'opening':opening,'mission':reset['mission'],'config':asdict(cfg),'records':records,
            'event':info['event'],'summary':e.env.summary(),'wall_s':time.perf_counter()-start}
    path.write_text(json.dumps(clean(result),default=float),encoding='utf-8')
    return {'opening':opening,'pairs':len(records),'valid':sum(r['training_weight']>0 for r in records),
            'event':info['event'],'wall_s':result['wall_s']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--openings',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    profile=json.loads(Path(a.profile).read_text(encoding='utf-8-sig'))
    from experiments.t1n_eval import parse_openings
    (out/'manifest.json').write_text(json.dumps(vars(a)|{'scheduled_reviews':[0,3,6,9,12,15]},indent=2),encoding='utf-8')
    with ProcessPoolExecutor(a.workers) as pool:
        fs=[pool.submit(job,(o,str(out),profile)) for o in parse_openings(a.openings)]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    (out/'finished.json').write_text(json.dumps({'completed':True,'time':time.time()}))
if __name__=='__main__':main()
