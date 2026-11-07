"""Fixed reset-time real/model paired intervals under a new public mission."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

def job(args):
    opening,out,profile=args;path=Path(out)/f'audit_{opening}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    from env.t1_b_task import T1BReviewEnv,MissionDistribution
    from env.t1_b_interval_cf import IntervalMissionPredictor,IntervalModelConfig,make_payload,calibrate_interval
    from env.t1_counterfactual import CounterfactualPredictor
    from experiments.t1b_cf_audit import truth_rollout
    from experiments.t1b_diagnostic import policy_action
    from experiments.t1b_interval_audit import clean
    from dynamics.spacecraft import pack_state
    start=time.perf_counter();mission=MissionDistribution(**profile)
    e=T1BReviewEnv(openings=[opening],mission=mission);obs,reset=e.reset()
    cfg=IntervalModelConfig();pred=IntervalMissionPredictor(e,cfg)
    a=policy_action(e,'earliest' if e.mission['urgent'] else 'tau130',obs)
    payload=make_payload(e,pred,a,cfg);records=[]
    if payload is not None:
        truth=CounterfactualPredictor(shadow_factory=lambda:T1BReviewEnv(openings=[opening],mission=mission))
        tf=truth_rollout(truth,e,a,cfg.maximum_macro_steps);ta=truth_rollout(truth,e,payload['a_tilde'],cfg.maximum_macro_steps)
        gate=calibrate_interval(payload,tf['return'],tf['observation'],len(tf['rewards']),tf['event'])
        errors={}
        for label,pr,tr in [('factual',payload['factual'],tf),('alternative',payload['alternative'],ta)]:
            errors[label]={'reward':abs(pr['return']-tr['return']),
                'position_m':float(np.linalg.norm(pr['q']-tr['q'])),
                'velocity_m_s':float(np.linalg.norm(pr['v']-tr['v'])),
                'observation':float(np.linalg.norm(pr['observation']-tr['observation'])),
                'terminal_match':pr['event']==tr['event']}
        records.append({'snapshot':{'target':pack_state(e.env.target_state).tolist(),'chaser':pack_state(e.env.chaser_state).tolist(),
            'observation':obs.tolist(),'a':a.tolist(),'alt':payload['a_tilde'].tolist(),'t':0.,'review_index':0},
            'model_factual':clean(payload['factual']),'model_alternative':clean(payload['alternative']),
            'truth_factual':clean(tf),'truth_alternative':clean(ta),'errors':errors,'training_weight':gate['weight'],
            'delayed_gate':gate,'true_local_difference':ta['return']-tf['return'],
            'model_local_difference':payload['alternative']['return']-payload['factual']['return'],
            'true_physical_difference_m':float(np.linalg.norm(ta['q']-tf['q']))})
    result={'kind':'RESET_PREFIX_ONLY_NOT_A_FULL_MISSION','opening':opening,'mission':reset['mission'],
            'records':records,'wall_s':time.perf_counter()-start}
    path.write_text(json.dumps(clean(result),default=float),encoding='utf-8')
    return {'opening':opening,'pairs':len(records),'valid':sum(r['training_weight']>0 for r in records)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--openings',required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    from experiments.t1n_eval import parse_openings
    profile=json.loads(Path(a.profile).read_text(encoding='utf-8-sig'))
    (out/'manifest.json').write_text(json.dumps(vars(a)|{'scheduled_review':[0],'not_full_mission':True},indent=2),encoding='utf-8')
    with ProcessPoolExecutor(a.workers) as pool:
        fs=[pool.submit(job,(o,str(out),profile)) for o in parse_openings(a.openings)]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    (out/'finished.json').write_text(json.dumps({'completed':True,'time':time.time()}))
if __name__=='__main__':main()
