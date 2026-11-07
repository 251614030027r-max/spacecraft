"""Frozen deterministic V4 evaluation with an unused tiny Replay allocation.
The loaded actor/critics and actual environment are unchanged; no updates occur.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from experiments.t1b_v2 import mission,dump
from env.t1_b_feasible_v4 import FeasibleArrivalEnvV4
from train.t1_b_v4_sac import FeasibleSupportSACV4
def checked_eval(job):
    opening,checkpoint,label,out,interface=job;assert interface=='v4';torch.set_num_threads(1);file=Path(out)/f'{label}_{opening}.json'
    if file.exists():return json.loads(file.read_text(encoding='utf-8'))
    env=FeasibleArrivalEnvV4(openings=[opening],mission=mission());obs,reset=env.reset()
    model=FeasibleSupportSACV4.load(checkpoint,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4,'buffer_size':4})
    initial=obs.tolist();done=False;k=0;disc=0.;first=None;trace=[]
    while not done:
        before=obs.copy();a=model.predict(obs,deterministic=True)[0];obs,r,t,x,i=env.step(a);done=t or x
        if first is None and i['support_transition']['learning_decision']:first={'tau_candidates':i['support_transition']['tau_candidates'],'index':i['support_transition']['candidate_index']}
        trace.append({'observation_before':before.tolist(),'observation_after':obs.tolist(),'raw_action':np.asarray(a).tolist(),'reward':float(r),'event':i['event'],'support':i['support_transition'],'review':i['review'],'execution':i['execution']});disc+=.999**k*r;k+=1
    res={'opening':opening,'label':label,'interface':'v4','checkpoint':str(checkpoint),'mission':reset['mission'],'event':i['event'],'discounted_return':disc,'time_s':env.env.time_s,'delta_v':env.env._delta_v,'first_decision':first,'initial_observation':initial}
    if res['event']=='numerical_failure':raise RuntimeError('True numerical_failure in frozen evaluation')
    with (Path(out)/f'{label}_{opening}.trace.jsonl').open('w',encoding='utf-8') as f:
        for row in trace:f.write(json.dumps(row)+'\n')
    dump(file,res);return res
