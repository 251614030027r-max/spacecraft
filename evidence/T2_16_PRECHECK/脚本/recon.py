import sys; sys.path.insert(0,'/home/user/spacecraft')
import numpy as np
from env.t2_stage import T2StageEnv, T2Mission
rows=[]
for op in range(1000000,1000016):
    env=T2StageEnv(openings=[op], mission=T2Mission(urgent_deadline_s=(140.,200.))); o,_=env.reset(options={'opening':op}); rng=np.random.default_rng(op); done=False
    while not done:
        rows.append(o.copy())
        o,_,t,tr,_=env.step(rng.uniform(-1,1,2) if env.generator.mode not in ('bridge','descent') else np.zeros(2)); done=t or tr
X=np.array(rows)
ph=np.mod(np.arctan2(X[:,6],X[:,7]),2*np.pi); rate=X[:,9]*np.radians(2.7)
ttp=ph/rate/150.0
print('n',len(X),'time_to_pass max|err|',np.abs(ttp-X[:,8]).max())
slack=np.clip((X[:,10]*300-ph/rate-60)/100,-3,3)
print('deadline slack max|err|',np.abs(slack-X[:,19]).max())
