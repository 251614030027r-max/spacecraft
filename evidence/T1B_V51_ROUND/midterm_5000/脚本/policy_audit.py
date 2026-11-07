import json,hashlib
from pathlib import Path
import numpy as np
import torch
from train.t1_b_v5_sac import FeasibleSupportSACV5 as FeasibleSupportSACV2
from experiments.t1b_v2 import dump
ROOT=Path(__file__).resolve().parents[1];REPO=Path('D:/py/DRL2');OPS=[*range(292140,292148),*range(292120,292128)]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(arm,seed,mark):
    out=ROOT/f'audit_{arm}_s{seed}_{mark}.json'
    if out.exists():return read(out)
    z=[]
    for op in OPS:
        if op>=292140:p=next((REPO/'evidence/T1B_V1/episodes/interface_0k').glob(f'*_{op}.json'))
        else:p=REPO/f'evidence/T1B_V1/episodes/E-SAC/10000/expanded_292120_292127/e_sac_rl_10000_expanded_{op}.json'
        z.append(read(p)['initial_observation'])
    z=torch.tensor(np.array(z,np.float32));models={}
    for step in (0,mark):
        path=ROOT/f'TRAIN_{arm}_s{seed}/checkpoints/step_{step}.zip';m=FeasibleSupportSACV2.load(path,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4,'buffer_size':4})
        mean,ls,_=m.actor.get_action_dist_params(z);p=m.support_probabilities(z);q1,q2=m.candidate_q(z,m.critic);q=torch.minimum(q1,q2).detach();alpha=m.alpha_of(z)
        loss=(p[:8]*(alpha[:8]*p[:8].clamp_min(1e-8).log()-q[:8])).sum(1).mean();g=torch.autograd.grad(loss,list(m.actor.parameters()),allow_unused=True)
        norm=float(torch.sqrt(sum((x*x).sum() for x in g if x is not None)));entropy=-(p*p.clamp_min(1e-8).log()).sum(1);ss=[]
        for k,op in enumerate(OPS):
            n=int((z[k,49:52]>.5).sum());raw=float(mean[k,0].tanh().detach());j=min(n-1,max(0,int((raw+1)*.5*n)))
            ss.append({'opening':op,'p':p[k,:n].detach().tolist(),'entropy':float(entropy[k].detach()),'Gaussian_mean':float(mean[k,0].detach()),'Gaussian_std':float(ls[k,0].exp().detach()),'Q1':q1[k,:n].detach().tolist(),'Q2':q2[k,:n].detach().tolist(),'Q_min':q[k,:n].tolist(),'Q_argmax':int(q[k,:n].argmax()),'selected_index':j,'tau_candidates_s':(z[k,46:46+n]*200).tolist()})
        models[str(step)]={'sha256':sha(path),'actor_grad_norm':norm,'per_state':ss,'alpha':float(alpha.mean())}
    n=sum(s['entropy']>.05 for s in models[str(mark)]['per_state'][:8]);result={'arm':arm,'seed':seed,'mark':mark,'models':models,'original8_entropy_gt_005':n,'entropy_gate_pass':n>=4,'observation_provenance':'Same frozen V1 initial observations; before any plan, V3 support/observation unchanged. Expanded8 diagnostic only.'};dump(out,result);return result

