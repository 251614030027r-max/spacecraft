"""Frozen exact candidate probabilities/Q. Uses saved initial observations; no physics or updates."""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from train.t1_b_feasible_sac import FeasibleSupportSAC
from env.t1_b_feasible_v1 import choice_index
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
rows=[];summary={}
for arm in ['e_sac','e_tcpf']:
    for mark in [0,2000,5000,10000]:
        file=ROOT/f'结果/TRAIN_{arm}_seed0/checkpoints/step_{mark}.zip' if mark<10000 else ROOT/f'结果/CONTINUE_{arm}_seed0_10k/checkpoints/step_{mark}.zip'
        if not file.exists():continue
        model=FeasibleSupportSAC.load(file,device='cpu');probs=[];indices=[]
        for op in range(292140,292148):
            case=json.loads((ROOT/f'结果/interface_0k/interface_0k_{op}.json').read_text(encoding='utf-8'))
            obs=torch.tensor([case['initial_observation']],dtype=torch.float32)
            with torch.no_grad():
                mean,logstd,_=model.actor.get_action_dist_params(obs)
                p=model.support_probabilities(obs)[0].numpy();q1,q2=[q[0].numpy() for q in model.candidate_q(obs,model.critic)]
            raw=float(torch.tanh(mean)[0,0]);n=int(sum(np.array(case['initial_observation'][49:52])>.5));idx=choice_index(raw,n)
            indices.append(idx);probs.append(float(p[0]))
            for j in range(n):
                rows.append({'arm':arm,'checkpoint':mark,'opening':op,'raw_deterministic':raw,'Gaussian_mean':float(mean[0,0]),'Gaussian_std':float(logstd.exp()[0,0]),'selected_index':idx,'candidate_index':j,'tau_s':case['initial_observation'][46+j]*200,'probability':float(p[j]),'Q1':float(q1[j]),'Q2':float(q2[j]),'minQ':float(min(q1[j],q2[j])),'model_sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        summary[f'{arm}_{mark}']={'first_indices':indices,'mean_probability_earliest':float(np.mean(probs)),'min_probability_earliest':min(probs),'max_probability_earliest':max(probs)}
with (ROOT/'记录/POLICY_CANDIDATE_SCAN.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'记录/POLICY_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
