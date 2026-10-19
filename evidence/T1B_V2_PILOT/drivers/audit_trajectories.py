"""B2 frozen same-observation comparison on the union of actual evaluation reviews.
No extra true rollouts, no learning. Truth ranking is available only at first states.
"""
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from train.t1_b_v2_sac import FeasibleSupportSACV2
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
def main():
    result={};all_rows=[]
    for mark in [2000,5000,10000]:
        files={a:ROOT/f'TRAIN_{a}_s0/checkpoints/step_{mark}.zip' for a in ['e_sac','e_tcpf']}
        if not all(p.exists() for p in files.values()):continue
        states=[];metadata=[]
        for arm in files:
            for trace in sorted((ROOT/f'eval_{arm}_{mark}').glob('*.trace.jsonl')):
                opening=int(trace.name.split('.')[0].rsplit('_',1)[1])
                for number,line in enumerate(trace.read_text(encoding='utf-8').splitlines()):
                    r=json.loads(line)
                    if not r['support']['learning_decision']:continue
                    states.append(r['observation_before']);metadata.append({'mark':mark,'source_arm':arm,'opening':opening,'review_trace_row':number,'source':str(trace)})
        if not states:continue
        z=torch.tensor(np.asarray(states,np.float32));data={}
        for arm,path in files.items():
            m=FeasibleSupportSACV2.load(path,device='cpu',custom_objects={'learning_rate':1e-4,'lr_schedule':lambda _:1e-4})
            with torch.no_grad():
                mean,_,_=m.actor.get_action_dist_params(z);raw=mean.tanh().flatten().numpy();p=m.support_probabilities(z).numpy();q1,q2=m.candidate_q(z,m.critic);q=torch.minimum(q1,q2).numpy()
            counts=(z[:,49:52].numpy()>.5).sum(1);selected=np.minimum(counts-1,np.maximum(0,((raw+1)*.5*counts).astype(int)))
            ranked=np.array([int(q[k,:n].argmax()) for k,n in enumerate(counts)])
            data[arm]={'selected':selected,'ranked':ranked,'probabilities':p,'q':q,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for k,meta in enumerate(metadata):
            all_rows.append({**meta,'e_sac_selected':int(data['e_sac']['selected'][k]),'e_tcpf_selected':int(data['e_tcpf']['selected'][k]),'choice_differs':bool(data['e_sac']['selected'][k]!=data['e_tcpf']['selected'][k]),'e_sac_Q_best':int(data['e_sac']['ranked'][k]),'e_tcpf_Q_best':int(data['e_tcpf']['ranked'][k]),'Q_ranking_best_differs':bool(data['e_sac']['ranked'][k]!=data['e_tcpf']['ranked'][k]),'e_sac_p':json.dumps(data['e_sac']['probabilities'][k].tolist()),'e_tcpf_p':json.dumps(data['e_tcpf']['probabilities'][k].tolist()),'e_sac_Qmin':json.dumps(data['e_sac']['q'][k].tolist()),'e_tcpf_Qmin':json.dumps(data['e_tcpf']['q'][k].tolist())})
        result[str(mark)]={'actual_review_states_in_union':len(states),'choice_difference_count':int((data['e_sac']['selected']!=data['e_tcpf']['selected']).sum()),'Q_best_difference_count':int((data['e_sac']['ranked']!=data['e_tcpf']['ranked']).sum()),'model_sha256':{a:d['sha256'] for a,d in data.items()},'population':'Union of genuine learning-review states actually visited by each deterministic evaluation arm; both frozen policies evaluated on every identical observation. Repeated states are retained and identified, not treated as iid task success samples.','truth_limit':'Only first-state candidate truth has fixed-teacher counterfactual returns; no false later-state true-Q ranking claim.'}
    (ROOT/'B2_TRAJECTORY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if all_rows:
        with (ROOT/'B2_TRAJECTORY_COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
