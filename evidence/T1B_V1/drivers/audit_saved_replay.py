"""Numerical frozen TD/MPC targets from newly saved real Replay; no updates."""
import sys,json,pickle,hashlib,csv
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from train.t1_b_feasible_sac import FeasibleSupportSAC
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
out={'scope':'New V1 saved Replay, frozen checkpoint numerical audit. No optimizer update. Recomputed targets are NOT substituted for historical update labels.','arms':{}};rows=[]
for arm in ['e_sac','e_tcpf']:
    path=ROOT/f'结果/TRAIN_{arm}_seed0'
    if not (path/'finished.json').exists():continue
    checkpoint=path/'checkpoints/step_5000.zip';replay_file=path/'replay_latest.pkl'
    model=FeasibleSupportSAC.load(checkpoint,device='cpu')
    assert all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
    with replay_file.open('rb') as f:buffer=pickle.load(f)
    model.replay_buffer=buffer;upper=buffer.pos
    valid=buffer.cf_weights[:upper,:,0]>0
    closed=buffer.due[:upper]&buffer.interval_ready[:upper]
    loc=np.argwhere(valid&closed if arm=='e_tcpf' else closed);loc=loc[np.linspace(0,len(loc)-1,min(64,len(loc)),dtype=int)] if len(loc) else loc
    # Set exact source indices without stochastic sample replacement.
    if len(loc):
        ii,ee=loc.T;buffer._last=(ii,ee)
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        data=ReplayBufferSamples(torch.tensor(buffer.observations[ii,ee]),torch.tensor(buffer.actions[ii,ee]),torch.tensor(buffer.next_observations[ii,ee]),torch.tensor(buffer.dones[ii,ee,None]*(1-buffer.timeouts[ii,ee,None])),torch.tensor(buffer.rewards[ii,ee,None]))
        with torch.no_grad():
            target,mask=model._td_targets(data,model.target_value(data.next_observations));cf=buffer.counterfactual();delta=model._delta_target(data,cf,None);qr=torch.cat(model.critic(data.observations,data.actions),1);qa=torch.cat(model.critic(data.observations,cf.actions),1)
        for j,(i,e) in enumerate(loc):rows.append({'arm':arm,'checkpoint':5000,'slot':int(i),'env':int(e),'real_executed_coordinate':float(buffer.actions[i,e,0]),'alternative_executed_coordinate':float(buffer.cf_actions[i,e,0]),'actual_interval_steps':int(buffer.interval_steps[i,e]),'actual_interval_return':float(buffer.interval_return[i,e,0]),'actual_interval_done':float(buffer.interval_done[i,e,0]),'real_interval_TD_target':float(target[j,0]),'factual_model_prefix_return':float(buffer.cf_fact_rewards[i,e,0]),'alternative_model_prefix_return':float(buffer.cf_rewards[i,e,0]),'model_prefix_discount':float(buffer.cf_discount[i,e,0]),'MPC_relative_Bellman_target':float(delta[j,0]),'Q1_difference':float(qa[j,0]-qr[j,0]),'Q2_difference':float(qa[j,1]-qr[j,1]),'confidence_weight':float(buffer.cf_weights[i,e]),'source':'Frozen saved5k Replay/weights, re-computation, no updates'})
    if arm=='e_sac':
        for r in rows:
            if r['arm']==arm:
                for k in ['alternative_executed_coordinate','factual_model_prefix_return','alternative_model_prefix_return','model_prefix_discount','MPC_relative_Bellman_target','Q1_difference','Q2_difference','confidence_weight']:r[k]=None
    out['arms'][arm]={'saved_checkpoint':str(checkpoint),'model_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'replay_path':str(replay_file),'replay_sha256':hashlib.sha256(replay_file.read_bytes()).hexdigest(),'stored_real_transitions_including_expert':upper*4,'expert_transitions':buffer.demo_slots*4,'closed_real_task_reviews':int(closed.sum()),'model_valid_slots':int(valid.sum()),'valid_model_and_real_interval_ready':int((valid&closed).sum()),'fixed_systematic_pairs_exported':len(loc),'exact_actual_sampling_counts':buffer.sample_counts,'interval_counts':buffer.interval_counts,'feedback_stats':buffer.feedback_stats}
if rows:
    with (ROOT/'记录/FROZEN_REPLAY_TARGETS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'记录/SAVED_REPLAY_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else (z.item() if isinstance(z,np.generic) else str(z))),encoding='utf-8')
print(json.dumps({a:{k:v for k,v in d.items() if k not in ['feedback_stats']} for a,d in out['arms'].items()},ensure_ascii=False))
