"""Frozen same-time factual Bellman error versus relative model signal; not alternate truth."""
import sys,json,csv
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'D:/py/DRL2')
from train.t1_b_feasible_sac import FeasibleSupportSAC
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
path=ROOT/'结果/TRAIN_e_tcpf_seed0'
model=FeasibleSupportSAC.load(path/'checkpoints/step_5000.zip',device='cpu')
rows=[];total=0;reasons={}
for line in (path/'model_prefix_releases.jsonl').read_text(encoding='utf-8').splitlines():
    r=json.loads(line);total+=1
    for reason in r['gate']['reasons']:reasons[reason]=reasons.get(reason,0)+1
    if r['gate']['weight']<=0 or 'actual_end_observation' not in r:continue
    f=r['payload']['factual'];a=r['payload']['alternative'];actual=r['actual_end_observation']
    assert f['steps']==a['steps']==r['actual_steps']
    with torch.no_grad():v=model.target_value(torch.tensor([f['observation'],a['observation'],actual],dtype=torch.float32)).flatten().numpy()
    yf=f['return']+f['discount']*(1-f['done'])*v[0]
    ya=a['return']+a['discount']*(1-a['done'])*v[1]
    yt=r['actual_return']+.999**r['actual_steps']*(not bool(r.get('actual_event')))*v[2]
    rows.append({'slot':r['slot'],'env':r['env'],'add_index':r['add_index'],'real_prefix_steps':r['actual_steps'],'weight':r['gate']['weight'],'relative_reward_signal':a['return']-f['return'],'relative_bootstrap_signal':(ya-yf)-(a['return']-f['return']),'relative_Bellman_signal':ya-yf,'factual_same_time_Bellman_error':yf-yt,'factual_reward_error':f['return']-r['actual_return'],'factual_position_error_m':r['gate']['errors']['position_m'],'interpretation':'Frozen5k critic re-computation; factual real branch only. No alternate true outcome; error vs signal magnitude is a risk diagnostic, not proof relative label sign is wrong.'})
def quant(vals):return {str(p):float(np.quantile(vals,p)) for p in [0,.25,.5,.75,.95,1]} if vals else None
summary={'scope':'Same-time factual Bellman audit using frozen5k target critic and logged true factual prefixes. No updates and no counterfactual RK45. Not historical update-time targets.','closed_prefixes':total,'valid_pairs_exported':len(rows),'invalid_reasons':reasons,'abs_relative_Bellman_quantiles':quant([abs(r['relative_Bellman_signal']) for r in rows]),'abs_factual_Bellman_error_quantiles':quant([abs(r['factual_same_time_Bellman_error']) for r in rows]),'abs_relative_reward_signal_quantiles':quant([abs(r['relative_reward_signal']) for r in rows]),'fraction_factual_abs_error_exceeds_relative_signal':float(np.mean([abs(r['factual_same_time_Bellman_error'])>abs(r['relative_Bellman_signal']) for r in rows])),'caution':'Factual common model error may cancel in paired differences; this ratio alone cannot establish wrong relative ordering or independent task benefit.'}
with (ROOT/'记录/MPC_FACTUAL_VALUE_CHECK.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'记录/MPC_VALUE_AUDIT.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
