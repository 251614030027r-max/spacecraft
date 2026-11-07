"""Make raw review tables from immutable logs; no model loads or updates."""
import json,csv,collections,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(name,x):(ROOT/'记录'/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(name,rr):
    with (ROOT/'记录'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
    coverage={};bucket_rows=[];qrows=[]
    for arm in ('e_sac','e_tcpf'):
        path=ROOT/f'TRAIN_{arm}_s0';p=path/'decisions.jsonl';data=[];incomplete=0
        with p.open('rb') as f:
            for line in f:
                if not line.endswith(b'\n'):incomplete+=1;continue
                data.append(json.loads(line))
        decisions=[r for r in data if r.get('support',{}).get('learning_decision')]
        buckets=collections.Counter()
        for r in decisions:
            if r['step']<=5000:
                b=(r['step']-1)//1000;j=r['support']['candidate_index'];urgent=bool(r['observation_before'][45]>.5)
                buckets[(b,j,urgent)]+=1
        for (b,j,u),n in sorted(buckets.items()):bucket_rows.append({'arm':arm,'seed':0,'step_from':1000*b+1,'step_to':1000*(b+1),'candidate_index':j,'urgent':u,'is_expert':False,'count':n})
        eligible=[r for r in decisions if 500<r['step']<=5000]
        n=sum(r['support']['candidate_index']!=0 for r in eligible)
        coverage[arm]={'logged_last_complete_step':max((r['step'] for r in data),default=None),'heartbeat_last_step':read(path/'heartbeat.json')['steps'],'log_complete_records':len(data),'incomplete_tail_records_excluded':incomplete,'post500_to5k_logged_decisions':len(eligible),'post500_to5k_logged_non_earliest':n,'non_earliest_fraction':n/len(eligible) if eligible else None,'counts_scope':'Complete persisted JSONL prefix only; host crash may lose buffered suffix. No estimate fills missing decisions.','expert_scope':'Counts above online only. Shared1000-step V3 expert Replay included separately with receipt/SHA, not counted as online.','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for mark in (2000,5000):
            a=read(ROOT/f'audit_{arm}_s0_{mark}.json');mm=a['models'][str(mark)]
            assert mm['sha256']==hashlib.sha256((path/f'checkpoints/step_{mark}.zip').read_bytes()).hexdigest()
            for s in mm['per_state']:
                for j,tau in enumerate(s['tau_candidates_s']):qrows.append({'arm':arm,'seed':0,'checkpoint':mark,'opening':s['opening'],'candidate_index':j,'tau_s':tau,'probability':s['p'][j],'Q1':s['Q1'][j],'Q2':s['Q2'][j],'Q_min':s['Q_min'][j],'selected':j==s['selected_index'],'Q_argmax':s['Q_argmax'],'entropy':s['entropy'],'Gaussian_mean':s['Gaussian_mean'],'Gaussian_std':s['Gaussian_std'],'original8_actor_gradient_norm':mm['actor_grad_norm'],'model_sha256':mm['sha256']})
    save('REPLAY_COVERAGE_5K.json',coverage);write_csv('BEHAVIOUR_COVERAGE_1K.csv',bucket_rows);write_csv('Q_POLICY_SCAN_2K_5K.csv',qrows)
    print(json.dumps({'coverage':coverage,'raw_Q_candidate_rows':len(qrows)},ensure_ascii=False))
if __name__=='__main__':main()
