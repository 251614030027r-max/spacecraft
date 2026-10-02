"""Small, explicitly selected replay set; recording only, no observation change."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/p2_feedback_20260910'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    candidates=[]
    for stage in ['p1_a','p1_b','p1_c']:
        folder=ROOT/f'logs/{stage}_20260910'
        if not folder.exists():continue
        for p in folder.glob('*.json'):
            if p.name in ['manifest.json','acceleration_control.json']:continue
            r=json.loads(p.read_text())['records'][0]
            candidates.append((p,r))
    candidates.sort(key=lambda pr:(pr[1]['commit_time_s_setting'], ['commit_at','hold_radius','ramp_in','lateral_adjust'].index(pr[1]['policy']),
                                 pr[1].get('hold_radius_m') or 0, abs(pr[1].get('lateral_angle_deg',0)),str(pr[0])))
    selected=[]
    for seed in [262005,262006]:
        for h in [20,35]:
            for completed in [True,False]:
                match=next(((p,r) for p,r in candidates if r['seed']==seed and r['horizon']==h and r['completed']==completed),None)
                if match:selected.append(match)
    m={'stage':'T5','status':'running','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'selection':'One success and one failure per seed/horizon when available, ordered by start time then policy/radius/angle; no selection by feedback values. Missing successful strata remain missing.',
       'workers':4,'runtime_diagnostics':False,'cache_target_trajectory':False,'timing_not_reportable':True,
       'slack_audit':'MPC reads self._slack.value after solver exception without clearing it; may be stale or zero. New info excludes every fallback step, and reports null if no valid solve.',
       'analysis_preregistered':{'episode_level_auc_separation':.75,'trend_bins_before_end_s':[[-15,-10],[-10,-5],[-5,-2]],
           'trend_gate':'>= half of failures show strictly increasing three-bin values, and a higher fraction than successes',
           'alarm_thresholds':{'fallback_fraction':.5,'slack_max':1.0,'force_peak':.95,'torque_peak':.95},
           'alarm_sustain_decisions':2,'alarm_ignore_first_s':10,'minimum_lead_s':2,
           'alarm_gate':'TPR >= 0.5, FPR <= 0.5, median lead >= 2 s; descriptive on the selected cases only'},
       'selected':[]}
    jobs=[]
    for i,(path,r) in enumerate(selected):
        name=f"case_{i}_s{r['seed']}_h{r['horizon']}_{'success' if r['completed'] else 'failure'}"
        cmd=['--seed',str(r['seed']),'--horizon',str(r['horizon']),'--policy',r['policy'],
             '--parametrization',r['waypoint_parametrization'],'--commit-time-s',str(r['commit_time_s_setting']),
             '--record-feedback','--no-diagnostics','--no-target-cache']
        if r.get('hold_radius_m') is not None:cmd+=['--hold-radius-m',str(r['hold_radius_m'])]
        if r['policy']=='ramp_in':cmd+=['--radius-step-m',str(r['radius_step_m'])]
        if r['policy']=='lateral_adjust':cmd+=['--lateral-angle-deg',str(r['lateral_angle_deg']),'--lateral-axis',str(r['lateral_axis'])]
        m['selected'].append({'artifact':name+'.json','source':str(path.relative_to(ROOT)),'seed':r['seed'],'horizon':r['horizon'],'completed':r['completed'],'command':cmd})
        jobs.append((name,cmd,r))
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    def run(job):
        name,cmd,old=job;p=OUT/(name+'.json')
        if not p.exists():
            with (OUT/(name+'.stdout.txt')).open('w',encoding='utf-8') as f:
                subprocess.run([sys.executable,'-B','-m','experiments.evaluate_hybrid_scripted',*cmd,'--output',str(p)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        new=json.loads(p.read_text())['records'][0]
        assert old['completed']==new['completed'] and old['final_time_s']==new['final_time_s'],name
        assert np.isclose(old['force_impulse_n_s'],new['force_impulse_n_s'],rtol=0,atol=1e-7),name
        assert all(x['slack_max'] is None if x['valid_solve_steps']==0 else x['slack_max'] is not None for x in new['feedback_trace'])
        print(name+' replay verified',flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run,j) for j in jobs]
            for f in as_completed(futures):f.result()
        m['status']='replays_complete_pending_analysis'
    except Exception as exc:m.update(status='failed',error=str(exc));raise
    finally:(OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')

if __name__=='__main__':main()
