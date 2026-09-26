"""Fixed approach-rate grid; resume completed cells and keep raw evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'logs/p1_b_20260910'

def run(cmd,name):
    path=OUT/(name+'.json')
    if path.exists(): return
    with (OUT/(name+'.stdout.txt')).open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-B','-m','experiments.evaluate_hybrid_scripted',*cmd,
            '--no-diagnostics','--no-target-cache','--output',str(path)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    r=json.loads(path.read_text())['records'][0]
    print(f"{name}: completed={r['completed']} t={r['final_time_s']:.1f}",flush=True)

def main():
    a=json.loads((ROOT/'logs/p1_a_20260910/manifest.json').read_text()); assert a['next_stage']=='T3'
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for seed in a['unrescued']:
        for t in [0,60,120]:
            for h in [20,35]:
                for step in [.1,.2,.4,.8,1.6]:
                    name=f'ramp_s{seed}_h{h}_t{t}_dr{step}'
                    cmd=['--seed',str(seed),'--horizon',str(h),'--policy','ramp_in','--commit-time-s',str(t),'--radius-step-m',str(step)]
                    jobs.append((cmd,name))
    m={'stage':'T3','status':'running','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'seeds':a['unrescued'],'horizons':[20,35],'times_s':[0,60,120],'radius_steps_m':[.1,.2,.4,.8,1.6],
       'runtime_diagnostics':False,'cache_target_trajectory':False,'workers':4,'timing_not_reportable':True,
       'direction':'desired-position ray; freeze inertial hold until start, then decrease persistent reference radius from current radius',
       'boundary':'Nominal radial speed is step/2 seconds; initial direction change and existing waypoint clipping need not obey that nominal speed. Diagnostic only, not a frozen learned interface.',
       'commands':{name:cmd for cmd,name in jobs}}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    try:
        run(['--seed','262006','--horizon','20','--policy','commit_at','--commit-time-s','0'],'acceleration_control')
        old=json.loads((ROOT/'logs/p1_a_20260910/commit_s262006_h20_t0.json').read_text())['records'][0]
        new=json.loads((OUT/'acceleration_control.json').read_text())['records'][0]
        assert old['completed']==new['completed'] and old['final_time_s']==new['final_time_s']
        assert old['qp_fallbacks_total']==new['qp_fallbacks_total']
        assert np.isclose(old['force_impulse_n_s'],new['force_impulse_n_s'],rtol=0,atol=1e-8)
        m['acceleration_validation']={'same_completion_time_and_fallback_count':True,'impulse_absolute_difference':abs(old['force_impulse_n_s']-new['force_impulse_n_s'])}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run,*job) for job in jobs]
            for f in as_completed(futures): f.result()
        m['status']='runs_complete_pending_analysis'
    except Exception as exc:
        m.update(status='failed',error=str(exc)); raise
    finally: (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')

if __name__=='__main__': main()
