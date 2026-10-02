"""Predeclared two-layer timing/radius grid; no compute claims."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/p1_a_20260910'

def main():
    p0=json.loads((ROOT/'logs/p0_horizons_20260910/manifest.json').read_text())
    assert p0['gate']=='continue_T2'
    OUT.mkdir(parents=True,exist_ok=False)
    manifest={'stage':'T2','status':'running','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'seeds':p0['persistent_failure_seeds'],'horizons':[20,35],
              'times_s':[0,30,60,90,120,150], 'radii_m':['initial',12,14,16,18],
              'parametrization':'absolute', 'workers':3,'timing_not_reportable':True,
              'diagnostics':True,'task_and_mpc_unchanged':True,
              'interpretation':'post-hoc per-seed action-space ceiling, not evidence of learnability',
              'cells':[]}
    jobs={}
    for seed in manifest['seeds']:
        for h in manifest['horizons']:
            for t in manifest['times_s']:
                control=f'commit_s{seed}_h{h}_t{t}'
                jobs[control]=['--policy','commit_at','--commit-time-s',str(t)]
                manifest['cells'].append({'layer':'timing_only','seed':seed,'horizon':h,'time_s':t,'radius_m':'initial','artifact':control+'.json'})
                for radius in manifest['radii_m']:
                    # No hold is ever executed at t=0, and radius=None is exactly
                    # the existing commit_at policy. Report all cells, reuse
                    # identical trajectories rather than computing duplicates.
                    name=control if radius=='initial' or t==0 else f'hold_s{seed}_h{h}_t{t}_r{radius}'
                    if name!=control:
                        jobs[name]=['--policy','hold_radius','--hold-radius-m',str(radius),'--commit-time-s',str(t)]
                    manifest['cells'].append({'layer':'timing_radius','seed':seed,'horizon':h,'time_s':t,'radius_m':radius,'artifact':name+'.json',
                        'identical_control_reuse':name==control})
    for name,flags in jobs.items():
        # Names encode only the fixed declared grid, not outcome selection.
        seed=int(name.split('_s')[1].split('_')[0]); h=int(name.split('_h')[1].split('_')[0])
        jobs[name]=[sys.executable,'-B','-m','experiments.evaluate_hybrid_scripted','--seed',str(seed),'--horizon',str(h),
                    '--parametrization','absolute',*flags,'--output',str(OUT/(name+'.json'))]
    manifest['commands']=jobs
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    def run(item):
        name,cmd=item
        with (OUT/(name+'.stdout.txt')).open('w',encoding='utf-8') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        r=json.loads((OUT/(name+'.json')).read_text())['records'][0]
        print(f"{name}: completed={r['completed']} t={r['final_time_s']:.1f}",flush=True)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(run,item) for item in jobs.items()]
            for f in as_completed(futures): f.result()
        manifest['status']='runs_complete_pending_analysis'
    except Exception as exc:
        manifest.update(status='failed',error=str(exc)); raise
    finally:
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': main()
