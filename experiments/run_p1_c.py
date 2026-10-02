from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'logs/p1_c_20260910'

def main():
    b=json.loads((ROOT/'logs/p1_b_20260910/manifest.json').read_text());assert b['next_stage']=='T4'
    OUT.mkdir(parents=True,exist_ok=True)
    jobs={}
    for seed in b['unrescued']:
        for h in [20,35]:
            for axis,angle in [(0,0)]+[(axis,angle) for axis in [0,1] for angle in [-30,-20,-10,10,20,30]]:
                name=f'lateral_s{seed}_h{h}_axis{axis}_angle{angle}'
                jobs[name]=['--seed',str(seed),'--horizon',str(h),'--policy','lateral_adjust','--parametrization','radial_local',
                            '--lateral-axis',str(axis),'--lateral-angle-deg',str(angle),'--no-diagnostics','--no-target-cache']
    m={'stage':'T4','status':'running','parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'seeds':b['unrescued'],'horizons':[20,35],'angles_deg':[-30,-20,-10,0,10,20,30],'axes':[0,1],
       'workers':4,'timing_not_reportable':True,'commands':jobs,
       'policy':'At each decision rotate the baseline radial_local desired-pose waypoint about two orthogonal tangent directions; preserve reference radius. Added angle fades with goal distance/goal radius to keep the original final goal.',
       'causal_control':'Zero angle controls the change from the previous absolute parameterization. A zero-angle rescue must not be attributed to lateral adjustment.',
       'bounds':'Angles are requested before existing radial_local action clipping; no interface limits changed. Finite scripted diagnostic, not a claim to exhaust lateral policies.'}
    (OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    def run(job):
        name,cmd=job;path=OUT/(name+'.json')
        if path.exists():return
        with (OUT/(name+'.stdout.txt')).open('w',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-B','-m','experiments.evaluate_hybrid_scripted',*cmd,'--output',str(path)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        r=json.loads(path.read_text())['records'][0];print(f"{name}: completed={r['completed']} t={r['final_time_s']:.1f}",flush=True)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run,job) for job in jobs.items()]
            for f in as_completed(futures):f.result()
        m['status']='runs_complete_pending_analysis'
    except Exception as exc:m.update(status='failed',error=str(exc));raise
    finally:(OUT/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')

if __name__=='__main__':main()
