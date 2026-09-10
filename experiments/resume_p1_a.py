import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]; out=root/'logs/p1_a_20260910'
m=json.loads((out/'manifest.json').read_text())
existing={p.name:json.loads(p.read_text())['records'][0] for p in out.glob('*.json') if p.name!='manifest.json'}
m['resume']={'completed_artifacts_at_resume':len(existing),'previous_processes_absent':True,'workers':4,'exact_prefix_reuse':{}}
pending=[]
for name,cmd in m['commands'].items():
    if name+'.json' in existing: continue
    def flag(k,default=None): return cmd[cmd.index(k)+1] if k in cmd else default
    seed=int(flag('--seed')); h=int(flag('--horizon')); t=float(flag('--commit-time-s'))
    radius=float(flag('--hold-radius-m')) if '--hold-radius-m' in cmd else None
    match=next((fname for fname,r in existing.items() if r['seed']==seed and r['horizon']==h
                and r['policy']==flag('--policy') and r['hold_radius_m']==radius
                and r['waypoint_parametrization']==flag('--parametrization')
                and r['commit_time_s'] is None and r['final_time_s']<=r['commit_time_s_setting']<=t),None)
    if match:
        m['resume']['exact_prefix_reuse'][name]=match
        for cell in m['cells']:
            if cell['artifact']==name+'.json':
                cell['artifact']=match
                cell['reuse_reason']='identical hold; source episode terminated before its earlier commit time'
        print(f'reuse {name} <- {match}',flush=True)
    else: pending.append((name,cmd))
def run(job):
    name,cmd=job
    with (out/(name+'.resume.stdout.txt')).open('w',encoding='utf-8') as f:
        subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(name+' complete',flush=True)
m['status']='resuming'; (out/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
try:
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(run,pending))
    m['status']='runs_complete_pending_analysis'
finally: (out/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
