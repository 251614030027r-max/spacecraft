"""Fixed T1 grid; separate serial timing from concurrent reachability runs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'logs/p0_horizons_20260910'

def run(command, name):
    with (OUT / (name + '.stdout.txt')).open('w', encoding='utf-8') as stdout:
        subprocess.run([sys.executable, '-B', '-m', *command], cwd=ROOT,
                       stdout=stdout, stderr=subprocess.STDOUT, check=True)
    print(name + ' complete', flush=True)

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {
        'stage': 'T1', 'status': 'running', 'baseline': '3e5694c',
        'code_parent': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'task': 'precapture_planning', 'reference': 'fixed', 'perception': None,
        'state_source': 'truth', 'linearization': 'analytic_local',
        'seeds': list(range(262000,262012)), 'new_horizons': [30,35],
        'reused_horizons': [20,50], 'reused_directory': 'logs/upper_na_f1',
        'timing': {'serial_single_process': True, 'seed':262000, 'steps':300,
                   'horizons':[20,30,35,50], 'runtime_diagnostics':False,
                   'includes_cold_first_solve':True},
        'reachability_workers':3, 'reachability_timing_not_reportable':True,
        'additional_replays': 'h20/h50 seeds 262005/262006: fill missing truth and actuator metrics; preserve originals',
        'gate': '|F| >= 2 continue; |F| <= 1 stop and report',
        'commands': [],
    }
    for h in [20,30,35,50]:
        name=f'compute_h{h}'
        cmd=['experiments.profile_precapture_mpc','--horizon',str(h),
             '--seed','262000','--steps','300','--no-diagnostics',
             '--output',str(OUT/(name+'.json'))]
        manifest['commands'].append(cmd)
    jobs=[]
    for h,seeds in [(30,range(262000,262012)),(35,range(262000,262012)),
                    (20,[262005,262006]),(50,[262005,262006])]:
        for seed in seeds:
            name=f'fixed_h{h}_{seed}'
            cmd=['experiments.diagnose_precapture_reference','--horizon',str(h),
                 '--seed',str(seed),'--guidance','fixed','--output',str(OUT/(name+'.json'))]
            jobs.append((cmd,name))
            manifest['commands'].append(cmd)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    try:
        for h,cmd in zip([20,30,35,50],manifest['commands'][:4]):
            run(cmd,f'compute_h{h}')
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(run,*job) for job in jobs]
            for future in as_completed(futures):
                future.result()
        manifest['status']='runs_complete_pending_analysis'
    except Exception as exc:
        manifest['status']='failed'
        manifest['error']=str(exc)
        raise
    finally:
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
