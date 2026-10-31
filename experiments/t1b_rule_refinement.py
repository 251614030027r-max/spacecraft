"""Frozen public-context rule refinement; fresh confirmation only after selection."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from unittest.mock import patch

def run(job):
    from experiments.t1b_evaluate import run as evaluate
    from experiments.t1b_diagnostic import policy_action as original
    def fixed(env,row,obs,model=None):
        if row=='deadline_rule_110':return original(env,'earliest' if env.mission['urgent'] else 'tau110',obs,model)
        if row=='deadline_rule':return original(env,'earliest' if env.mission['urgent'] else 'tau130',obs,model)
        return original(env,row,obs,model)
    with patch('experiments.t1b_diagnostic.policy_action',side_effect=fixed):return evaluate(job)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--profile',required=True)
    p.add_argument('--openings',required=True);p.add_argument('--rows',required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    from experiments.t1n_eval import parse_openings
    openings=parse_openings(a.openings)
    if any(300000<=o<1000000 for o in openings):raise ValueError('Formal block sealed')
    profile=json.loads(Path(a.profile).read_text(encoding='utf-8-sig'))
    jobs=[(r,o,str(out),profile,0,o==openings[0] and r=='zero') for r in a.rows.split(',') for o in openings]
    (out/'manifest.json').write_text(json.dumps({'args':vars(a),'jobs':jobs,
        'fixed_rule':'urgent earliest; economy tau110; keep live plan',
        'selection_data':'292020-25 and 292100-107; they are not independent confirmation of this refinement'},indent=2),encoding='utf-8')
    with ProcessPoolExecutor(a.workers) as pool:
        fs=[pool.submit(run,j) for j in jobs]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    (out/'finished.json').write_text(json.dumps({'completed':True,'time':time.time()}))
if __name__=='__main__':main()
