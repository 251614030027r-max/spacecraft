import sys, json, os
sys.path.insert(0, '/home/user/spacecraft')
from pathlib import Path
from multiprocessing import get_context
OUT = Path(sys.argv[1])
def job(x):
    op, pol, cert = x
    f = OUT / f'{pol}_{"cert" if cert else "base"}_{op}.json'
    if f.exists(): return json.loads(f.read_text())
    import torch; torch.set_num_threads(1)
    from env.t2_stage import T2StageEnv
    from experiments.t2 import mission_profile, run_episode, policy_action
    env = T2StageEnv(openings=[op], shaping=False, obs_profile='reduced_priv', mission=mission_profile((140.0, 200.0)))
    env.certify = cert
    r = run_episode(env, op, lambda e, o: policy_action(pol, e, o))
    r['policy'] = pol; r['certify'] = cert; r['cert_interventions'] = env.stats.get('cert_interventions', 0)
    f.write_text(json.dumps(r, default=float)); return r
if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    urg = [int(x) for x in open(OUT.parent / 'urgent.txt').read().split(',')]
    jobs = [(op, pol, c) for op in urg for pol in ('c_10_0', 'rally') for c in (True, False)]
    jobs += [(op, 'c_10_0', True) for op in range(294000, 294048) if op not in urg]
    with get_context('spawn').Pool(4, maxtasksperchild=6) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs)):
            print(i + 1, len(jobs), r['policy'], r['certify'], r['opening'], r['event'], r['cert_interventions'], flush=True)
