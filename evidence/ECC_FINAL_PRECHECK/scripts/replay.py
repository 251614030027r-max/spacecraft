"""Regression replays (294320-294367 are regression only): re-fly the executed intents recorded in RUN_ECC
(``evidence/ECC_DEV/C/<run>/<run>_<opening>.json``: a_exec where the old certification intervened, else a) on the natural
mission with the old certification off, and run the new governor in monitor mode at every decision: does the replay
reproduce the recorded outcome, and what does the governor judge at the decisions where the old certification acted?
Usage: python -B replay.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
from env.t2_stage import T2Mission, T2StageEnv

CASES = (('ECC_s10', 294323), ('ECC_s10_nocert', 294323), ('ECC_s11', 294323), ('ECC_s10', 294334), ('ECC_s10_nocert', 294334),
         ('EACH_C_s10_cert', 294338), ('EACH_C_s11_cert', 294342))
out = {}
for run, op in CASES:
    rec = json.loads(Path(f'evidence/ECC_DEV/C/{run}/{run}_{op}.json').read_text())
    env = T2StageEnv(openings=[op], obs_profile='cut_priv', mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), shaping=False)
    env.reset(options={'opening': op}); env.govern = 'monitor'
    rows, ret = [], 0.0
    for d in rec['decisions']:
        c = d.get('cert') or {}
        a = np.asarray(c['a_exec'] if c.get('intervened') else d['a'], float)
        if env.generator.mode in ('bridge', 'descent'):
            break
        _, _, term, trunc, i = env.step(a); ret += i['true_reward']
        g = i.get('gov') or {}
        rows.append({'t_start': g.get('t'), 't_end': i['time_s'], 'mode': i['mode'], 'event': i['event'], 'old_cert': c.get('intervened', False),
                     'old_alt_kind': c.get('alt_kind'), **{k: g.get(k) for k in ('status', 'critical', 'tau_min', 'own_event', 'own_kind', 'own_t',
                                                                                  'own_admissible', 'alt', 'alt_t', 'alt_kind', 'n_admissible')}})
        if term or trunc:
            break
    out[f'{run}_{op}'] = {'recorded': {k: rec[k] for k in ('event', 'return', 'time_s')}, 'replayed': {'event': i['event'], 'return': ret,
                          'time_s': env.env.time_s}, 'decisions': rows}
    print(run, op, rec['event'], '->', i['event'], round(rec['return'], 2), round(ret, 2),
          [(r['t_start'], r['status'], r['old_cert'], r.get('own_kind'), r.get('own_admissible')) for r in rows], flush=True)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
