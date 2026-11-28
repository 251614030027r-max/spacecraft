"""Env-level equivalence (final pretest precheck): fixed openings, fixed random action sequences, reduced_priv, shaping on,
urgent U(140, 200). Run from a checkout with PYTHONPATH set to it: python -B equivalence_env.py OUT.json"""
import sys, json, hashlib
import numpy as np
from env.t2_stage import T2StageEnv, T2Mission
out = []
for opening in (1000001, 1000002, 1000003, 1000005):
    env = T2StageEnv(openings=[opening], shaping=True, obs_profile='reduced_priv', mission=T2Mission(urgent_deadline_s=(140.0, 200.0)))
    obs, _ = env.reset(options={'opening': opening})
    rng = np.random.default_rng(opening); rec = {'opening': opening, 'steps': []}; done = False; k = 0
    while not done and k < 40:
        a = rng.uniform(-1, 1, 2)
        obs, r, term, trunc, info = env.step(a); done = term or trunc; k += 1
        rec['steps'].append({'obs': hashlib.sha256(np.asarray(obs, np.float32).tobytes()).hexdigest()[:16], 'r': round(float(r), 9),
                             'true': round(float(info['true_reward']), 9), 'event': info['event'], 'mode': info['mode'], 't': round(float(info['time_s']), 6)})
    out.append(rec)
json.dump(out, open(sys.argv[1], 'w'), indent=0)
print('done', [len(r['steps']) for r in out])
