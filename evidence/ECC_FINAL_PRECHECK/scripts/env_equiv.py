"""Environment-level equivalence of the earlier paths (EACH / reduced_priv, EACH_C / cut_priv, SAC_C view, the old
certification on cut_ecc) between a reference checkout and the working tree: fixed random action sequences, natural
mission U(140, 200), shaping on; per step the observation hash, shaped and true reward, event, mode, time, certification
record. Usage: python -B env_equiv.py ROOT OUT.json (run once per checkout, then compare the JSONs)."""
import hashlib, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from env.t2_stage import T2Mission, T2StageEnv

out = {}
for profile, certify in (('reduced_priv', False), ('cut_priv', False), ('cut_ecc', True)):
    for op in (294020, 294031):
        env = T2StageEnv(openings=[op], obs_profile=profile, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), shaping=True)
        env.certify = certify
        o, _ = env.reset(options={'opening': op}); rng = np.random.default_rng(op); rows = [hashlib.sha256(o.tobytes()).hexdigest()]
        for k in range(6):
            o, r, term, trunc, i = env.step(rng.uniform(-1, 1, size=2))
            rows.append([hashlib.sha256(o.tobytes()).hexdigest(), float(r), float(i['true_reward']), i['event'], i['mode'], float(i['time_s']),
                         json.dumps(i.get('cert'), sort_keys=True)])
            if term or trunc:
                break
        out[f'{profile}_{op}'] = rows
json.dump(out, open(sys.argv[2], 'w'), indent=1)
