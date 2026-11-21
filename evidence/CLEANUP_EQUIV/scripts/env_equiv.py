"""Environment equivalence: python -B env_equiv.py ROOT CASE OUT.json (run in both checkouts, compare)."""
import hashlib, json, sys, time, inspect
sys.path.insert(0, sys.argv[1])
import numpy as np
from env.t2_stage import T2StageEnv
try:
    from env.t2_stage import mission_from_name as mission_of
except ImportError:
    from experiments.t2 import mission_profile as mission_of
from dataclasses import replace

CASES = {  # name: (certify, geometry, window, mission, budget)
    'ecc_urgent': (True, True, True, 'urgent', None),
    'each_urgent': (False, True, True, 'urgent', None),
    'nogeo_economy': (True, False, True, 'economy', None),
    'rigid_urgent': (True, True, False, 'urgent', None),
    'ecc_natural': (True, True, True, 'natural', None),
    'ecc_resource': (True, True, True, 'economy', 2.0),
    'ecc_lowfuel': (True, True, True, 'economy', 0.3),
}
case = sys.argv[2]; certify, geometry, window, mname, budget = CASES[case]
mission = mission_of(mname)
if budget is not None:
    mission = replace(mission, budget_m_s=budget)
kw = dict(mission=mission, shaping=True, geometry=geometry, window=window)
if 'obs_profile' in inspect.signature(T2StageEnv.__init__).parameters:
    kw['obs_profile'] = 'cut_priv'
out = {}
for op in (294020, 294031):
    env = T2StageEnv(openings=[op], **kw)
    env.certify = certify; env.audit_logging = True
    o, _ = env.reset(options={'opening': op}); rng = np.random.default_rng(op)
    rows = [hashlib.sha256(o.tobytes()).hexdigest()]
    for k in range(30):
        o, r, term, trunc, i = env.step(rng.uniform(-1, 1, size=2))
        cert = i.get('cert')
        if cert is not None:
            cert = {k2: v for k2, v in cert.items() if k2 != 'wall_s'}
        rows.append([hashlib.sha256(o.tobytes()).hexdigest(), repr(float(r)), repr(float(i['true_reward'])), i['event'], i['mode'],
                     repr(float(i['time_s'])), repr(float(i['delta_v'])), i['held'], i['plan_status'], repr(i['planned_dv']),
                     json.dumps(cert, sort_keys=True), json.dumps(i.get('decision_audit'), sort_keys=True),
                     json.dumps({k2: i.get(k2) for k2 in ('min_margins', 'terminal', 'budget_m_s')}, sort_keys=True)])
        if term or trunc:
            break
    rows.append(json.dumps({k2: env.stats.get(k2) for k2 in ('stages', 'holds', 'rejected', 'bridge_attempts', 'macro', 'rate_limited',
                                                              'entries_lost', 'cert_interventions')}, sort_keys=True))
    rows.append(json.dumps(env.generator.bridge.recovery_log))
    out[f'{case}_{op}'] = rows
json.dump(out, open(sys.argv[3], 'w'), indent=1)
