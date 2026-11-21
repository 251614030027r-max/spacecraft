"""Round-2 executor equivalence (docs/final_method/ROUND2.md section 2):
    python -B env_equiv.py ROOT OUT.json OPENING [OPENING ...]
Runs the checkout at ROOT. The first-draft code (science 72a8d41: ``certify`` + ``long_entry``, forced urgent mission,
48-value observation) and the round-2 code (``coordination`` with correction and long entry, ``potential='lead24'``,
17-value observation) are driven with the same intent sequences; every row must be identical between the two outputs.
Per opening two sequences: uniform intents, and slow intents (a_p in [-1, -0.2]) that reach the last pass."""
import hashlib, inspect, json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from env.t2_stage import T2StageEnv

OLD = 'certify' in inspect.signature(T2StageEnv.__init__).parameters
KEEP = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 18, 20, 21, 22, 23)


def make(op):
    if OLD:
        from env.t2_stage import mission_from_name
        env = T2StageEnv(openings=[op], mission=mission_from_name('urgent'), shaping=True, certify=True)
        env.long_entry = True
    else:
        env = T2StageEnv(openings=[op], shaping=True, coordination=True, potential='lead24')
    env.audit_logging = True
    return env


def obs_of(o):
    o = np.asarray(o, np.float32)
    return o[24:][list(KEEP)] if OLD else o


def coord_of(i):
    c = i.get('cert') if OLD else i.get('coord')
    if c is None:
        return None
    if OLD:
        kind = 'launch' if c.get('long_entry') else ('correct' if c.get('intervened') else ('keep' if c['own_event'] else 'none'))
    else:
        kind = c['kind']
    return {'a': c['a'], 'own_event': c['own_event'], 'own_t': c['own_t'], 'kind': kind, 'a_exec': c.get('a_exec'),
            'alt_t': c.get('alt_t'), 'cands': [{k: d[k] for k in ('b', 'event', 'kind', 't', 'dv', 'dv_entry')} for d in c.get('cands', [])]}


def h(o):
    return hashlib.sha256(obs_of(o).tobytes()).hexdigest()


out = {}
for op in [int(x) for x in sys.argv[3:]]:
    for mode in ('uniform', 'slow'):
        env = make(op)
        o, _ = env.reset(options={'opening': op}); rng = np.random.default_rng([op, 7 if mode == 'slow' else 3])
        rows = [h(o), repr(env.mission['deadline_s'])]
        for k in range(40):
            a = rng.uniform(-1, 1, size=2)
            if mode == 'slow':
                a[0] = -1.0 + 0.8 * (a[0] + 1.0) / 2.0
            o, r, term, trunc, i = env.step(a)
            rows.append([h(o), repr(float(r)), repr(float(i['true_reward'])), i['event'], i['mode'], repr(float(i['time_s'])),
                         repr(float(i['delta_v'])), i['held'], i['plan_status'], repr(i['planned_dv']),
                         json.dumps(coord_of(i), sort_keys=True),
                         json.dumps({k2: v for k2, v in (i.get('decision_audit') or {}).items()}, sort_keys=True),
                         json.dumps({k2: i.get(k2) for k2 in ('min_margins', 'terminal')}, sort_keys=True)])
            if term or trunc:
                break
        st = dict(env.stats)
        if OLD and 'cert_interventions' in st:
            st['corrections'] = st.pop('cert_interventions')
        rows.append(json.dumps({k2: st.get(k2) for k2 in ('stages', 'holds', 'rejected', 'bridge_attempts', 'macro', 'rate_limited',
                                                           'entries_lost', 'corrections', 'long_entries')}, sort_keys=True))
        rows.append(json.dumps(env.generator.bridge.recovery_log))
        out[f'{op}_{mode}'] = rows
        print(op, mode, rows[-2], flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
