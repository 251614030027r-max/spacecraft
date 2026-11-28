"""294334 (ECC_s10, RUN_ECC unsafe): re-fly the recorded executed intents and log the truth every 2 s from 40 s on --
range, mode, cone / mouth region, inward closing speed and the executor's braking envelope margin, and the unsafe
reason. Usage: PYTHONPATH=. python -B replay_294334_truth.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
from env.t2_stage import T2Mission, T2StageEnv, relative_inertial
from env.t2_probe import ExecutorProbe, executor_gate

rec = json.loads(Path('evidence/ECC_DEV/C/ECC_s10/ECC_s10_294334.json').read_text())
env = T2StageEnv(openings=[294334], obs_profile='cut_priv', mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), shaping=False)
env.reset(options={'opening': 294334})
pr = ExecutorProbe(env); rows = []
step = env.env.step
def logged(*x, **k):
    out = step(*x, **k)
    if env.env.time_s >= 40.0 - 1e-9:
        q, v = relative_inertial(env.env); g = executor_gate(q, v, env.env.target_state.rotation, pr.task, pr.descent_radius_m)
        m = env.env.metrics()
        rows.append({'t': round(env.env.time_s, 1), 'mode': env.generator.mode, 'range': g['range_m'], 'in_cone': g['in_cone'], 'mouth': g['mouth'],
                     'closing': float(-(q / np.linalg.norm(q)) @ v), 'envelope_margin': g['radial_margin_m_s'],
                     'metrics_envelope_margin': float(m.radial_envelope_margin_m_s), 'event': out[3].get('event') if isinstance(out[3], dict) else None,
                     'info': {k: out[3].get(k) for k in ("violations",) if isinstance(out[3], dict) and k in out[3]}})
    return out
env.env.step = logged
for d in rec['decisions']:
    c = d.get('cert') or {}
    a = np.asarray(c['a_exec'] if c.get('intervened') else d['a'], float)
    _, _, term, trunc, i = env.step(a)
    if term or trunc:
        break
Path(sys.argv[1]).write_text(json.dumps({'event': i['event'], 'rows': rows}, indent=1, default=str))
for r in rows: print(r['t'], r['mode'], round(r['range'], 2), r['in_cone'], r['mouth'], round(r['closing'], 3), round(r['envelope_margin'], 3) if np.isfinite(r['envelope_margin']) else 'inf', r['event'], r['info'])
