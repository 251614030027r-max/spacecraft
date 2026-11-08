"""Re-fly the executed intents of a governed screen episode (a_exec of the governor records) and log the truth every 2 s
from FROM s: mode, range, cone / mouth, closing speed, braking-envelope margin, violations at the end.
Usage: PYTHONPATH=. python -B replay_truth.py EPISODE.json FROM"""
import json, sys
import numpy as np
from env.t2_probe import ExecutorProbe, executor_gate
from env.t2_stage import T2StageEnv, relative_inertial
from experiments.t2 import mission_profile

rec = json.load(open(sys.argv[1])); t_from = float(sys.argv[2]); op = rec['opening']
env = T2StageEnv(openings=[op], shaping=False, mission=mission_profile('urgent'), obs_profile='cut_priv')
env.reset(options={'opening': op}); pr = ExecutorProbe(env)
step = env.env.step
def logged(*x, **k):
    out = step(*x, **k)
    if env.env.time_s >= t_from - 1e-9:
        q, v = relative_inertial(env.env); g = executor_gate(q, v, env.env.target_state.rotation, pr.task, pr.descent_radius_m)
        info = out[3] if isinstance(out[3], dict) else {}
        print(round(env.env.time_s, 1), env.generator.mode, round(g['range_m'], 2), g['in_cone'], g['mouth'], round(float(-(q / np.linalg.norm(q)) @ v), 3),
              round(g['radial_margin_m_s'], 3) if np.isfinite(g['radial_margin_m_s']) else 'inf', info.get('event'), info.get('violations', ''))
    return out
env.env.step = logged
for d in rec['decisions']:
    g = d.get('gov') or {}
    a = np.asarray(g.get('a_exec', d['a']), float)
    if env.generator.mode in ('bridge', 'descent'):
        break
    _, _, term, trunc, i = env.step(a)
    if term or trunc:
        break
print('replayed', i['event'], env.env.time_s, 'recorded', rec['event'], rec['time_s'])
