"""T2.8 pre-check (read-only): score fixed key states with trained checkpoints' critics and actors.

    python -B -m experiments.t2_keyscore --states DIR --checkpoints LABEL=ZIP [LABEL=ZIP ...] --output SCORES.json

``--states`` holds one JSON per audit opening written by the upper window's key-state probe (evidence
T2_8_PRECHECK): for each key stage decision the stored observation ``obs`` and the candidate intents ``cands[i].a``
(a 3 x 3 grid), each with its truth consequence. Nothing here flies physics or trains: every checkpoint is loaded once
and, at every stored state, reports min_i Q_i(obs, a) for each candidate and the deterministic actor's action. The
observation layout is unchanged since env 0.4.3, so checkpoints of T2.6W / T2.6W1 / T2.7 can score states recorded on
the current executor. Minutes on one core; no worker pool.
"""
import argparse, json
from pathlib import Path

import numpy as np


def as_profile(obs, profile):
    """T2.11: key states are stored with the engineered observation; 'reduced_priv' checkpoints read the reduced vector
    followed by the engineered one (env.t2_stage)."""
    o = np.asarray(obs, np.float32)
    if profile != 'reduced_priv' or o.shape[0] != 24:
        return o
    from env.t2_stage import PLANNER_FEATURES
    r = o.copy(); r[list(PLANNER_FEATURES)] = 0.0
    return np.concatenate([r, o])


def score(checkpoint, decisions, profile='engineered'):
    import torch as th
    th.set_num_threads(1)
    from train.t2_mve_sac import T2SAC
    model = T2SAC.load(checkpoint, device='cpu')
    out = []
    with th.no_grad():
        for d in decisions:
            ob = as_profile(d['obs'], profile); o = th.as_tensor(ob[None])
            acts = np.asarray([c['a'] for c in d['cands']], np.float32)
            oo = o.repeat(len(acts), 1)
            q = th.min(th.cat(model.critic(oo, th.as_tensor(acts)), dim=1), dim=1).values.numpy()
            a_pi = model.predict(ob, deterministic=True)[0]
            q_pi = float(th.min(th.cat(model.critic(o, th.as_tensor(np.asarray(a_pi, np.float32)[None])), dim=1)))
            out.append({'opening': d['opening'], 't': d['t'], 'q': q.tolist(), 'actor': np.asarray(a_pi).tolist(), 'q_actor': q_pi})
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--checkpoints', nargs='+', required=True, help='LABEL=ZIP')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    decisions = []
    for f in sorted(a.states.glob('*.json')):
        r = json.loads(f.read_text(encoding='utf-8'))
        for d in r.get('decisions', []):
            if d.get('obs') is not None and d.get('cands'):
                decisions.append({'opening': r['opening'], 't': d['t'], 'obs': d['obs'], 'cands': d['cands']})
    result = {'states': len(decisions), 'scores': {}}
    for item in a.checkpoints:
        label, zip_ = item.split('=', 1)
        from experiments.t2 import obs_profile_of
        try:
            profile = obs_profile_of(zip_)
        except FileNotFoundError:
            profile = 'engineered'
        result['scores'][label] = {'checkpoint': zip_, 'profile': profile, 'rows': score(zip_, decisions, profile)}
        print(label, len(decisions), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result), encoding='utf-8')


if __name__ == '__main__':
    main()
