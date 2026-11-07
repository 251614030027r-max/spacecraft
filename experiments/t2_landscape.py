"""Critic action landscapes along each policy's own deterministic trajectories (read-only diagnosis; no training).

    python -B -m experiments.t2_landscape run --runs LABEL=ZIP ... --openings 294011,294019,294035,294045 --urgent-deadline 140,200 --workers W --output DIR
    python -B -m experiments.t2_landscape summary --input DIR --output J

Question (T2.12 / T2.14, 2026-11-04): the arms with counterfactual targets time out on openings where the
privileged-critic ablation enters within four stages; their policies close more slowly and use smaller along-sweep
speeds. Is the counterfactual feedback distorting the critic's action preferences (critic side), or does the critic
still prefer the decisive intents while the actor does not follow (actor side)?

``run``: every LABEL's checkpoint flies each opening deterministically, as ``experiments.t2 eval`` does. At every stage
decision the observation is scored by *every* listed checkpoint's critic, min_i Q_i(obs, a), on a 9 x 9 intent grid (and
at its own actor's intent). ``summary``: per policy and scorer, the grid intent the critic prefers, the critic's value
gap between its preferred grid intent and the policy's own intent (in the scorer's TD-noise-free units), and how often
the critic prefers a decisive intent (|a_p| >= 0.75 and |a_z| >= 0.75) where the policy did not choose one.
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GRID = np.array([(p, z) for p in np.linspace(-1, 1, 9) for z in np.linspace(-1, 1, 9)], dtype=np.float32)


def _job(args):
    label, zip_, scorers, opening, urgent, out = args
    import torch as th
    th.set_num_threads(1)
    file = Path(out) / f'{label}_{opening}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    from experiments.t2 import mission_profile, obs_profile_of
    from train.t2_mve_sac import T2SAC
    model = T2SAC.load(zip_, device='cpu')
    critics = {k: T2SAC.load(z, device='cpu') for k, z in scorers.items()}
    env = T2StageEnv(openings=[opening], shaping=False, window=True, mission=mission_profile(urgent), obs_profile=obs_profile_of(zip_))
    obs, _ = env.reset(options={'opening': opening})
    rows, done, info = [], False, {}
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            obs, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc; continue
        a = model.predict(obs, deterministic=True)[0]
        rec = {'t': env.env.time_s, 'a': np.asarray(a).tolist(), 'q': {}}
        o = th.as_tensor(np.repeat(np.asarray(obs, np.float32)[None], len(GRID) + 1, axis=0))
        acts = th.as_tensor(np.vstack([GRID, np.asarray(a, np.float32)[None]]))
        with th.no_grad():
            for k, m in critics.items():
                q = th.min(th.cat(m.critic(o, acts), dim=1), dim=1).values.numpy()
                rec['q'][k] = q.round(3).tolist()
        obs, _, term, trunc, info = env.step(a); done = term or trunc
        rec['range_m'] = info.get('range_m'); rec['event'] = info.get('event')
        rows.append(rec)
    res = {'label': label, 'opening': opening, 'event': info.get('event'), 'time_s': info.get('time_s'), 'urgent': env.mission['urgent'], 'decisions': rows}
    file.write_text(json.dumps(res), encoding='utf-8')
    return res


def run(runs, openings, urgent, workers, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = [(lab, z, runs, op, urgent, out) for lab, z in runs.items() for op in openings]
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(_job, jobs):
            print(json.dumps({'label': r['label'], 'opening': r['opening'], 'event': r['event'], 'decisions': len(r['decisions'])}), flush=True)


def summary(inp):
    files = sorted(Path(inp).glob('*.json'))
    out = {}
    decisive = (np.abs(GRID[:, 0]) >= 0.75) & (np.abs(GRID[:, 1]) >= 0.75)
    for f in files:
        r = json.loads(f.read_text(encoding='utf-8'))
        for d in r['decisions']:
            a = np.asarray(d['a'])
            for scorer, q in d['q'].items():
                q = np.asarray(q); qg, qa = q[:-1], q[-1]
                best = int(np.argmax(qg))
                key = f"{r['label']} <- {scorer}"
                s = out.setdefault(key, {'decisions': 0, 'gap': [], 'pref_decisive': 0, 'own_decisive': 0, 'pref_decisive_not_taken': 0,
                                         'pref_a': [], 'events': {}})
                s['decisions'] += 1; s['gap'].append(float(qg[best] - qa)); s['pref_a'].append(GRID[best].tolist())
                own_dec = abs(a[0]) >= 0.75 and abs(a[1]) >= 0.75
                s['pref_decisive'] += int(decisive[best]); s['own_decisive'] += int(own_dec)
                s['pref_decisive_not_taken'] += int(decisive[best] and not own_dec)
        own = out.get(f"{r['label']} <- {r['label']}")
        if own is not None:
            own['events'][str(r['opening'])] = r['event']
    for k, s in out.items():
        g = np.asarray(s.pop('gap')); pa = np.asarray(s.pop('pref_a'))
        s.update(gap_median=float(np.median(g)), gap_mean=float(np.mean(g)), pref_a_mean=pa.mean(0).round(2).tolist(),
                 pref_abs_az_mean=float(np.abs(pa[:, 1]).mean()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('run'); r.add_argument('--runs', nargs='+', required=True); r.add_argument('--openings', required=True)
    r.add_argument('--urgent-deadline', default='140,200'); r.add_argument('--workers', type=int, default=4); r.add_argument('--output', required=True)
    q = s.add_parser('summary'); q.add_argument('--input', required=True); q.add_argument('--output', required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        runs = dict(x.split('=', 1) for x in a.runs)
        run(runs, [int(x) for x in a.openings.split(',')], tuple(float(x) for x in a.urgent_deadline.split(',')), a.workers, a.output)
    else:
        res = summary(a.input); Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
