"""T2.6W1 mechanism audit: critic and label action-value ranking against the truth, on trained checkpoints.

    python -B -m experiments.t2_rank run --checkpoint ZIP --label L --openings A-B --calibration CAL.json --workers W
                                         --output DIR [--urgent-deadline LO,HI] [--window 48,112] [--max-decisions 3]
    python -B -m experiments.t2_rank summary --input DIR [DIR ...] --output RANK.json

``run`` flies the checkpoint's deterministic policy on the urgent missions of the openings (economy missions are
skipped). At up to ``--max-decisions`` stage decisions per episode whose deadline lies ``--window`` seconds ahead,
the environment is branched before the executed stage and, after it, the K EACC candidates the learner would
receive (stage-2 policy sample from the same checkpoint, as in training) and the executed intent are each scored
four ways and flown on their own truth branch:

  critic   min_i Q_i(s, a) of the checkpoint (the ranking the policy acts on)
  label1   the stage-1 model target r1 + gamma (1 - d1) V(o1)               (production label, V of the checkpoint)
  label2   r1 + gamma (1 - d1) [r2 + gamma (1 - d2) V(o2)] where a stage-2 prediction exists, else label1
  diff     y_fact + (label1 - yhat_fact): the executed intent's real one-step target plus the model's candidate
           difference (yhat_fact the same model's raw prediction of the executed intent); the executed intent
           scores its real target under label1 / label2 / diff
  truth    the intent for one decision, then the checkpoint's deterministic policy to the end of the episode:
           the discounted shaped return sum_k gamma^k r'_k (the quantity the critic and the labels estimate)

V(o) = mean over 16 actor samples of min_i Qbar_i(o, a) - alpha log pi(a | o) (the learner's soft value). Labels use
the env's calibrations warmed by ``experiments.t2_audit warmup`` and the episode's own executed stages and entries.
Nothing here trains, and env / learner code is untouched (labels and truth come from env.t2_stage as frozen).

``summary`` reports per run: pairwise ranking accuracy on decisive pairs (truth outcome capture vs not, and truth
return apart by >= 50) and Kendall tau per decision for each scorer; label errors against the truth by label kind
(event = entry / deadline / budget end, bootstrap = stage end); the share of decisions where the truth separates the
outcomes and where each scorer ranks the best truth item first.
"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

SCORERS = ('critic', 'label1', 'label2', 'diff')


def _scorer(model):
    """Critic and soft value of a T2SAC checkpoint, numpy in, float out."""
    import torch as th
    ent = float(model.ent_coef_tensor) if getattr(model, 'ent_coef_tensor', None) is not None else float(model.ent_coef)

    def q(obs, act):
        with th.no_grad():
            qs = model.critic(th.as_tensor(np.asarray(obs, np.float32)[None]), th.as_tensor(np.asarray(act, np.float32)[None]))
        return float(th.min(th.cat(qs, dim=1)))

    def v(obs, n=16):
        with th.no_grad():
            o = th.as_tensor(np.repeat(np.asarray(obs, np.float32)[None], n, axis=0))
            a, logp = model.actor.action_log_prob(o)
            qs = th.min(th.cat(model.critic_target(o, a), dim=1), dim=1).values
            return float((qs - ent * logp.reshape(-1)).mean())
    return q, v


def _truth(snapshot, a, policy, gamma):
    """The intent for one decision, then the deterministic policy to the end: discounted shaped return and outcome."""
    from experiments.t2_audit import branch
    b = branch(snapshot)
    obs, r, term, trunc, info = b.step(np.asarray(a, dtype=np.float64))
    g, k, done = float(r), 1, term or trunc
    entered = b._stage_entry is not None
    while not done:
        obs, r, term, trunc, info = b.step(policy(obs))
        g += gamma ** k * float(r); k += 1; done = term or trunc
    return {'G': g, 'event': info.get('event'), 'decisions': k, 'entered_first': bool(entered), 'time_s': b.env.time_s}


def run_job(job):
    opening, checkpoint, label, cal, out, urgent, window, max_dec, seed = job
    import torch; torch.set_num_threads(1); torch.manual_seed(seed)
    file = Path(out) / f'{label}_{opening}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import T2StageEnv
    from experiments.t2 import mission_profile
    from experiments.t2_audit import label_policy_from, load_calibration
    from train.t2_mve_sac import K_CANDIDATES, T2SAC
    model = T2SAC.load(checkpoint, device='cpu')
    gamma = float(model.gamma)
    q, v = _scorer(model)
    policy = lambda o: model.predict(o, deterministic=True)[0]
    env = T2StageEnv(openings=[opening], candidates=K_CANDIDATES, shaping=True, mission=mission_profile(urgent))
    obs, _ = env.reset(options={'opening': opening})
    if not env.mission['urgent']:
        res = {'opening': opening, 'label': label, 'urgent': False}
        file.write_text(json.dumps(res), encoding='utf-8'); return res
    load_calibration(env, cal)
    env.label_policy = label_policy_from(checkpoint, opening)
    rows, done, t0, info = [], False, time.time(), {}
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            obs, _, term, trunc, info = env.step(np.zeros(2)); done = term or trunc; continue
        a = policy(obs)
        left = env.mission['deadline_s'] - env.env.time_s
        audit = len(rows) < max_dec and window[0] <= left <= window[1]
        snap = obs0 = None
        if audit:
            from experiments.t2_audit import branch
            snap, obs0 = branch(env), np.asarray(obs, np.float32).copy()
        obs, r, term, trunc, info = env.step(a); done = term or trunc
        m = info['model']
        if not audit or m['diag']['reason'] not in ('stage', 'entry'):
            continue
        f = m['fact']
        yhat_f = f['reward'] + (0.0 if f['done'] else gamma * v(f['next_obs']))
        y_f = float(r) + (0.0 if term else gamma * v(obs))
        items = [{'a': np.asarray(a, float).tolist(), 'executed': True, 'kind': None,
                  'critic': q(obs0, a), 'label1': y_f, 'label2': y_f, 'diff': y_f}]
        for c in m['cands']:
            l1 = c['reward'] + (0.0 if c['done'] else gamma * v(c['next_obs']))
            s2 = c.get('s2')
            l2 = l1 if (c['done'] or s2 is None) else c['reward'] + gamma * (s2['reward'] + (0.0 if s2['done'] else gamma * v(s2['next_obs'])))
            items.append({'a': np.asarray(c['a'], float).tolist(), 'executed': False, 'done': bool(c['done']), 'kind': int(c.get('kind', 0)),
                          's2_done': None if s2 is None else bool(s2['done']), 'critic': q(obs0, c['a']), 'label1': l1, 'label2': l2,
                          'diff': y_f + (l1 - yhat_f)})
        for it in items:
            it['truth'] = _truth(snap, it['a'], policy, gamma)
        rows.append({'t': float(info['time_s'] - info['stage_s']), 'left_s': float(left), 'items': items})
        print(json.dumps({'label': label, 'opening': opening, 'left': round(left), 'n': len(items), 'wall': round(time.time() - t0)}), flush=True)
    res = {'opening': opening, 'label': label, 'urgent': True, 'checkpoint': str(checkpoint), 'mission': env.mission,
           'event': info.get('event'), 'decisions': rows, 'wall_s': time.time() - t0}
    file.write_text(json.dumps(res, default=float), encoding='utf-8')
    return res


def _pairs(items, key):
    """Decisive pairs of one decision: (i, j) with item i truly better (capture vs not, or return >= 50 apart)."""
    out = []
    for i, j in combinations(range(len(items)), 2):
        ti, tj = items[i]['truth'], items[j]['truth']
        ci, cj = ti['event'] == 'success', tj['event'] == 'success'
        if ci != cj or abs(ti['G'] - tj['G']) >= 50.0:
            hi, lo = (i, j) if (ti['G'], ci) > (tj['G'], cj) else (j, i)
            out.append((hi, lo))
    return out


def _kendall(x, y):
    s = n = 0
    for i, j in combinations(range(len(x)), 2):
        a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if a != 0 and b != 0:
            s += a * b; n += 1
    return s / n if n else None


def summarize(rows: list) -> dict:
    """Ranking readings of one run's decisions (list of {'items': [...]}) -- see the module docstring."""
    out = {'decisions': len(rows), 'items': sum(len(r['items']) for r in rows)}
    sep = [r for r in rows if len(_pairs(r['items'], None)) > 0]
    out['decisions_truth_separates'] = len(sep)
    for s in SCORERS:
        right = wrong = tie = 0; taus = []; top = 0
        for r in rows:
            it = r['items']
            for hi, lo in _pairs(it, s):
                d = it[hi][s] - it[lo][s]
                right += d > 1e-9; wrong += d < -1e-9; tie += abs(d) <= 1e-9
            tau = _kendall([x[s] for x in it], [x['truth']['G'] for x in it])
            if tau is not None:
                taus.append(tau)
            if r in sep:
                best = max(x['truth']['G'] for x in it)
                top += it[int(np.argmax([x[s] for x in it]))]['truth']['G'] >= best - 1e-9
        n = right + wrong + tie
        out[s] = {'decisive_pairs': n, 'right': right, 'wrong': wrong, 'tie': tie,
                  'accuracy': (right + 0.5 * tie) / n if n else None,
                  'kendall_tau_mean': float(np.mean(taus)) if taus else None,
                  'top1_on_separating': top / len(sep) if sep else None}
    for s in ('label1', 'label2', 'diff'):        # target level against the truth (diff: label1 shifted onto the real target)
        errs = {'event': [], 'bootstrap': []}
        for r in rows:
            for x in r['items']:
                if not x['executed']:
                    errs['event' if x['done'] else 'bootstrap'].append(abs(x[s] - x['truth']['G']))
        out[f'{s}_abs_error'] = {k: {'n': len(v), 'median': float(np.median(v)) if v else None} for k, v in errs.items()}
    return out


def summary(inputs, out_path):
    runs = {}
    for d in inputs:
        for f in sorted(Path(d).glob('*.json')):
            r = json.loads(f.read_text(encoding='utf-8'))
            if r.get('urgent'):
                runs.setdefault(r['label'], []).extend(r['decisions'])
    res = {label: summarize(rows) for label, rows in sorted(runs.items())}
    Path(out_path).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    return res


def openings(text):
    a, b = text.split('-'); return list(range(int(a), int(b) + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['run', 'summary'])
    p.add_argument('--checkpoint'); p.add_argument('--label'); p.add_argument('--openings'); p.add_argument('--calibration')
    p.add_argument('--workers', type=int, default=4); p.add_argument('--output', required=True)
    p.add_argument('--urgent-deadline', default=None); p.add_argument('--window', default='48,112')
    p.add_argument('--max-decisions', type=int, default=3); p.add_argument('--input', nargs='*')
    a = p.parse_args()
    if a.mode == 'summary':
        summary(a.input, a.output); return
    urgent = tuple(float(x) for x in a.urgent_deadline.split(',')) if a.urgent_deadline else None
    window = tuple(float(x) for x in a.window.split(','))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    jobs = [(o, a.checkpoint, a.label, a.calibration, str(out), urgent, window, a.max_decisions, o) for o in openings(a.openings)]
    with ProcessPoolExecutor(a.workers) as pool:
        for f in as_completed([pool.submit(run_job, j) for j in jobs]):
            r = f.result(); print(json.dumps({'label': r['label'], 'opening': r['opening'], 'urgent': r['urgent'], 'event': r.get('event'),
                                              'decisions': len(r.get('decisions', []))}), flush=True)


if __name__ == '__main__':
    main()
