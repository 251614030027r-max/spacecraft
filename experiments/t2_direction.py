"""Truth direction audit of the learner 6.1 execution feedback (read-only; no training). Gate of T2.16.

    python -B -m experiments.t2_direction run --policies LABEL=ZIP ... --openings A-B --calibration CAL.json --workers W --output DIR [--branches Q,M,E,P]
    python -B -m experiments.t2_direction summary --input DIR --output J

Question: does the executor's correction move the policy's intent in a direction the truth prefers to the critic's own
action gradient? Learner 6.1 adds the slope g of the correction field to the actor's action gradient grad_a Q (clipped
to its scale); it can beat the privileged critic alone (e_priv) only if, where the two directions differ, the truth
prefers the corrected one.

``run``: each LABEL's checkpoint (an ``e_priv`` run on ``reduced_priv``) flies each opening deterministically on an
environment that labels as the ``e_pi`` arm does (K candidates, one stage, candidates separated, the policy's own intent
through the candidates' treatment; warm calibration from CAL.json, then the episode's own executed stages). At every
stage decision, before the stage is flown, the state is branched (``experiments.t2_audit.branch``). After the stage:

- the learner's own code (``T2SAC._actor_correction``, constants FB_*, feedback 'rel') on this one transition, its
  FB_BATCH rows drawn with independent noise, gives g (mean over the identified rows) and the critic gradient
  gq = grad_a min_i Q_i(o, a_pi);
- three truth branches from the decision state fly a_pi + DELTA * u for u = gq / |gq| (critic, "Q"),
  (gq + g_c) / |gq + g_c| with g_c = g clipped to FB_CLIP |gq| (the 6.1 update, "M") and g / |g| (the executor alone,
  "E"), and then the same deterministic policy to the end of the episode; the true return from the decision on is
  recorded (and a_pi's own, from the episode).

Learner 6.3 (gate A2): branch "P" flies a_pi + DELTA * u_P, u_P the direction from a_pi to the best candidate on the
fused advantage when it exceeds the TD noise (``_actor_correction`` with feedback 'pull'); computed after, and seeded
independently of, the 'rel' correction, so Q / M / E are unchanged. ``summary`` reports the same gate for P against Q
under ``pull`` (decisions where u_P turns the critic's direction, cos(u_P, u_Q) < DISAGREE_COS).

``summary`` applies the pre-registered gate (T2_16 run order) on the decisions where the 6.1 update turns the critic's
direction (cos(u_M, u_Q) < DISAGREE_COS) and both branches ended:
  D data   >= MIN_INFORMATIVE decisions with |R_M - R_Q| > TIE;
  P1 wins  share of those with R_M > R_Q >= WIN_SHARE;
  P2 mean  mean (R_M - R_Q) over the turned decisions > 0;
  S safety unsafe endings of the M branches <= of the Q branches.
"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DELTA = 0.25            # step along a unit direction in the intent square ([-1, 1]^2)
DISAGREE_COS = 0.9      # the 6.1 update turns the critic's direction by more than ~26 degrees
TIE = 1.0               # |R_M - R_Q| at or below this is a tie
MIN_INFORMATIVE = 20
WIN_SHARE = 0.60
URGENT = (140.0, 200.0)


def _unit(v):
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _fly(snap, a, policy):
    """Truth: intent ``a`` from the branched decision state, then the deterministic policy to the end."""
    from experiments.t2_audit import branch
    b = branch(snap)
    obs, r, term, trunc, info = b.step(np.asarray(a, np.float64))
    ret, done = float(info['true_reward']), term or trunc
    while not done:
        obs, r, term, trunc, info = b.step(policy(obs)); ret += float(info['true_reward']); done = term or trunc
    return {'return': ret, 'event': info.get('event'), 'time_s': info.get('time_s')}


def correction(model, obs, next_obs, a, reward, done, info, seed, feedback: str = 'rel'):
    """g (mean over identified rows), gq at a, readings: the learner's own ``_actor_correction`` on this transition
    (``feedback`` 'rel' = e_pi's slope, 'pull' = e_pull's direction to the best candidate)."""
    model.feedback = feedback
    import torch as th
    from train.t2_mve_sac import StageReplay, _slim
    model.replay_buffer = StageReplay(4, model.observation_space, model.action_space, device='cpu')
    model.replay_buffer.add(np.asarray(obs, np.float32)[None], np.asarray(next_obs, np.float32)[None], np.asarray(a, np.float32)[None],
                            np.array([reward]), np.array([done]), [_slim(info)])
    b = (info.get('model') or {}).get('diag', {}).get('entry_bias')
    model.entry_bias_now = None if b is None else float(b)
    th.manual_seed(seed); np.random.seed(seed)
    ob, sel, g, a_ref, finfo = model._actor_correction()
    a_q = th.as_tensor(np.asarray(a, np.float32)[None]).requires_grad_(True)
    o_q = th.as_tensor(np.asarray(obs, np.float32)[None])
    gq = th.autograd.grad(th.min(th.cat(model.critic(o_q, a_q), dim=1), dim=1).values.sum(), a_q)[0][0].numpy().astype(float)
    if feedback == 'pull' and sel is not None:          # gate A2: the non-local rows only (the others are e_pi's slope)
        sel = sel & model._fb_pull_rows
    if sel is None or not bool(sel.any()):
        return None, gq, finfo
    return g[sel].mean(0).numpy().astype(float), gq, finfo


def _job(args):
    label, zip_, opening, cal, out, branches = (*args, 'Q,M,E') if len(args) == 5 else args
    branches = set(branches.split(','))
    import torch as th
    th.set_num_threads(1)
    file = Path(out) / f'{label}_{opening}.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    from env.t2_stage import CANDIDATE_MIN_SEP_M_S, T2StageEnv
    from experiments.t2 import mission_profile, obs_profile_of
    from experiments.t2_audit import branch, load_calibration
    from train.t2_mve_sac import FB_CLIP, K_CANDIDATES, T2SAC
    t0 = time.time()
    model = T2SAC.load(zip_, device='cpu'); model.feedback = 'rel'
    policy = lambda o: model.predict(o, deterministic=True)[0]
    env = T2StageEnv(openings=[opening], candidates=K_CANDIDATES, shaping=True, entry_stages=1, window=True, mission=mission_profile(URGENT),
                     obs_profile=obs_profile_of(zip_), cand_min_sep=CANDIDATE_MIN_SEP_M_S)
    env.ref_labels = True
    obs, _ = env.reset(options={'opening': opening}); load_calibration(env, cal)
    rows, rewards, done, k = [], [], False, 0
    while not done:
        if env.generator.mode in ('bridge', 'descent'):
            obs, r, term, trunc, info = env.step(np.zeros(2)); rewards.append(float(info['true_reward'])); done = term or trunc; continue
        a = np.asarray(policy(obs), np.float64)
        snap = branch(env)
        env.policy_action = a
        nobs, r, term, trunc, info = env.step(a); done = term or trunc
        rec = {'k': k, 't': info.get('time_s'), 'a': a.tolist(), 'reason': (info.get('model') or {}).get('diag', {}).get('reason'),
               'reward_index': len(rewards)}
        rewards.append(float(info['true_reward']))
        g, gq, finfo = correction(model, obs, nobs, a, r, done, info, seed=opening * 100 + k)
        rec.update(gq=gq.tolist(), fb={kk: finfo.get(kk) for kk in ('fb_rows', 'fb_identified', 'fb_cands', 'fb_rho_mean', 'fb_corr_abs_mean')})
        uq = _unit(gq)
        if g is not None and uq is not None:
            gn = float(np.linalg.norm(g)); qn = float(np.linalg.norm(gq))
            g_c = g * min(1.0, FB_CLIP * qn / max(gn, 1e-9))
            um, ue = _unit(gq + g_c), _unit(g)
            rec.update(g=g.tolist(), g_norm=gn, gq_norm=qn, cos_g_gq=float(np.dot(g, gq) / max(gn * qn, 1e-12)),
                       cos_m_q=None if um is None else float(np.dot(um, uq)))
            for name, u in (('Q', uq), ('M', um), ('E', ue)):
                if u is not None and name in branches:
                    a_b = np.clip(a + DELTA * u, -1.0, 1.0)
                    rec[name] = {'a': a_b.tolist(), **_fly(snap, a_b, policy)}
        if 'P' in branches and uq is not None:               # gate A2 (learner 6.3, e_pull)
            gp, _, pinfo = correction(model, obs, nobs, a, r, done, info, seed=opening * 100 + k + 50_000, feedback='pull')
            up = None if gp is None else _unit(gp)
            rec['pull'] = {kk: pinfo.get(kk) for kk in ('fb_identified', 'fb_pull_adv_mean', 'fb_pull_dist_mean')}
            if up is not None:
                rec['cos_p_q'] = float(np.dot(up, uq))
                if 'Q' not in rec:
                    a_q = np.clip(a + DELTA * uq, -1.0, 1.0); rec['Q'] = {'a': a_q.tolist(), **_fly(snap, a_q, policy)}
                a_p = np.clip(a + DELTA * up, -1.0, 1.0); rec['P'] = {'a': a_p.tolist(), **_fly(snap, a_p, policy)}
            model.feedback = 'rel'
        rows.append(rec); obs = nobs; k += 1
    for rec in rows:
        rec['R_pi'] = float(np.sum(rewards[rec.pop('reward_index'):]))
    res = {'label': label, 'checkpoint': str(zip_), 'opening': opening, 'event': info.get('event'), 'return': float(np.sum(rewards)),
           'mission': env.mission, 'decisions': rows, 'wall_s': time.time() - t0}
    file.write_text(json.dumps(res, default=float), encoding='utf-8')
    return res


def run(policies, openings, cal, workers, out, branches='Q,M,E'):
    Path(out).mkdir(parents=True, exist_ok=True)
    jobs = [(lab, z, op, cal, out, branches) for lab, z in policies.items() for op in openings]
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(_job, j) for j in jobs]):
            r = f.result()
            n = sum(1 for d in r['decisions'] if 'M' in d or 'P' in d)
            print(json.dumps({'label': r['label'], 'opening': r['opening'], 'event': r['event'], 'decisions': len(r['decisions']),
                              'branched': n, 'wall_s': round(r['wall_s'])}), flush=True)


def summary(inp):
    files = sorted(Path(inp).glob('*.json'))
    dec, episodes = [], 0
    for f in files:
        r = json.loads(f.read_text(encoding='utf-8')); episodes += 1
        for d in r['decisions']:
            dec.append({**d, 'label': r['label'], 'opening': r['opening']})
    branched = [d for d in dec if 'M' in d and 'Q' in d]
    turned = [d for d in branched if d.get('cos_m_q') is not None and d['cos_m_q'] < DISAGREE_COS]
    diff = np.array([d['M']['return'] - d['Q']['return'] for d in turned]) if turned else np.zeros(0)
    informative = diff[np.abs(diff) > TIE]
    wins = int((informative > 0).sum())
    unsafe = lambda x: sum(1 for d in turned if d[x]['event'] == 'unsafe')

    def pair(x, y, ds):
        v = np.array([d[x]['return'] - (d['R_pi'] if y == 'pi' else d[y]['return']) for d in ds if x in d and (y == 'pi' or y in d)])
        inf = v[np.abs(v) > TIE]
        return {'n': int(len(v)), 'informative': int(len(inf)), 'win_share': float((inf > 0).mean()) if len(inf) else None,
                'mean': float(v.mean()) if len(v) else None, 'median': float(np.median(v)) if len(v) else None}
    boot = None
    if len(diff):
        rng = np.random.default_rng(2716)
        bm = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        boot = [float(np.percentile(bm, 5)), float(np.percentile(bm, 95))]
    crit = {'D_data': bool(len(informative) >= MIN_INFORMATIVE),
            'P1_wins': bool(len(informative) and wins / len(informative) >= WIN_SHARE),
            'P2_mean': bool(len(diff) and diff.mean() > 0),
            'S_safety': bool(unsafe('M') <= unsafe('Q'))}
    verdict = 'pass' if all(crit.values()) else ('insufficient' if not crit['D_data'] else 'fail')
    by_label = {}
    for lab in sorted({d['label'] for d in dec}):
        ds = [d for d in turned if d['label'] == lab]
        by_label[lab] = {'decisions': sum(1 for d in dec if d['label'] == lab), 'branched': sum(1 for d in branched if d['label'] == lab),
                         'M_vs_Q_turned': pair('M', 'Q', ds)}
    pt = [d for d in dec if 'P' in d and 'Q' in d and d.get('cos_p_q') is not None and d['cos_p_q'] < DISAGREE_COS]
    pdiff = np.array([d['P']['return'] - d['Q']['return'] for d in pt]) if pt else np.zeros(0)
    pinf = pdiff[np.abs(pdiff) > TIE]; pwins = int((pinf > 0).sum())
    punsafe = lambda x: sum(1 for d in pt if d[x]['event'] == 'unsafe')
    pcrit = {'D_data': bool(len(pinf) >= MIN_INFORMATIVE), 'P1_wins': bool(len(pinf) and pwins / len(pinf) >= WIN_SHARE),
             'P2_mean': bool(len(pdiff) and pdiff.mean() > 0), 'S_safety': bool(punsafe('P') <= punsafe('Q'))}
    pboot = None
    if len(pdiff):
        rng = np.random.default_rng(2717)
        pb = [rng.choice(pdiff, len(pdiff)).mean() for _ in range(2000)]
        pboot = [float(np.percentile(pb, 5)), float(np.percentile(pb, 95))]
    pull = {'verdict': 'pass' if all(pcrit.values()) else ('insufficient' if not pcrit['D_data'] else 'fail'), 'criteria': pcrit,
            'fired': sum(1 for d in dec if 'P' in d), 'turned': len(pt), 'informative': int(len(pinf)), 'wins': pwins,
            'win_share': pwins / len(pinf) if len(pinf) else None, 'mean_P_minus_Q_turned': float(pdiff.mean()) if len(pdiff) else None,
            'mean_P_minus_Q_turned_90ci': pboot, 'mean_P_minus_Q_without_top2': float(np.sort(pdiff)[:-2].mean()) if len(pdiff) > 2 else None,
            'unsafe_P': punsafe('P'), 'unsafe_Q': punsafe('Q'),
            'P_vs_pi_turned': pair('P', 'pi', pt), 'Q_vs_pi_turned': pair('Q', 'pi', pt)}
    return {'verdict': verdict, 'criteria': crit, 'pull': pull,
            'constants': {'delta': DELTA, 'disagree_cos': DISAGREE_COS, 'tie': TIE, 'min_informative': MIN_INFORMATIVE, 'win_share': WIN_SHARE},
            'episodes': episodes, 'decisions': len(dec), 'branched': len(branched), 'branched_share': len(branched) / max(len(dec), 1),
            'turned': len(turned), 'informative': int(len(informative)), 'wins': wins,
            'win_share': wins / len(informative) if len(informative) else None,
            'mean_M_minus_Q_turned': float(diff.mean()) if len(diff) else None, 'mean_M_minus_Q_turned_90ci': boot,
            'unsafe_M': unsafe('M') if turned else 0, 'unsafe_Q': unsafe('Q') if turned else 0,
            'reported': {'M_vs_Q_all_branched': pair('M', 'Q', branched), 'E_vs_Q_turned': pair('E', 'Q', turned),
                         'M_vs_pi_turned': pair('M', 'pi', turned), 'Q_vs_pi_turned': pair('Q', 'pi', turned),
                         'cos_g_gq_median': float(np.median([d['cos_g_gq'] for d in branched])) if branched else None,
                         'g_over_gq_median': float(np.median([d['g_norm'] / max(d['gq_norm'], 1e-9) for d in branched])) if branched else None},
            'by_label': by_label}


def plot(summary_json, inp, png):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    s = json.loads(Path(summary_json).read_text(encoding='utf-8'))
    pts = []
    for f in sorted(Path(inp).glob('*.json')):
        for d in json.loads(f.read_text(encoding='utf-8'))['decisions']:
            if 'M' in d and 'Q' in d and d.get('cos_m_q') is not None:
                pts.append((d['cos_m_q'], d['M']['return'] - d['Q']['return']))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    if pts:
        x, y = np.array(pts).T
        ax[0].scatter(x, y, s=10, alpha=0.6); ax[0].axhline(0, color='k', lw=0.8); ax[0].axvline(DISAGREE_COS, color='r', ls='--', lw=0.8)
    ax[0].set_xlabel('cos(6.1 update, critic gradient)'); ax[0].set_ylabel('true return: corrected step - critic step')
    w, n = s['wins'], s['informative']
    ax[1].bar(['corrected better', 'critic better'], [w, n - w], color=['tab:green', 'tab:gray'])
    ax[1].axhline(WIN_SHARE * n, color='r', ls='--', lw=0.8)
    ax[1].set_title(f"verdict {s['verdict']}: {w}/{n} informative turned decisions")
    fig.tight_layout(); fig.savefig(png, dpi=150)


def _ops(spec):
    if '-' in spec:
        a, b = spec.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(',')]


def main():
    p = argparse.ArgumentParser(description=__doc__); s = p.add_subparsers(dest='cmd', required=True)
    r = s.add_parser('run'); r.add_argument('--policies', nargs='+', required=True); r.add_argument('--openings', required=True)
    r.add_argument('--calibration', required=True); r.add_argument('--workers', type=int, default=4); r.add_argument('--output', required=True)
    r.add_argument('--branches', default='Q,M,E', help='truth branches: Q critic, M 6.1/6.2 corrected, E executor slope, P 6.3 pull (gate A2)')
    q = s.add_parser('summary'); q.add_argument('--input', required=True); q.add_argument('--output', required=True)
    g = s.add_parser('plot'); g.add_argument('--summary', required=True); g.add_argument('--input', required=True); g.add_argument('--output', required=True)
    a = p.parse_args()
    if a.cmd == 'run':
        run(dict(x.split('=', 1) for x in a.policies), _ops(a.openings), a.calibration, a.workers, a.output, a.branches)
    elif a.cmd == 'summary':
        res = summary(a.input); Path(a.output).write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))
    else:
        plot(a.summary, a.input, a.output)


if __name__ == '__main__':
    main()
