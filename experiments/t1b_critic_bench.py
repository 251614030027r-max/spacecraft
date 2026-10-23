"""Offline critic bench for the T1-B V6/V7 learner: no environment step, no new real outcome.

Answers, on a frozen checkpoint and its saved Replay, the two questions that otherwise cost an
online round each (research-layer review of V7, 2026-10-20):

1. **Direction**: cosines between the parameter gradients of the transition TD, the intent-batch
   real target and the MPC-paired advantage (V6 only logged norms; a norm ratio < 1 does not
   exclude opposite directions).
2. **Learnability**: starting from the checkpoint's critic, train copies offline on the stored
   intent batches with three loss sets (real intent only / real intent + MPC advantage / transition
   TD + real intent + MPC advantage = the online V6/V7 loss) and report, every ``--report-every``
   updates, the candidate-Q ranking of reference first-review states against their candidate truth
   (best-candidate agreement, pairwise sign agreement, regret of the Q-greedy choice).

The actor is frozen (soft values use its probabilities). Reference truth: the candidate scans
(``scan_<op>_<j>.json`` of ``experiments.t1b_v2 scan`` or ``r_<op>_<j>.json`` of the V1 audit);
reference states: the initial observations in evaluation directories (``initial_observation``, comma-separated).

    python -B -m experiments.t1b_critic_bench --checkpoint step_5000.zip --replay replay_latest.pkl \
        --truth evidence/T1B_V1_AUDIT/candidate_truth_scan,evidence/T1B_V2_PILOT/scan_292120 \
        --states evidence/T1B_V1/episodes/interface_0k --updates 3000 --output bench.json
"""
import argparse, copy, glob, json, os
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.utils import polyak_update

from train.t1_b_v7_sac import FeasibleSupportSACV7


def load_truth(dirs):
    truth = {}
    for d in dirs:
        for f in glob.glob(str(Path(d) / '*.json')):
            x = json.load(open(f, encoding='utf-8-sig'))
            if 'opening' not in x or x.get('skipped'):
                continue
            j = x.get('first_index', x.get('j0'))
            if j is None or 'discounted_return' not in x:
                continue
            truth.setdefault(int(x['opening']), {})[int(j)] = float(x['discounted_return'])
    return truth


def load_states(dirs):
    rows = {}
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / '*.json'))):
            x = json.load(open(f, encoding='utf-8-sig'))
            if 'initial_observation' in x and int(x['opening']) not in rows:
                rows[int(x['opening'])] = np.asarray(x['initial_observation'], np.float32)
    return sorted(rows.items())


def ranking(model, states, truth):
    agree = pairs = pair_ok = n = 0; regret = []; steps = []
    with th.no_grad():
        for op, z in states:
            t = truth.get(op); k = int((z[49:52] > .5).sum())
            if not t or k < 2 or len(t) < k:
                continue
            q = th.minimum(*model.candidate_q(th.tensor(z[None]), model.critic))[0, :k].numpy()
            tv = np.array([t[j] for j in range(k)]); n += 1; steps.append(float(np.mean(np.abs(np.diff(q)))))
            best = int(np.argmax(q)); agree += int(tv[best] >= tv.max() - 1.0); regret.append(float(tv.max() - tv[best]))
            for i in range(k):
                for j in range(i + 1, k):
                    if abs(tv[i] - tv[j]) > .5:
                        pairs += 1; pair_ok += int((q[i] - q[j]) * (tv[i] - tv[j]) > 0)
    return {'states': n, 'best_agreement': agree, 'pairwise_sign': [pair_ok, pairs], 'mean_regret_of_q_greedy': float(np.mean(regret)) if regret else None,
            'mean_abs_q_step': float(np.mean(steps)) if steps else None}


def losses(model, batch=256, intent_batch=256):
    """The three V6/V7 critic losses on fresh batches (same code paths as FeasibleSupportSACV6.train)."""
    b = model.replay_buffer
    b.use(0); data = b.sample(batch)
    t, m = model._td_targets(data, model.target_value(data.next_observations))
    td = model._td_loss(model.critic(data.observations, data.actions), t, m)
    dec = b.sample_decisions(intent_batch); intent = adv = None
    if dec is not None:
        ti, mi = model._td_targets(dec, model.target_value(dec.next_observations))
        cur = model.critic(dec.observations, dec.actions); intent = model._td_loss(cur, ti, mi)
        rels = []
        for k in range(b.SLOTS):
            b.use(k); cf = b.counterfactual(); w = model._cf_weights(cf)
            if float(w.sum()) > 0:
                with th.no_grad():
                    delta = model._delta_target(dec, cf, None)
                rels.append(model._relative_loss(model.critic(dec.observations, cf.actions), cur, delta, w))
        b.use(0)
        adv = sum(rels) if rels else None
    return td, intent, adv


def flat_grad(loss, params):
    g = th.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return th.cat([(x if x is not None else th.zeros_like(p)).reshape(-1) for x, p in zip(g, params)])


def cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def directions(model, probes):
    params = list(model.critic.parameters()); out = {'intent_vs_adv': [], 'td_vs_adv': [], 'td_vs_intent': []}
    for _ in range(probes):
        td, intent, adv = losses(model)
        if intent is None or adv is None:
            continue
        gt, gi, ga = flat_grad(td, params), flat_grad(intent, params), flat_grad(adv, params)
        out['intent_vs_adv'].append(cos(gi, ga)); out['td_vs_adv'].append(cos(gt, ga)); out['td_vs_intent'].append(cos(gt, gi))
    return {k: {'median': float(np.median(v)), 'p10': float(np.percentile(v, 10)), 'p90': float(np.percentile(v, 90)), 'n': len(v)} if v else None
            for k, v in out.items()}


def train_variant(model, variant, updates, report_every, states, truth, lr=1e-4):
    critic0, target0 = copy.deepcopy(model.critic.state_dict()), copy.deepcopy(model.critic_target.state_dict())
    opt = th.optim.Adam(model.critic.parameters(), lr=lr); curve = [{'update': 0, **ranking(model, states, truth)}]
    for u in range(1, updates + 1):
        td, intent, adv = losses(model)
        parts = {'intent': [intent], 'intent+mpc': [intent, adv], 'online_v7': [td, intent, adv]}[variant]
        loss = sum(p for p in parts if p is not None)
        opt.zero_grad(); loss.backward(); opt.step(); polyak_update(model.critic.parameters(), model.critic_target.parameters(), model.tau)
        if u % report_every == 0:
            curve.append({'update': u, **ranking(model, states, truth)})
    model.critic.load_state_dict(critic0); model.critic_target.load_state_dict(target0)
    return curve


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--replay', required=True)
    p.add_argument('--truth', required=True); p.add_argument('--states', required=True)
    p.add_argument('--updates', type=int, default=3000); p.add_argument('--report-every', type=int, default=500)
    p.add_argument('--probes', type=int, default=40); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True); a = p.parse_args()
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(k, '1')
    th.set_num_threads(1); np.random.seed(a.seed); th.manual_seed(a.seed)
    model = FeasibleSupportSACV7.load(a.checkpoint, device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})
    model.load_replay_buffer(a.replay)
    b = model.replay_buffer; upper = b.buffer_size if b.full else b.pos
    if model.delta_lambda == 0.:
        model.delta_lambda = 1.0   # an E-SAC checkpoint has no pairs; the bench still reports what its Replay holds
    truth = load_truth(a.truth.split(',')); states = load_states(a.states.split(','))
    out = {'checkpoint': a.checkpoint, 'replay': a.replay, 'stored_transitions': int(upper * b.n_envs),
           'closed_decisions': int((b.due[:upper] & b.interval_ready[:upper]).sum()),
           'admitted_pairs_by_slot': [int((s['cf_weights'][:upper] > 0).sum()) for s in b._sets],
           'reference_states': len(states), 'at_checkpoint': ranking(model, states, truth), 'gradient_cosines': directions(model, a.probes)}
    print(json.dumps({k: out[k] for k in ('closed_decisions', 'admitted_pairs_by_slot', 'at_checkpoint', 'gradient_cosines')}), flush=True)
    out['offline_training'] = {}
    for v in ('intent', 'intent+mpc', 'online_v7'):
        out['offline_training'][v] = train_variant(model, v, a.updates, a.report_every, states, truth)
        print(v, json.dumps(out['offline_training'][v][-1]), flush=True)
    Path(a.output).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
