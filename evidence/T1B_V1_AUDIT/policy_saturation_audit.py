import json, glob, numpy as np, torch as th
from train.t1_b_feasible_sac import FeasibleSupportSAC
obs = []
for f in sorted(glob.glob('evidence/T1B_V1/episodes/interface_0k/*.json')):
    obs.append(json.load(open(f, encoding='utf-8-sig'))['initial_observation'])
z = th.tensor(np.array(obs, np.float32))
out = {}
for arm in ('E-SAC', 'E-TCPF'):
    for ck in (0, 2000, 5000, 10000):
        m = FeasibleSupportSAC.load(f'evidence/T1B_V1/models/{arm}/step_{ck}.zip', device='cpu')
        mean, logstd, _ = m.actor.get_action_dist_params(z)
        p = m.support_probabilities(z); q = th.minimum(*m.candidate_q(z, m.critic)).detach()
        alpha = m.alpha_of(z)
        qterm = -(p*q).sum(1).mean(); ent = (alpha*p*p.clamp_min(1e-8).log()).sum(1).mean()
        params = list(m.actor.parameters())
        g = th.autograd.grad(qterm+ent, params, allow_unused=True)
        gn = float(th.sqrt(sum((x*x).sum() for x in g if x is not None)))
        edge = float(np.arctanh(-1/3)); zz = ((edge - mean)/logstd.exp()).detach().numpy().ravel()
        soft = th.softmax(q/alpha, 1)
        out[f'{arm}_{ck}'] = {'alpha': float(alpha.mean()), 'z_edge01': zz.round(2).tolist(), 'phi_edge01': (np.exp(-zz**2/2)/np.sqrt(2*np.pi)).tolist(),
                              'p0': p[:, 0].tolist(), 'dQ_01': (q[:, 0]-q[:, 1]).tolist(), 'softopt_p0': soft[:, 0].tolist(),
                              'actor_grad_norm': gn, 'entropy_mean': float(-(p*p.clamp_min(1e-8).log()).sum(1).mean())}
        print(arm, ck, 'alpha %.3f' % float(alpha.mean()), 'z', np.round(zz, 1), 'gradnorm %.3g' % gn, 'dQ01 %.2f' % float((q[:,0]-q[:,1]).mean()), 'softopt p0 min %.6f' % float(soft[:,0].min()), 'H %.3g' % out[f'{arm}_{ck}']['entropy_mean'])
json.dump(out, open('/tmp/claude-0/-home-user-spacecraft/bc85fbf5-725d-5c89-990b-cb6f6a76397b/scratchpad/t1b_scan/grad_audit.json', 'w'), indent=1)
