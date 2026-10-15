"""Validation B: nominal one-QP prediction vs real 2 s closed loop, executed and counterfactual branch."""
import sys, json, numpy as np, torch
from types import SimpleNamespace
from env.t1_gym import T1GStarEnv
from env.t1_counterfactual import CounterfactualPredictor, prediction_errors
source, openings, stride, out = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], int(sys.argv[3]), sys.argv[4]
policy = None
if source == 'm0':
    from stable_baselines3 import SAC
    policy = SAC.load(sys.argv[5], device='cpu'); torch.manual_seed(0)
KAPPA_A1 = (0.7 - 0.3) / 0.35 - 1.0
def rule_action(env, tau0):
    if env.generator.remaining_s(env.env) > 0: return np.array([0.0, KAPPA_A1])
    return np.array([tau0 / 80.0 - 1.0, KAPPA_A1])
f = open(out, 'w')
for opening in openings:
    real = T1GStarEnv(openings=[opening]); obs, _ = real.reset()
    pred = CounterfactualPredictor(); k = 0
    while True:
        if source == 'm0': a = np.asarray(policy.predict(obs, deterministic=False)[0], dtype=np.float64)
        else: a = rule_action(real, 0.0 if source == 'rv0' else 70.0) if (k == 0 or source == 'rv0') else rule_action(real, 0.0)
        if source == 'rv70' and k == 0: a = np.array([70.0/80.0 - 1.0, KAPPA_A1])
        if k % stride == 0:
            rec = {'source': source, 'opening': opening, 'k': k, 'time_s': real.env.time_s, 'action': a.tolist()}
            pred.sync(real); p_a = pred.predict(a)
            alt, kind = pred.alternative(real, a, p_a); rec['kind'] = kind
            pred.sync(real); cl_a = pred.closed_loop(a)
            e_a = prediction_errors(p_a, cl_a['reward'], cl_a['observation'], cl_a['relative_position_eci_m'])
            rec.update(status_a=p_a.plan_status, tau_eff_a=p_a.tau_effective_s, sat_a=p_a.saturation, e_a=e_a,
                       dobs_a=(p_a.observation - cl_a['observation']).tolist(), event_a=cl_a['event'], slack_a=p_a.solve_slack,
                       wall_pred_a=p_a.wall_s)
            if alt is not None:
                pred.sync(real); p_b = pred.predict(alt)
                pred.sync(real); cl_b = pred.closed_loop(alt)
                e_b = prediction_errors(p_b, cl_b['reward'], cl_b['observation'], cl_b['relative_position_eci_m'])
                after = SimpleNamespace(env=SimpleNamespace(time_s=real.env.time_s + 2.0, config=real.env.config))
                w, reasons = pred.gate(after, p_a, p_b, e_a)
                rec.update(alt=alt.tolist(), status_b=p_b.plan_status, tau_eff_b=p_b.tau_effective_s, sat_b=p_b.saturation,
                           e_b=e_b, dobs_b=(p_b.observation - cl_b['observation']).tolist(), event_b=cl_b['event'],
                           w_structural=w, reasons=reasons, wall_pred_b=p_b.wall_s,
                           dQ_obs=float(np.linalg.norm(p_b.observation - p_a.observation)))
        obs2, r, te, tr, info = real.step(a)
        if k % stride == 0:
            rec['real_matches_shadow_closed_loop'] = bool(np.array_equal(obs2, cl_a['observation']) and r == cl_a['reward'])
            f.write(json.dumps(rec) + '\n'); f.flush()
        obs = obs2; k += 1
        if te or tr: break
    print(source, opening, 'done', k, info['event'], flush=True)
