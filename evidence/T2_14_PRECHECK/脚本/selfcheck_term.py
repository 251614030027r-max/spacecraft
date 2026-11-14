"""Learner 5.0 self-check: real environment (reduced_priv, warm calibrations) -> e_term / e_term2 learners; per-decision
wall time against the T2.11 e_acc environment on the same openings and actions.

    python -B evidence/T2_14_PRECHECK/脚本/selfcheck_term.py CAL.json OUT.json
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np, torch as th; th.set_num_threads(1)
from stable_baselines3.common.logger import configure
from env.t2_stage import T2StageEnv, T2Mission, CANDIDATE_MIN_SEP_M_S
from experiments.t2_audit import load_calibration
import train.t2_mve_sac as t

cal, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for arm in ('e_acc', 'e_term', 'e_term2'):
    model = t.make_model(t.SpacesEnv(t.OBS_DIM_PRIV), arm, 0, smoke=True); model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    env = T2StageEnv(seed_base=1_000_007, candidates=8, shaping=True, seed=1, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), obs_profile='reduced_priv',
                     pref=arm == 'e_acc', entry_stages={'e_acc': 2, 'e_term': 1, 'e_term2': 2}[arm], cand_min_sep=0.0 if arm == 'e_acc' else CANDIDATE_MIN_SEP_M_S)
    if arm == 'e_acc':
        env.label_policy = lambda o: (np.random.default_rng(0).uniform(-1, 1, 2).astype(np.float32), 0.0)
    elif arm == 'e_term2':
        env.label_policy = lambda o: (model.actor(th.as_tensor(np.asarray(o, np.float32)[None]), deterministic=True)[0].detach().numpy(), 0.0)
    obs, _ = env.reset(); load_calibration(env, cal)
    rng = np.random.default_rng(1); t0 = time.time(); n = 40
    for i in range(n):
        if arm == 'e_acc':
            env.policy_action = model.predict(obs, deterministic=True)[0]
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        nobs, r, term, trunc, info = env.step(a)
        info = t._slim(info)
        model.replay_buffer.add(obs[None], nobs[None], a[None], np.array([r]), np.array([term or trunc]), [info])
        obs = env.reset()[0] if (term or trunc) else nobs
    wall = time.time() - t0
    rb = model.replay_buffer
    res[arm] = {'decisions': n, 'per_decision_s': wall / n, 'labelled': rb.model_counts.get('with_candidates'), 'candidates': rb.model_counts.get('candidates'),
                'stage1_terminal': rb.model_counts.get('entry_candidates', 0) + rb.model_counts.get('cutoff_candidates', 0),
                'stage2': rb.model_counts.get('two_stage', 0), 'stage2_terminal': rb.model_counts.get('two_stage_entry', 0) + rb.model_counts.get('two_stage_cutoff', 0)}
    if arm != 'e_acc':
        model.num_timesteps = n; model.learning_starts = 0; model.model_start = 0; model.td_ms = None
        model.train(gradient_steps=300, batch_size=16)
        ms = model.model_stats
        keys = ('term_rows', 'term2_rows', 'term_share', 'comp_share', 'w_mean', 'term_target_mean', 'term_q_minus_target_mean', 'loss_share', 'real_td', 'model_td')
        seg = lambda lo, hi: {k: round(float(np.mean([m[k] for m in ms[lo:hi] if m.get(k) is not None])), 3) if any(m.get(k) is not None for m in ms[lo:hi]) else None for k in keys}
        res[arm]['first_probes'] = seg(0, 3); res[arm]['last_probes'] = seg(len(ms) - 3, len(ms))
    print(arm, json.dumps(res[arm]), flush=True)
out.write_text(json.dumps(res, indent=1), encoding='utf-8')
