"""Learner 6.0 self-check: real environment (reduced_priv, warm calibrations, ref labels of the policy's own intent) ->
e_pi / e_pit learners (critic on real transitions only, execution-corrected actor gradient).

    python -B evidence/T2_16_PRECHECK/脚本/selfcheck_pi.py CAL.json OUT.json
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
for arm in ('e_pi', 'e_pit'):
    model = t.make_model(t.SpacesEnv(t.OBS_DIM_PRIV), arm, 0, smoke=True); model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    env = T2StageEnv(seed_base=1_000_007, candidates=8, shaping=True, seed=1, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), obs_profile='reduced_priv',
                     entry_stages=1, cand_min_sep=CANDIDATE_MIN_SEP_M_S)
    env.ref_labels = True
    obs, _ = env.reset(); load_calibration(env, cal)
    rng = np.random.default_rng(1); t0 = time.time(); n = 40
    for i in range(n):
        env.policy_action = model.predict(obs, deterministic=True)[0]
        a = np.clip(env.policy_action + rng.normal(0, 0.3, 2), -1, 1).astype(np.float32)
        nobs, r, term, trunc, info = env.step(a)
        info = t._slim(info)
        model.replay_buffer.add(obs[None], nobs[None], a[None], np.array([r]), np.array([term or trunc]), [info])
        obs = env.reset()[0] if (term or trunc) else nobs
    wall = time.time() - t0
    rb = model.replay_buffer
    res[arm] = {'decisions': n, 'per_decision_s': wall / n, 'labelled': rb.model_counts.get('with_candidates'), 'ref': rb.model_counts.get('ref', 0),
                'candidates': rb.model_counts.get('candidates')}
    model.num_timesteps = n; model.learning_starts = 0; model.model_start = 0; model.td_ms = None
    model.train(gradient_steps=300, batch_size=16)
    ms = model.model_stats
    keys = ('fb_rows', 'fb_cands', 'fb_applied', 'fb_rho_mean', 'fb_corr_mean', 'fb_corr_abs_mean', 'fb_g_norm_mean', 'fb_gq_rms', 'fb_cos', 'real_td', 'model_td')
    seg = lambda lo, hi: {k: round(float(np.mean([m[k] for m in ms[lo:hi] if m.get(k) is not None])), 3) if any(m.get(k) is not None for m in ms[lo:hi]) else None for k in keys}
    res[arm]['first_probes'] = seg(0, 3); res[arm]['last_probes'] = seg(len(ms) - 3, len(ms))
    print(arm, json.dumps(res[arm]), flush=True)
out.write_text(json.dumps(res, indent=1), encoding='utf-8')
