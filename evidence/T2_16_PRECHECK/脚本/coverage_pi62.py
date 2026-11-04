"""Learner 6.2 coverage check (6.1's script with the 6.2 readings fb_turn_cos, fb_m_ratio) (sandbox): collection interleaved with learning as in training (warm calibrations, two
updates per stored transition, the policy's own deterministic intent as the reference at every decision), and the share
of feedback rows actually applied to the actor over time.

    python -B evidence/T2_16_PRECHECK/脚本/coverage_pi.py CAL.json ARM DECISIONS OUT.json
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np, torch as th; th.set_num_threads(1)
from stable_baselines3.common.logger import configure
from env.t2_stage import T2StageEnv, T2Mission, CANDIDATE_MIN_SEP_M_S
from experiments.t2_audit import load_calibration
import train.t2_mve_sac as t

cal, arm, n, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
model = t.make_model(t.SpacesEnv(t.OBS_DIM_PRIV), arm, 0, smoke=True); model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
model.learning_starts = 32; model.model_start = 32; model.batch_size = 64
env = T2StageEnv(seed_base=1_000_007, candidates=8, shaping=True, seed=1, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), obs_profile='reduced_priv',
                 entry_stages=1, cand_min_sep=CANDIDATE_MIN_SEP_M_S)
env.ref_labels = True
obs, _ = env.reset(); load_calibration(env, cal)
t0 = time.time(); episodes = []
for i in range(n):
    env.policy_action = model.predict(obs, deterministic=True)[0]
    a = model.predict(obs, deterministic=False)[0] if i >= model.learning_starts else np.random.default_rng(i).uniform(-1, 1, 2).astype(np.float32)
    nobs, r, term, trunc, info = env.step(a)
    slim = t._slim(info)
    model.replay_buffer.add(obs[None], nobs[None], np.asarray(a, np.float32)[None], np.array([r]), np.array([term or trunc]), [slim])
    model.num_timesteps += 1
    if term or trunc:
        episodes.append(info['event']); obs = env.reset()[0]
    else:
        obs = nobs
    if model.num_timesteps >= model.learning_starts:
        model.train(gradient_steps=2, batch_size=model.batch_size)
ms = [m for m in model.model_stats]
fb = [m for m in model.model_stats if 'fb_rows' in m]
def seg(lo, hi):
    s = [m for m in fb if lo <= m['steps'] < hi]
    g = lambda k: (round(float(np.mean([m[k] for m in s if m.get(k) is not None])), 3) if any(m.get(k) is not None for m in s) else None)
    return {'probes': len(s), **{k: g(k) for k in ('fb_rows', 'fb_identified', 'fb_applied', 'fb_applied_share', 'fb_rho_mean', 'fb_corr_mean',
                                                   'fb_g_norm_mean', 'fb_gq_rms', 'fb_cos', 'fb_turn_cos', 'fb_m_ratio', 'fb_label_age_mean')}}
res = {'arm': arm, 'decisions': n, 'wall_s': time.time() - t0, 'episodes': episodes, 'refs': model.replay_buffer.model_counts.get('ref', 0),
       'labelled': model.replay_buffer.model_counts.get('with_candidates'),
       'segments': {f'{lo}-{hi}': seg(lo, hi) for lo, hi in ((0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n + 1))}}
out.write_text(json.dumps(res, indent=1), encoding='utf-8'); print(json.dumps(res, indent=1))
