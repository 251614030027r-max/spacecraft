"""EACC iteration self-check (learner 4.1, env 0.11.0): real environment (reduced_priv, one-stage labels with the executed intent's same-treatment anchor)
-> EACC learner (e_rel) updates. Compares the per-decision wall time with the T2.11 e_acc environment (stage 2 + opportunity
judgements) on the same openings and actions.

    python -B evidence/T2_13_PRECHECK/脚本/selfcheck_iter.py CAL.json OUT.json
"""
import sys, time, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import numpy as np, torch; torch.set_num_threads(1)
from stable_baselines3.common.logger import configure
from env.t2_stage import T2StageEnv, T2Mission
from experiments.t2_audit import load_calibration
import train.t2_mve_sac as t
from env.t2_stage import CANDIDATE_MIN_SEP_M_S
t.CANDIDATE_MIN_SEP_M_S = CANDIDATE_MIN_SEP_M_S

cal, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for arm in ('e_acc', 'e_rel'):
    env = T2StageEnv(seed_base=1_000_007, candidates=8, shaping=True, seed=1, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)),
                     obs_profile='reduced_priv', pref=arm == 'e_acc', entry_stages=2 if arm == 'e_acc' else 1,
                     cand_min_sep=t.CANDIDATE_MIN_SEP_M_S if arm == 'e_rel' else 0.0)
    model = t.make_model(t.SpacesEnv(t.OBS_DIM_PRIV), arm, 0, smoke=True); model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
    if arm == 'e_acc':
        env.label_policy = lambda o: (np.random.default_rng(0).uniform(-1, 1, 2).astype(np.float32), 0.0)
    obs, _ = env.reset(); load_calibration(env, cal)
    rng = np.random.default_rng(1); t0 = time.time(); n = 40; kinds = []
    for i in range(n):
        if arm == 'e_acc':
            env.policy_action = model.predict(obs, deterministic=True)[0]
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        nobs, r, term, trunc, info = env.step(a)
        f = (info['model'] or {}).get('fact')
        if f is not None and f.get('same') is not None:
            kinds.append((int(f['same']['kind']), bool(f['same']['done'])))
        info = t._slim(info)
        model.replay_buffer.add(obs[None], nobs[None], a[None], np.array([r]), np.array([term or trunc]), [info])
        obs = env.reset()[0] if (term or trunc) else nobs
    wall = time.time() - t0
    res[arm] = {'decisions': n, 'wall_s': wall, 'per_decision_s': wall / n, 'labelled': model.replay_buffer.model_counts.get('with_candidates'),
                'candidates': model.replay_buffer.model_counts.get('candidates'), 'resampled': model.replay_buffer.model_counts.get('resampled', 0), 'same_kinds': kinds}
    if arm == 'e_rel':
        model.num_timesteps = n; model.learning_starts = 0; model.model_start = 0; model.td_ms = None
        model.train(gradient_steps=300, batch_size=16)
        ms = model.model_stats
        seg = lambda lo, hi, keys: {k: round(float(np.mean([m[k] for m in ms[lo:hi] if m.get(k) is not None])), 3) if any(m.get(k) is not None for m in ms[lo:hi]) else None for k in keys}
        keys = ('model_rows', 'w_mean', 'abs_err_median', 'same_err_median', 'same_err_p90', 'delta_abs_mean_sigma', 'entry_share',
                'class_differs_share', 'rank_agree', 'loss_share', 'grad_cos', 'real_td', 'model_td', 'sigma')
        res[arm]['first_probes'] = seg(0, 3, keys); res[arm]['last_probes'] = seg(len(ms) - 3, len(ms), keys)
        res[arm]['same_obs_dim'] = int(model.replay_buffer.same_obs.shape[-1]); res[arm]['same_valid'] = int(model.replay_buffer.same_valid.sum())
    print(arm, json.dumps({k: v for k, v in res[arm].items() if k != 'same_kinds'}), flush=True)
out.write_text(json.dumps(res, indent=1), encoding='utf-8')
