"""Self-check 2: real environment (reduced interface, EACC labels + opportunity judgements) -> EACC-A learner updates."""
import sys, time, json
sys.path.insert(0, '/home/user/spacecraft')
import numpy as np, torch; torch.set_num_threads(1)
from pathlib import Path
from stable_baselines3.common.logger import configure
from env.t2_stage import T2StageEnv, T2Mission
from experiments.t2_audit import load_calibration
import train.t2_mve_sac as t
env = T2StageEnv(seed_base=1_000_007, candidates=8, shaping=True, seed=1, mission=T2Mission(urgent_deadline_s=(140.0, 200.0)), obs_profile='reduced_priv', pref=True)
model = t.make_model(t.SpacesEnv(t.OBS_DIM_PRIV), 'e_acc', 0, smoke=True); model.set_logger(configure(None, [])); model._current_progress_remaining = 1.0
env.label_policy = lambda o: (np.random.uniform(-1, 1, 2).astype(np.float32), 0.0)
obs, _ = env.reset(); load_calibration(env, Path('/home/user/spacecraft/evidence/T2_8_PRECHECK/输入/CAL.json'))
rng = np.random.default_rng(1); t0 = time.time(); pend = []; n_pref = n_lab = 0
for i in range(40):
    env.policy_action = model.predict(obs, deterministic=True)[0]
    a = rng.uniform(-1, 1, 2).astype(np.float32)
    nobs, r, term, trunc, info = env.step(a); info = t._slim(info)
    if info.get('pref'):
        n_pref += 1; pend.append(t.dec_features(info['pref']['f'][t.PREF_EXEC]))
    model.replay_buffer.add(obs[None], nobs[None], a[None], np.array([r]), np.array([term or trunc]), [info])
    if term or trunc:
        for fx in pend: model.dec_cal.observe(fx, info.get('event') == 'success'); n_lab += 1
        pend = []; obs, _ = env.reset()
    else:
        obs = nobs
wall = time.time() - t0
print(f'40 real decisions: {wall:.0f} s ({wall/40:.1f} s each); with opportunity judgements {n_pref}; outcomes labelled {n_lab}')
model.dec_cal.w = np.array(json.load(open(sys.argv[1]))['w']); model.dec_cal.hits = [1.0] * t.ACC_MIN_LABELS   # trusted P4 weights
model.num_timesteps = 40; model.learning_starts = 0; model.td_ms = None
model.train(gradient_steps=300, batch_size=16)
ms = model.model_stats
def seg(lo, hi, keys): return {k: round(float(np.mean([m[k] for m in ms[lo:hi] if m.get(k) is not None])), 3) if any(m.get(k) is not None for m in ms[lo:hi]) else None for k in keys}
keys = ('acc_rows', 'acc_q_ordered', 'acc_actor_dist', 'acc_q_share', 'model_rows', 'real_td')
print('first probes', seg(0, 3, keys)); print('last probes ', seg(len(ms) - 3, len(ms), keys))
nobs_shapes = {model.replay_buffer.cand_obs.shape[-1], model.replay_buffer.observations.shape[-1]}
print('replay obs / label next_obs dims', nobs_shapes, 'labels with candidates', model.replay_buffer.model_counts.get('with_candidates'))
