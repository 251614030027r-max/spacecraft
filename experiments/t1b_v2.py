"""T1-B V2 driver (learning-layer repair of V1; task, G*, MPC and truth unchanged).

    python -B -m experiments.t1b_v2 contract --output OUT
    python -B -m experiments.t1b_v2 expert   --output OUT --steps 1000            # V1 teacher, only if the V1 pickle is absent
    python -B -m experiments.t1b_v2 train    --arm {e_sac,e_tcpf} --seed S --steps N --expert PKL --output OUT
    python -B -m experiments.t1b_v2 eval     --checkpoint ZIP --label L --openings 292140-292147 --output OUT
    python -B -m experiments.t1b_v2 scan     --openings 292140-292147 --output OUT   # truth of each first-review candidate
    python -B -m experiments.t1b_v2 audit    --checkpoints A.zip,B.zip --episodes DIR --output OUT.json

Paths are explicit; the mission profile and the untrained initial weights are the published V1
files (SHA-256 checked). Every run starts from the same untrained interface_0k weights as V1.
"""
import argparse, hashlib, json, os, pickle, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import torch

from env.t1_b_task import MissionDistribution
from env.t1_b_feasible_v1 import bin_center
from env.t1_b_feasible_v2 import FeasibleArrivalEnvV2, VERSION as VERSION_V2
from env.t1_b_feasible_v3 import FeasibleArrivalEnvV3, VERSION as VERSION_V3
from env.t1_b_feasible_v4 import FeasibleArrivalEnvV4, VERSION as VERSION_V4
from train.t1_b_v4_sac import FeasibleSupportSACV4, V4_BEHAVIOUR_EPSILON
from env.t1_b_feasible_v5 import FeasibleArrivalEnvV5, VERSION as VERSION_V5
from train.t1_b_v5_sac import FeasibleSupportSACV5
from train.t1_b_v6_sac import FeasibleSupportSACV6, CF_WEIGHT
from env.t1_gym import T1GymConfig
from env.t1_n_review import N_DUE_INDEX
from train.t1_b_v2_sac import FeasibleSupportSACV2, CANDIDATE_ALPHA, BEHAVIOUR_EPSILON
from train.t1_tcpf import CosineLearningRate

PROFILE = Path('evidence/T1B_V1/profile_mission_v3.json')
PROFILE_SHA = 'd1eaa6c55ef53874c1f5757fcbe802f270060496ea540adc92053ee806329ff0'
INIT = Path('evidence/T1B_V1/models/E-SAC/step_0.zip')
INIT_SHA = '40da22c6ea8483a4ab75e0f07bd924a98d005d297482d850b3d48f8cb46c8174'
EXPERT_SHA = 'bf3c24f9f8db7eb156a5ab42835f6faeefee2332efbd59ba7c35ed733f5ffe80'
# v2: a1698ac pilot interface; v3: committed arrival intent (env/t1_b_feasible_v3.py). Same learner.
INTERFACES = {'v2': (FeasibleArrivalEnvV2, VERSION_V2), 'v3': (FeasibleArrivalEnvV3, VERSION_V3), 'v4': (FeasibleArrivalEnvV4, VERSION_V4), 'v5': (FeasibleArrivalEnvV5, VERSION_V5), 'v6': (FeasibleArrivalEnvV5, 't1b-intent-replay-mpc-advantage-v6')}
# v6: V5.1 interface + intent-level replay; E-TCPF adds the MPC-paired advantage (train/t1_b_v6_sac.py)
# v5: E-TCPF only -- full-horizon paired MPC counterfactual of the first decision (train/t1_b_v5_sac.py); its E-SAC is v4's
# v4: V3 decisions + every unchosen candidate paired + behaviour epsilon 0.5 (train/t1_b_v4_sac.py)
LEARNERS = {'v2': FeasibleSupportSACV2, 'v3': FeasibleSupportSACV2, 'v4': FeasibleSupportSACV4, 'v5': FeasibleSupportSACV5, 'v6': FeasibleSupportSACV6}
TEACHER_INTERFACE = {'v2': 'v2', 'v3': 'v3', 'v4': 'v3', 'v5': 'v3', 'v6': 'v3'}   # V4 learning decisions are V3's: the V3 teacher file is valid


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def default(x): return x.tolist() if isinstance(x, np.ndarray) else (x.item() if isinstance(x, np.generic) else str(x))
def dump(path, x):
    path = Path(path); tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(x, ensure_ascii=False, indent=1, default=default), encoding='utf-8'); tmp.replace(path)
def openings(text):
    a, _, b = text.partition('-'); return list(range(int(a), int(b or a) + 1))


def mission():
    assert sha(PROFILE) == PROFILE_SHA, 'mission profile changed'
    return MissionDistribution(**json.loads(PROFILE.read_text(encoding='utf-8-sig')))


def make_env(rank, feedback=False, seed=0, interface='v2'):
    return INTERFACES[interface][0](config=T1GymConfig(seed_base=1_000_000 * (seed + 1) + rank * 100_000), mission=mission(), feedback=feedback)


def model_for(env, arm, seed, output=None, smoke=False, interface='v2'):
    kwargs = {'feedback_audit_path': str(Path(output) / 'model_prefix_releases.jsonl')} if output else {}
    m = LEARNERS[interface]('MlpPolicy', env, seed=seed, device='cpu', verbose=0, policy_kwargs={'net_arch': [128, 128]}, gamma=.999,
                             buffer_size=20000 if smoke else 100000, batch_size=64 if smoke else 256, learning_starts=32 if smoke else 500,
                             learning_rate=CosineLearningRate(1e-4, 1e-5), ent_coef=.005, tau=.005, train_freq=(1, 'step'), gradient_steps=4,
                             interval_bellman=True, actor_every_transitions=3, mc_lambda=0., eq_lambda=0., audit_every=0,
                             delta_lambda=CF_WEIGHT if interface == 'v6' and arm == 'e_tcpf' else 0.,
                             delta_target_share=None if interface == 'v6' else ((.10 if interface == 'v5' else .05) if arm == 'e_tcpf' else None),
                             delta_cap_share=.2 if interface == 'v5' else .1,
                             delta_warmup_decisions=32 if smoke else 500, delta_calibration_steps=32 if smoke else 500, delta_probe_every=5,
                             delta_min_probes=5 if smoke else 30, delta_diagnostics_every=20, preserve_warmup_options=False,
                             replay_buffer_kwargs=kwargs)
    assert sha(INIT) == INIT_SHA, 'initial weights changed'
    init = FeasibleSupportSACV2.load(INIT, device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})
    m.policy.load_state_dict(init.policy.state_dict())
    return m


# ------------------------------------------------------------------ contract
def contract(out, interface='v2'):
    torch.set_num_threads(1)
    env = INTERFACES[interface][0](openings=[292140], mission=mission()); obs, _ = env.reset()
    m = model_for(env, 'e_sac', 0, smoke=True, interface=interface)
    init = FeasibleSupportSACV2.load(INIT, device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})
    same = float(np.max(np.abs(init.predict(obs, deterministic=True)[0] - m.predict(obs, deterministic=True)[0])))
    assert same == 0.0
    z = torch.tensor(obs[None]); assert float(m.alpha_of(z)) == CANDIDATE_ALPHA
    p = m.support_probabilities(z); assert abs(float(p.sum()) - 1) < 1e-6
    # alternative rule: deterministic, never the chosen index, covers both directions
    alts = {(i, env.alternative_index([80., 100., 120.], i)) for i in range(3) for env.current_opening in range(292140, 292160)}
    assert all(i != j for i, j in alts) and {(1, 0), (1, 2)} <= alts
    dump(out / 'CONTRACT.json', {'passed': True, 'init_weights_max_abs_action_diff': same, 'candidate_alpha': CANDIDATE_ALPHA,
                                 'behaviour_epsilon': m.behaviour_epsilon, 'version': INTERFACES[interface][1], 'alternative_pairs_seen': sorted(alts)})
    print('CONTRACT PASS', flush=True)


# ------------------------------------------------------------------ teacher replay (V1 rule)
def teacher_action(z):
    n = int((z[49:52] > .5).sum()); choices = z[46:49] * 200
    j = int(np.argmin(abs(choices[:n] - z[28] * 200))) if n and z[27] > .5 else 0
    return [bin_center(j, n) if n else 0.]


def expert(out, transitions, interface='v2'):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = SubprocVecEnv([partial(make_env, k, False, 0, interface) for k in range(4)], start_method='spawn'); obs = env.reset(); records = []
    for _ in range(transitions // 4):
        nxt, rewards, dones, infos = env.step(np.array([teacher_action(z) for z in obs], np.float32))
        terminal = np.stack([i.get('terminal_observation', nxt[k]) for k, i in enumerate(infos)])
        records.append({'obs': obs.copy(), 'next_obs': terminal.copy(), 'action': np.stack([i['executed_action'] for i in infos]), 'reward': rewards.copy(),
                        'done': dones.copy(), 'infos': [{k: i.get(k) for k in ['event', 'TimeLimit.truncated', 'review', 'execution', 'mission_deadline_s',
                                                                              'mission_budget_m_s', 'budget_remaining_m_s', 'support_transition']} for i in infos]})
        obs = nxt
    env.close()
    with (out / 'expert_replay.pkl').open('wb') as f: pickle.dump(records, f)
    dump(out / 'EXPERT_RECEIPT.json', {'transitions': transitions, 'sha256': sha(out / 'expert_replay.pkl'), 'interface': interface, 'rule': 'V1 teacher: earliest without plan, keep a live plan'})


# ------------------------------------------------------------------ training
def train(out, arm, steps, seed, expert_file, smoke=False, interface='v2'):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from experiments.t1b_repair_v1 import RecordV1
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if not smoke and subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('formal training requires a clean committed checkout')
    if not smoke and interface == 'v2' and sha(expert_file) != EXPERT_SHA:
        raise RuntimeError('expert replay is not the V1 teacher file')
    if not smoke and interface != 'v2':
        # the teacher rule is V1's, recollected on the same interface (its flags/candidates must match)
        receipt = json.loads((Path(expert_file).parent / 'EXPERT_RECEIPT.json').read_text(encoding='utf-8'))
        if receipt.get('interface') != TEACHER_INTERFACE[interface] or receipt.get('sha256') != sha(expert_file) or receipt.get('transitions') != 1000:
            raise RuntimeError('expert replay is not a 1000-transition teacher collection on this interface')
    out.mkdir(parents=True, exist_ok=False); (out / 'checkpoints').mkdir()
    torch.set_num_threads(1)
    env = SubprocVecEnv([partial(make_env, k, arm == 'e_tcpf', seed, interface) for k in range(4)], start_method='spawn')
    model = model_for(env, arm, seed, out, smoke, interface)
    records = pickle.loads(Path(expert_file).read_bytes())
    for r in records: model.replay_buffer.add(r['obs'], r['next_obs'], r['action'], r['reward'], r['done'], r['infos'])
    b = model.replay_buffer; b.demo_slots = b.pos; b._open = [None] * 4; b._prefix = [None] * 4; b._episode = [[] for _ in range(4)]
    dump(out / 'manifest.json', {'version': INTERFACES[interface][1], 'interface': interface, 'arm': arm, 'seed': seed, 'steps': steps, 'smoke_not_formal': smoke, 'git_head': head,
                                 'profile': asdict(mission()), 'profile_sha256': PROFILE_SHA, 'init_weights': str(INIT), 'init_sha256': INIT_SHA,
                                 'expert_sha256': sha(expert_file), 'expert_real_transitions': len(records) * 4, 'expert_sampling_fraction': .2,
                                 'candidate_alpha': CANDIDATE_ALPHA, 'behaviour_epsilon': model.behaviour_epsilon,
                                 'cf_alternative_rule': INTERFACES[interface][0].ALTERNATIVE_RULE if arm == 'e_tcpf' else None,
                                 'unchanged_from_V1': 'task, D/B, support, G*, MPC, truth, network, init, optimiser, trainer, expert Replay, CF gate and 5 %/10 % share'})
    model.learn(total_timesteps=steps, callback=RecordV1(out, steps)); model.save(out / 'final')
    dump(out / 'learning_audit.json', model.support_stats); dump(out / 'feedback_audit.json', b.feedback_stats); dump(out / 'delta_calibration.json', model.delta_calibration)
    dump(out / 'finished.json', {'steps': model.num_timesteps, 'critic_updates': model._n_updates, 'actor_updates': model.actor_updates,
                                 'behaviour_counts': model.behaviour_counts, 'expert_sampling': b.sample_counts, 'interval_counts': b.interval_counts})
    env.close(); print('TRAIN COMPLETE', arm, seed, steps, flush=True)


# ------------------------------------------------------------------ evaluation (deterministic, serial per opening)
def eval_job(job):
    opening, checkpoint, label, out, interface = job; torch.set_num_threads(1); file = Path(out) / f'{label}_{opening}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    env = INTERFACES[interface][0](openings=[opening], mission=mission()); obs, reset = env.reset()
    model = FeasibleSupportSACV2.load(checkpoint, device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})
    initial, done, k, disc, first = obs.tolist(), False, 0, 0., None
    while not done:
        a = model.predict(obs, deterministic=True)[0]; obs, r, t, x, i = env.step(a); done = t or x
        if first is None and i['support_transition']['learning_decision']:
            first = {'tau_candidates': i['support_transition']['tau_candidates'], 'index': i['support_transition']['candidate_index']}
        disc += .999 ** k * r; k += 1
    res = {'opening': opening, 'label': label, 'interface': interface, 'checkpoint': str(checkpoint), 'mission': reset['mission'], 'event': i['event'],
           'discounted_return': disc, 'time_s': env.env.time_s, 'delta_v': env.env._delta_v, 'first_decision': first, 'initial_observation': initial}
    dump(file, res); return res


def evaluate(out, checkpoint, label, ops, workers, interface='v2'):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(eval_job, (o, checkpoint, label, str(out), interface)) for o in ops]):
            r = f.result(); print(json.dumps({k: r[k] for k in ('opening', 'label', 'event', 'discounted_return', 'first_decision')}, default=default), flush=True)


# ------------------------------------------------------------------ candidate truth scan
def scan_job(job):
    opening, j0, out, interface = job; torch.set_num_threads(1); file = Path(out) / f'scan_{opening}_{j0}.json'
    if file.exists(): return json.loads(file.read_text(encoding='utf-8'))
    env = INTERFACES[interface][0](openings=[opening], mission=mission()); obs, reset = env.reset()
    done, k, disc, choices0, skipped = False, 0, 0., None, False
    while not done:
        z = obs; n = int((z[49:52] > .5).sum()); a = [0.]
        if z[N_DUE_INDEX] > .5 and n >= 2:
            if choices0 is None:
                choices0 = (z[46:46 + n] * 200).tolist()
                if j0 >= n: skipped = True; break
                a = [bin_center(j0, n)]
            else:
                a = teacher_action(z)
        obs, r, t, x, i = env.step(np.array(a, np.float32)); done = t or x; disc += .999 ** k * r; k += 1
    res = {'opening': opening, 'first_index': j0, 'first_candidates_s': choices0, 'skipped': skipped, 'mission': reset['mission'],
           'continuation': 'V1 teacher (keep a live plan, earliest without plan)'}
    if not skipped: res.update(event=i['event'], discounted_return=disc, time_s=env.env.time_s, delta_v=env.env._delta_v)
    dump(file, res); return res


def scan(out, ops, workers, interface='v2'):
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        for f in as_completed([pool.submit(scan_job, (o, j, str(out), interface)) for o in ops for j in range(3)]):
            r = f.result(); print(json.dumps({k: r.get(k) for k in ('opening', 'first_index', 'first_candidates_s', 'event', 'discounted_return')}, default=default), flush=True)


# ------------------------------------------------------------------ policy audit (offline, no updates)
def audit(checkpoints, episodes, output):
    obs = [json.loads(Path(f).read_text(encoding='utf-8-sig'))['initial_observation'] for f in sorted(Path(episodes).glob('*.json'))]
    z = torch.tensor(np.array(obs, np.float32)); rows = {}
    edge = float(np.arctanh(-1 / 3))
    for path in checkpoints:
        m = FeasibleSupportSACV2.load(path, device='cpu', custom_objects={'learning_rate': 1e-4, 'lr_schedule': lambda _: 1e-4})
        mean, logstd, _ = m.actor.get_action_dist_params(z); p = m.support_probabilities(z)
        q = torch.minimum(*m.candidate_q(z, m.critic)).detach(); alpha = m.alpha_of(z)
        loss = -(p * q).sum(1).mean() + (alpha * p * p.clamp_min(1e-8).log()).sum(1).mean()
        g = torch.autograd.grad(loss, list(m.actor.parameters()), allow_unused=True)
        rows[str(path)] = {'alpha': float(alpha.mean()), 'p': p.detach().numpy().round(6).tolist(), 'q': q.numpy().round(3).tolist(),
                           'z_first_edge': ((edge - mean) / logstd.exp()).detach().numpy().ravel().round(2).tolist(),
                           'actor_grad_norm': float(torch.sqrt(sum((x * x).sum() for x in g if x is not None))),
                           'entropy': float(-(p * p.clamp_min(1e-8).log()).sum(1).mean())}
        print(path, 'grad %.3g' % rows[str(path)]['actor_grad_norm'], 'H %.3f' % rows[str(path)]['entropy'], flush=True)
    dump(output, rows)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('mode', choices=['contract', 'expert', 'train', 'eval', 'scan', 'audit'])
    p.add_argument('--output', required=True); p.add_argument('--arm', choices=['e_sac', 'e_tcpf'], default='e_sac'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=5000); p.add_argument('--expert'); p.add_argument('--checkpoint'); p.add_argument('--checkpoints')
    p.add_argument('--label'); p.add_argument('--openings', default='292140-292147'); p.add_argument('--episodes'); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--interface', choices=sorted(INTERFACES), default='v2')
    a = p.parse_args(); out = Path(a.output)
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(k, '1')
    if a.mode not in ('train', 'audit'): out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'contract': contract(out, a.interface)
    elif a.mode == 'expert': expert(out, a.steps, a.interface)
    elif a.mode == 'train': train(out, a.arm, a.steps, a.seed, a.expert, a.smoke, a.interface)
    elif a.mode == 'eval': evaluate(out, a.checkpoint, a.label, openings(a.openings), a.workers, a.interface)
    elif a.mode == 'scan': scan(out, openings(a.openings), a.workers, a.interface)
    else: audit(a.checkpoints.split(','), a.episodes, out)


if __name__ == '__main__':
    main()
