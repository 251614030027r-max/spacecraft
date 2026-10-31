"""T1-B V7 learner: V6 (intent-level replay + MPC-paired outcome advantage) unchanged; the replay
attaches the asynchronous real-controller full-horizon pairs of ``env.t1_b_feasible_v7``.

The decision transition carries ``counterfactual_marker``; the real discounted return from that
decision is accumulated per environment; the episode's terminal transition carries
``counterfactual_final`` (the factual and alternative model branches), which is gated against the
real outcome exactly as in V5.1/V6 (same terminal event within 10 steps, |return error| <= 5).
The worker pool lives in the training process (SB3 environment subprocesses are daemonic): the
environment emits ``counterfactual_snapshot`` at the decision; ``_store_transition`` submits the
branches and, at the episode's terminal transition, waits for them and attaches the payloads.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from env.t1_b_feasible_v7 import CF_WORKERS, _init_worker, _run_branch, payloads_from

from train.t1_b_interval_sac import DelayedModelReplayBuffer
from train.t1_b_v4_sac import FeasibleSupportSACV4
from train.t1_b_v6_sac import FeasibleSupportSACV6, IntentReplay

CF_INFO_KEYS = ('counterfactual_pending', 'counterfactual', 'counterfactual_marker', 'counterfactual_final', 'counterfactual_snapshot')


class AsyncIntentReplay(IntentReplay):
    def add(self, obs, next_obs, action, reward, done, infos):
        self.use(0)
        slot, index = self.pos, self.n_adds
        markers = [bool(i.get('counterfactual_marker')) for i in infos]
        finals = [i.get('counterfactual_final') for i in infos]
        clean = [{k: v for k, v in i.items() if k not in CF_INFO_KEYS} for i in infos]
        for k in range(1, self.SLOTS):
            self._sets[k]['cf_weights'][slot] = 0.
        DelayedModelReplayBuffer.add(self, obs, next_obs, action, reward, done, clean)
        for e in range(self.n_envs):
            if markers[e]:
                self._prefix[e] = [slot, index, 0., 0, None]
            p = self._prefix[e]
            if p is None:
                continue
            p[2] += self.gamma ** p[3] * float(reward[e]); p[3] += 1
            if bool(done[e]):
                self._prefix[e] = None
                if finals[e]:
                    p[4] = list(finals[e])
                    self._release(e, p, np.asarray(next_obs[e]), infos[e].get('event'), infos[e].get('TimeLimit.truncated', False))
        self.use(0)


class FeasibleSupportSACV7(FeasibleSupportSACV6):
    def _setup_model(self):
        self.replay_buffer_class = AsyncIntentReplay
        super(FeasibleSupportSACV4, self)._setup_model()

    def _cf_pool(self, spec):
        if getattr(self, '_pool', None) is None:
            self._pool = ProcessPoolExecutor(CF_WORKERS, mp_context=mp.get_context('spawn'), initializer=_init_worker, initargs=(spec,))
            self._cf_jobs = {}
        return self._pool

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        for e, info in enumerate(infos):
            job = info.pop('counterfactual_snapshot', None)
            if job is not None:
                pool = self._cf_pool(job['spec'])
                self._cf_jobs[e] = (job, [pool.submit(_run_branch, (job['snap'], r)) for r in job['raws']])
                info['counterfactual_marker'] = True
            if bool(dones[e]) and getattr(self, '_cf_jobs', None) and e in self._cf_jobs:
                job, futures = self._cf_jobs.pop(e)
                info['counterfactual_final'] = payloads_from(job, [f.result() for f in futures])
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), '_pool', '_cf_jobs']
