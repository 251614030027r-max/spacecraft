"""Event-aligned value feedback (arm e_evt, ``docs/ecc2/ECC_EVT.md``): segments, storage, target."""
import numpy as np
import pytest

from env.t2_stage import GAMMA_STAGE, OBS_DIM, OBS_DIM_PRIV
from train.t2_mve_sac import (EVENT_LAMBDA, EVENT_SPAN, EventReplay, EventSamples, SpacesEnv, _event_store, event_segments,
                              make_model)

G = GAMMA_STAGE


def test_segments_end_at_the_first_mode_event():
    r = [1.0, 2.0, 3.0, 100.0]
    ev = [False, False, False, True]          # the fourth macro-step starts the entry and runs to the capture
    term = [False, False, False, True]
    segs = event_segments(r, ev, term)
    assert [s[:2] for s in segs] == [(0, 3), (1, 3), (2, 3), (3, 3)]
    k, m, g, t, d = segs[0]
    assert g == pytest.approx(1 + 2 * G + 3 * G ** 2 + 100 * G ** 3) and t and d == pytest.approx(G ** 4)
    assert segs[3][2] == pytest.approx(100.0) and segs[3][4] == pytest.approx(G)   # an event decision: the one-step target


def test_segments_span_and_open_tail():
    r = list(range(1, 8)); ev = [False] * 7; term = [False] * 7
    segs = event_segments(r, ev, term, span=3)
    assert [s[:2] for s in segs] == [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]     # decisions 5, 6: segment still open
    assert all(not s[3] for s in segs) and segs[0][4] == pytest.approx(G ** 3)
    closed = event_segments(r, ev, term, span=3, closed=True)
    assert [s[:2] for s in closed][-2:] == [(5, 6), (6, 6)]


def test_lost_entry_ends_the_segment_without_terminal():
    # entry started in decision 1 and was lost (hold): mode event, not terminal -> bootstrap after decision 1
    segs = event_segments([0.5, 0.2, 0.1], [False, True, False], [False, False, False], closed=True)
    assert [s[:2] for s in segs] == [(0, 1), (1, 1), (2, 2)] and not segs[0][3]


def _model():
    return make_model(SpacesEnv(OBS_DIM_PRIV), 'e_evt', seed=0, smoke=True)


def test_e_evt_model_is_symmetric_and_uses_the_event_replay():
    m = _model()
    assert isinstance(m.replay_buffer, EventReplay)
    assert m.policy.critic_dim == OBS_DIM and m.policy.actor_dim == OBS_DIM


def test_store_sample_and_train_step():
    m = _model(); rng = np.random.default_rng(1); pend = []
    for ep in range(6):
        for k in range(4):
            o, o1 = rng.normal(size=OBS_DIM_PRIV).astype(np.float32), rng.normal(size=OBS_DIM_PRIV).astype(np.float32)
            last = k == 3
            info = {'mode_event': last}
            _event_store(m.replay_buffer, pend, (o, o1, rng.uniform(-1, 1, 2).astype(np.float32), 100.0 if last else -1.0,
                                                 last, last, info))
        assert pend == []                      # every segment closed by the terminal decision
    assert m.replay_buffer.pos == 24
    first = m.replay_buffer.ev_returns[0, 0]
    assert first == pytest.approx(-1 - G - G ** 2 + 100 * G ** 3, rel=1e-5)
    s = m.replay_buffer.sample(8)
    assert isinstance(s, EventSamples) and s.ev_returns.shape == (8, 1) and s.ev_next_observations.shape == (8, OBS_DIM_PRIV)
    m.num_timesteps = 24; m.learning_starts = 0
    from stable_baselines3.common.logger import configure
    m.set_logger(configure(None, [])); m._current_progress_remaining = 1.0
    m.train(gradient_steps=2, batch_size=8)
    assert m._n_updates == 2 and 0 < EVENT_LAMBDA < 1 and EVENT_SPAN >= 2


def test_long_entry_is_off_by_default_and_only_due_at_the_last_pass():
    from env.t2_stage import STAGE_S, T2StageEnv
    env = T2StageEnv.__new__(T2StageEnv)          # the rule only reads the cached opportunity evaluation and the clock
    assert T2StageEnv.__init__.__defaults__ is not None
    env._opp = {'period': 150.0, 'latest': 120.0}

    class _Clock:
        time_s = 40.0
    env.env = _Clock()
    no_event = {'own_event': 0, 'intervened': False}
    assert env._long_entry_due(no_event, {'time_to_pass_s': 10.0})                  # pass imminent, last before 120 s
    assert not env._long_entry_due(no_event, {'time_to_pass_s': STAGE_S + 5.0})     # pass not within this stage
    assert not env._long_entry_due({'own_event': 1, 'intervened': False}, {'time_to_pass_s': 10.0})
    assert not env._long_entry_due({'own_event': 0, 'intervened': True}, {'time_to_pass_s': 10.0})
    env._opp = {'period': 150.0, 'latest': 400.0}
    assert not env._long_entry_due(no_event, {'time_to_pass_s': 10.0})              # another pass remains
