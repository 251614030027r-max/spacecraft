import json

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from env.t2_stage import OBS_DIM, OBS_DIMS
from experiments.t2 import load_checkpoint
from train.t2_mve_sac import SpacesEnv, make_model, td3_exploration_action


@pytest.fixture
def model():
    return make_model(SpacesEnv(OBS_DIMS["cut_priv"]), "e_td3", 10, smoke=True)


def test_td3_exploration_is_independent_and_reproducible(model):
    obs = np.zeros(OBS_DIMS["cut_priv"], dtype=np.float32)
    first = np.random.default_rng([10, 3003])
    second = np.random.default_rng([10, 3003])
    actions = np.stack([td3_exploration_action(model, obs, first) for _ in range(8)])
    np.testing.assert_array_equal(actions, [td3_exploration_action(model, obs, second) for _ in range(8)])
    assert np.any(actions != model.predict(obs, deterministic=True)[0])
    assert np.all(np.abs(actions) <= 1)


def test_td3_checkpoint_is_deterministic_and_hides_execution_features(model, tmp_path):
    obs = np.zeros(OBS_DIMS["cut_priv"], dtype=np.float32)
    other = obs.copy()
    other[OBS_DIM:] = 100
    path = tmp_path / "final.zip"
    model.save(path)
    (tmp_path / "manifest.json").write_text(json.dumps({"arm": "e_td3"}), encoding="utf-8")
    loaded = load_checkpoint(path)
    action = loaded.predict(obs, deterministic=True)[0]
    np.testing.assert_array_equal(action, model.predict(obs, deterministic=True)[0])
    np.testing.assert_array_equal(action, loaded.predict(other, deterministic=True)[0])
    assert loaded.actor.features_extractor.features_dim == OBS_DIM
    assert loaded.critic.features_extractor.features_dim == OBS_DIM


def test_td3_updates_the_actor_every_second_critic_update(model):
    model.set_logger(configure(None, []))
    obs = np.ones((1, OBS_DIMS["cut_priv"]), dtype=np.float32)
    model.replay_buffer.add(obs, obs, np.zeros((1, 2), dtype=np.float32),
                            np.ones(1), np.zeros(1), [{}])
    assert model.gradient_steps == 2
    initial = [p.detach().clone() for p in model.actor.parameters()]
    model.train(gradient_steps=1, batch_size=1)
    assert model._n_updates == 1
    assert all(torch.equal(a, b) for a, b in zip(initial, model.actor.parameters()))
    model.train(gradient_steps=1, batch_size=1)
    assert model._n_updates == 2
    assert any(not torch.equal(a, b) for a, b in zip(initial, model.actor.parameters()))
