"""V3 value heads and the one-way arbiter (M3 / M4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.hybrid_env import PrecaptureHybridEnv
from train.train_hybrid import accelerated_training_configs
from train.v3_values import (
    ValueEnsemble,
    baseline_mask,
    discounted_returns,
    fit_ensemble,
    one_way_rule,
)


def _v3_env() -> PrecaptureHybridEnv:
    environment_config, hybrid_config = accelerated_training_configs(
        horizon_steps=35,
        waypoint_parametrization="task_state_v3",
        execution_feedback=True,
        adaptive_task=True,
    )
    return PrecaptureHybridEnv(environment_config, hybrid_config)


def test_discounted_returns_run_to_the_end_of_the_episode() -> None:
    returns = discounted_returns([1.0, 0.0, 2.0], gamma=0.5)
    np.testing.assert_allclose(returns, [1.0 + 0.5 * 0.0 + 0.25 * 2.0, 0.0 + 0.5 * 2.0, 2.0])


def test_observation_slices_tile_the_v3_observation_and_mask_the_private_state() -> None:
    env = _v3_env()
    observation, _ = env.reset(seed=263000)
    slices = env.policy_observation_slices()
    ordered = sorted(slices.values(), key=lambda s: s.start)
    assert ordered[0].start == 0 and ordered[-1].stop == observation.size == 42
    assert all(a.stop == b.start for a, b in zip(ordered, ordered[1:]))
    # the blocks hold what their names say
    np.testing.assert_allclose(observation[slices["task_state"]], [0.0, 0.0])
    np.testing.assert_allclose(
        observation[slices["applied_direction"]], env._v3_observation_direction(), atol=1e-6
    )
    mask = baseline_mask(slices, observation.size)
    assert mask[slices["task_state"]].sum() == 0 and mask[slices["applied_direction"]].sum() == 0
    assert mask.sum() == observation.size - 5
    env.close()


@pytest.mark.parametrize(
    ("branch", "mu_l", "sd_l", "mu_b", "sd_b", "expected"),
    [
        (None, 10.0, 1.0, 5.0, 1.0, "learned"),  # confidently better
        (None, 10.0, 3.0, 5.0, 3.0, "baseline"),  # better but not confidently
        ("learned", 5.0, 1.0, 10.0, 1.0, "baseline"),  # confident handback
        ("learned", 5.0, 3.0, 10.0, 3.0, "learned"),  # unsure: stay
        ("baseline", 100.0, 0.0, -100.0, 0.0, "baseline"),  # never back to learned
    ],
)
def test_one_way_rule(branch, mu_l, sd_l, mu_b, sd_b, expected) -> None:
    assert one_way_rule(branch, mu_l, sd_l, mu_b, sd_b, z=1.0) == expected


def test_ensemble_fits_a_known_function_ignores_masked_inputs_and_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    episodes = np.repeat(np.arange(20), 30)
    x = rng.normal(size=(episodes.size, 4)).astype(np.float32)
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0
    mask = np.array([1, 1, 1, 0], dtype=np.float32)
    model = fit_ensemble(x, y, episodes, input_mask=mask, seed=1, heads=3, max_epochs=200, patience=20)
    mu = model.head_values(x).mean(axis=0)
    assert np.mean(np.abs(mu - y)) < 0.3
    # a masked input cannot move the prediction
    shifted = x.copy()
    shifted[:, 3] += 100.0
    np.testing.assert_allclose(model.head_values(shifted), model.head_values(x))
    model.save(tmp_path / "v.pt")
    loaded = ValueEnsemble.load(tmp_path / "v.pt")
    np.testing.assert_allclose(loaded.head_values(x[:5]), model.head_values(x[:5]), rtol=0, atol=1e-6)
    assert len(model.meta["heads"]) == 3
