import numpy as np

from dynamics.lie import inverse_transform, make_transform, se3_exp, se3_log, so3_exp
from dynamics.relative import RelativeState, reconstruct_chaser_state, relative_state
from dynamics.types import SpacecraftState


def _state(transform: np.ndarray, twist: np.ndarray) -> SpacecraftState:
    return SpacecraftState(transform[:3, :3], transform[:3, 3], twist[:3], twist[3:])


def test_identical_absolute_states_have_zero_relative_error() -> None:
    state = SpacecraftState(
        so3_exp(np.array([0.2, -0.1, 0.3])),
        np.array([7e6, 2e6, -1e6]),
        np.array([0.1, 0.2, -0.3]),
        np.array([5000.0, -2000.0, 300.0]),
    )
    relative = relative_state(state, state.copy())
    # 约 7000 km 的绝对坐标在 T^-1 T 中发生浮点消去，平移残差约 1e-10 m。
    assert np.allclose(relative.transform, np.eye(4), atol=5e-10)
    assert np.allclose(relative.twist, np.zeros(6), atol=1e-9)


def test_corrected_relative_twist_matches_finite_difference() -> None:
    target_transform = make_transform(
        so3_exp(np.array([0.2, -0.1, 0.3])), np.array([10.0, -4.0, 2.0])
    )
    chaser_transform = make_transform(
        so3_exp(np.array([-0.3, 0.15, 0.2])), np.array([13.0, 1.0, -3.0])
    )
    target_twist = np.array([0.05, 0.2, 0.01, 2.0, -1.0, 0.5])
    chaser_twist = np.array([-0.1, 0.04, 0.2, 1.2, 0.3, -0.7])
    target = _state(target_transform, target_twist)
    chaser = _state(chaser_transform, chaser_twist)
    relative = relative_state(target, chaser)

    dt = 1e-7
    target_next = target_transform @ se3_exp(dt * target_twist)
    chaser_next = chaser_transform @ se3_exp(dt * chaser_twist)
    relative_next = inverse_transform(target_next) @ chaser_next
    body_increment = inverse_transform(relative.transform) @ relative_next
    finite_difference = se3_log(body_increment, project=True) / dt
    assert np.allclose(finite_difference, relative.twist, atol=3e-7)


def test_reconstruct_chaser_state_exactly_inverts_relative_state() -> None:
    target = _state(
        make_transform(so3_exp(np.array([0.1, -0.2, 0.05])), np.array([7e6, 2.0, -3.0])),
        np.array([0.02, -0.03, 0.01, 0.5, 7600.0, -0.4]),
    )
    relative = RelativeState(
        make_transform(so3_exp(np.array([-0.2, 0.1, 0.3])), np.array([-7.0, 1.2, -0.5])),
        np.array([0.01, 0.02, -0.03, -0.2, 0.1, 0.05]),
    )
    reconstructed = reconstruct_chaser_state(target, relative)
    recovered = relative_state(target, reconstructed)
    assert np.allclose(recovered.transform, relative.transform, atol=1e-9)
    assert np.allclose(recovered.twist, relative.twist, atol=1e-10)
