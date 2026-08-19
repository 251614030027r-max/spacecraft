import numpy as np
import pytest

from dynamics.lie import (
    adjoint,
    hat3,
    hat6,
    inverse_transform,
    is_so3,
    left_jacobian_so3,
    left_jacobian_so3_inv,
    make_transform,
    project_to_so3,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
    vee3,
    vee6,
)


def test_hat_vee_roundtrip_and_cross_product() -> None:
    vector = np.array([0.3, -0.4, 1.2])
    other = np.array([-0.1, 2.0, 0.7])
    assert np.allclose(vee3(hat3(vector)), vector)
    assert np.allclose(hat3(vector) @ other, np.cross(vector, other))
    twist = np.array([0.1, -0.2, 0.3, 4.0, -5.0, 6.0])
    assert np.allclose(vee6(hat6(twist)), twist)


@pytest.mark.parametrize(
    "rotation_vector",
    [
        np.zeros(3),
        np.array([1e-10, -2e-10, 3e-10]),
        np.array([0.2, -0.3, 0.4]),
        (np.pi - 1e-7) * np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0),
    ],
)
def test_so3_exp_log_roundtrip(rotation_vector: np.ndarray) -> None:
    rotation = so3_exp(rotation_vector)
    recovered = so3_log(rotation)
    assert is_so3(rotation, atol=1e-12)
    assert np.allclose(so3_exp(recovered), rotation, atol=2e-8)


def test_left_jacobian_inverse() -> None:
    for phi in (np.zeros(3), np.array([0.2, -0.4, 0.6]), np.array([2.8, 0.1, -0.2])):
        product = left_jacobian_so3(phi) @ left_jacobian_so3_inv(phi)
        assert np.allclose(product, np.eye(3), atol=1e-10)


def test_se3_exp_log_roundtrip() -> None:
    eta = np.array([0.4, -0.2, 0.1, 12.0, -4.0, 2.0])
    transform = se3_exp(eta)
    recovered = se3_log(transform)
    assert np.allclose(recovered, eta, atol=1e-10)


def test_adjoint_conjugation_identity() -> None:
    transform = make_transform(
        so3_exp(np.array([0.3, -0.2, 0.1])),
        np.array([4.0, -2.0, 1.0]),
    )
    twist = np.array([0.1, 0.2, -0.3, 1.0, 2.0, -1.0])
    left = hat6(adjoint(transform) @ twist)
    right = transform @ hat6(twist) @ inverse_transform(transform)
    assert np.allclose(left, right, atol=1e-12)


def test_projection_restores_rotation_matrix() -> None:
    rotation = so3_exp(np.array([0.2, 0.4, -0.1]))
    perturbed = rotation.copy()
    perturbed[0, 0] += 2e-5
    projected = project_to_so3(perturbed)
    assert is_so3(projected, atol=1e-12)
    assert np.linalg.det(projected) == pytest.approx(1.0, abs=1e-12)
