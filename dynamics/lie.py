"""SO(3)/SE(3) operations with twist order ``xi=[omega; v]``."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
_SMALL_ANGLE = 1e-8
_NEAR_PI = 1e-6


def _as_vector(value: ArrayLike, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} 必须为 shape=({size},)，实际为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含有非有限数")
    return array


def hat3(vector: ArrayLike) -> FloatArray:
    x, y, z = _as_vector(vector, 3, "vector")
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def vee3(matrix: ArrayLike) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("matrix 必须为 shape=(3,3)")
    return np.array([value[2, 1], value[0, 2], value[1, 0]], dtype=np.float64)


def hat6(twist: ArrayLike) -> FloatArray:
    xi = _as_vector(twist, 6, "twist")
    result = np.zeros((4, 4), dtype=np.float64)
    result[:3, :3] = hat3(xi[:3])
    result[:3, 3] = xi[3:]
    return result


def vee6(matrix: ArrayLike) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("matrix 必须为 shape=(4,4)")
    return np.concatenate((vee3(value[:3, :3]), value[:3, 3]))


def project_to_so3(matrix: ArrayLike) -> FloatArray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("matrix 必须为 shape=(3,3)")
    u, _, vt = np.linalg.svd(value)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def is_so3(matrix: ArrayLike, atol: float = 1e-9) -> bool:
    value = np.asarray(matrix, dtype=np.float64)
    return bool(
        value.shape == (3, 3)
        and np.allclose(value.T @ value, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(value), 1.0, atol=atol)
    )


def so3_exp(rotation_vector: ArrayLike) -> FloatArray:
    phi = _as_vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(phi))
    phi_hat = hat3(phi)
    if theta < _SMALL_ANGLE:
        theta2 = theta * theta
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * phi_hat + b * (phi_hat @ phi_hat)


def _axis_near_pi(rotation: FloatArray) -> FloatArray:
    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    axis = np.zeros(3, dtype=np.float64)
    value = max(0.0, 0.5 * (diagonal[index] + 1.0))
    axis[index] = np.sqrt(value)
    if axis[index] < 1e-8:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
    else:
        j, k = [item for item in range(3) if item != index]
        axis[j] = (rotation[index, j] + rotation[j, index]) / (4.0 * axis[index])
        axis[k] = (rotation[index, k] + rotation[k, index]) / (4.0 * axis[index])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("接近 pi 的旋转轴提取失败")
    axis /= norm
    skew_hint = vee3(rotation - rotation.T)
    if np.linalg.norm(skew_hint) > 1e-10 and float(axis @ skew_hint) < 0.0:
        axis *= -1.0
    return axis


def so3_log(rotation: ArrayLike, *, project: bool = False) -> FloatArray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("rotation 必须为 shape=(3,3)")
    if project:
        value = project_to_so3(value)
    elif not is_so3(value, atol=1e-7):
        raise ValueError("rotation 不是有效 SO(3) 矩阵")
    cosine = float(np.clip(0.5 * (np.trace(value) - 1.0), -1.0, 1.0))
    theta = float(np.arccos(cosine))
    if theta < _SMALL_ANGLE:
        return 0.5 * vee3(value - value.T)
    if np.pi - theta < _NEAR_PI:
        return theta * _axis_near_pi(value)
    return theta / (2.0 * np.sin(theta)) * vee3(value - value.T)


def left_jacobian_so3(rotation_vector: ArrayLike) -> FloatArray:
    phi = _as_vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(phi))
    phi_hat = hat3(phi)
    if theta < _SMALL_ANGLE:
        theta2 = theta * theta
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
        c = 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0
    else:
        b = (1.0 - np.cos(theta)) / (theta * theta)
        c = (theta - np.sin(theta)) / (theta * theta * theta)
    return np.eye(3) + b * phi_hat + c * (phi_hat @ phi_hat)


def left_jacobian_so3_inv(rotation_vector: ArrayLike) -> FloatArray:
    phi = _as_vector(rotation_vector, 3, "rotation_vector")
    theta = float(np.linalg.norm(phi))
    phi_hat = hat3(phi)
    if theta < _SMALL_ANGLE:
        coefficient = 1.0 / 12.0 + theta * theta / 720.0
    else:
        half = 0.5 * theta
        coefficient = 1.0 / (theta * theta) - np.cos(half) / (
            2.0 * theta * np.sin(half)
        )
    return np.eye(3) - 0.5 * phi_hat + coefficient * (phi_hat @ phi_hat)


def make_transform(rotation: ArrayLike, position: ArrayLike) -> FloatArray:
    r = np.asarray(rotation, dtype=np.float64)
    p = _as_vector(position, 3, "position")
    if r.shape != (3, 3):
        raise ValueError("rotation 必须为 shape=(3,3)")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = r
    result[:3, 3] = p
    return result


def split_transform(transform: ArrayLike) -> tuple[FloatArray, FloatArray]:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("transform 必须为 shape=(4,4)")
    return value[:3, :3].copy(), value[:3, 3].copy()


def inverse_transform(transform: ArrayLike) -> FloatArray:
    rotation, position = split_transform(transform)
    return make_transform(rotation.T, -rotation.T @ position)


def se3_exp(exponential_coordinates: ArrayLike) -> FloatArray:
    eta = _as_vector(exponential_coordinates, 6, "exponential_coordinates")
    phi, rho = eta[:3], eta[3:]
    return make_transform(so3_exp(phi), left_jacobian_so3(phi) @ rho)


def se3_log(transform: ArrayLike, *, project: bool = False) -> FloatArray:
    rotation, position = split_transform(transform)
    phi = so3_log(rotation, project=project)
    rho = left_jacobian_so3_inv(phi) @ position
    return np.concatenate((phi, rho))


def adjoint(transform: ArrayLike) -> FloatArray:
    rotation, position = split_transform(transform)
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = rotation
    result[3:, :3] = hat3(position) @ rotation
    result[3:, 3:] = rotation
    return result


def ad6(twist: ArrayLike) -> FloatArray:
    xi = _as_vector(twist, 6, "twist")
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = hat3(xi[:3])
    result[3:, :3] = hat3(xi[3:])
    result[3:, 3:] = hat3(xi[:3])
    return result
