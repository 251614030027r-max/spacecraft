"""动力学层的显式数据结构，定义动力学层中最基本的三类数据对象：航天器状态、航天器物理参数、六维广义力。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _vector3(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} 必须是 shape=(3,)；实际为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含有非有限数")
    return array.copy()


def _matrix3(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} 必须是 shape=(3,3)；实际为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含有非有限数")
    return array.copy()


@dataclass
class SpacecraftState:
    """单航天器绝对状态；速度均在本体系表达，位置在 ECI 表达。"""

    rotation: FloatArray
    position: FloatArray
    omega: FloatArray
    velocity: FloatArray

    def __post_init__(self) -> None:
        self.rotation = _matrix3(self.rotation, "rotation")
        self.position = _vector3(self.position, "position")
        self.omega = _vector3(self.omega, "omega")
        self.velocity = _vector3(self.velocity, "velocity")

    @property
    def twist(self) -> FloatArray:
        return np.concatenate((self.omega, self.velocity))

    def copy(self) -> "SpacecraftState":
        return SpacecraftState(
            self.rotation.copy(),
            self.position.copy(),
            self.omega.copy(),
            self.velocity.copy(),
        )


@dataclass
class SpacecraftParameters:
    mass: float
    inertia: FloatArray

    def __post_init__(self) -> None:
        self.mass = float(self.mass)
        self.inertia = _matrix3(self.inertia, "inertia")
        if not np.isfinite(self.mass) or self.mass <= 0.0:
            raise ValueError("mass 必须为有限正数")
        if not np.allclose(self.inertia, self.inertia.T, atol=1e-12):
            raise ValueError("inertia 必须对称")
        if np.min(np.linalg.eigvalsh(self.inertia)) <= 0.0:
            raise ValueError("inertia 必须正定")

    @property
    def generalized_inertia(self) -> FloatArray:
        result = np.zeros((6, 6), dtype=np.float64)
        result[:3, :3] = self.inertia
        result[3:, 3:] = self.mass * np.eye(3)
        return result


@dataclass
class GeneralizedForce:
    """顺序固定为 [torque; force]，两者都在本体系表达。"""

    torque: FloatArray
    force: FloatArray

    def __post_init__(self) -> None:
        self.torque = _vector3(self.torque, "torque")
        self.force = _vector3(self.force, "force")

    @classmethod
    def zeros(cls) -> "GeneralizedForce":
        return cls(np.zeros(3), np.zeros(3))

    @classmethod
    def from_vector(cls, value: ArrayLike) -> "GeneralizedForce":
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (6,):
            raise ValueError("广义力必须是 shape=(6,)，顺序为 [torque; force]")
        return cls(array[:3], array[3:])

    @property
    def vector(self) -> FloatArray:
        return np.concatenate((self.torque, self.force))

    def __add__(self, other: "GeneralizedForce") -> "GeneralizedForce":
        if not isinstance(other, GeneralizedForce):
            return NotImplemented
        return GeneralizedForce(self.torque + other.torque, self.force + other.force)
