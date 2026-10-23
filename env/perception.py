"""Geometric camera measurements for the Phase-2 perception sub-task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dynamics.relative import RelativeState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DEFAULT_FEATURE_POINTS_TARGET_M = (
    (-1.50, -0.30, -0.30),
    (-1.50, -0.30, +0.30),
    (-1.50, +0.30, -0.30),
    (-1.50, +0.30, +0.30),
    (-1.70, +0.18, -0.10),
)


@dataclass(frozen=True)
class PerceptionConfig:
    """Fixed camera and target-feature specification for A1."""

    image_width_px: int = 1024
    image_height_px: int = 1024
    principal_point_u_px: float = 512.0
    principal_point_v_px: float = 512.0
    focal_length_px: float = 430.0
    fov_half_angle_rad: float = float(np.deg2rad(50.0))
    max_measurement_distance_m: float = 30.0
    pixel_noise_std: float = 1.0
    feature_points_target_m: tuple[tuple[float, float, float], ...] = (
        DEFAULT_FEATURE_POINTS_TARGET_M
    )

    def __post_init__(self) -> None:
        values = (
            self.principal_point_u_px,
            self.principal_point_v_px,
            self.focal_length_px,
            self.fov_half_angle_rad,
            self.max_measurement_distance_m,
            self.pixel_noise_std,
        )
        points = np.asarray(self.feature_points_target_m, dtype=np.float64)
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if min(values) <= 0.0 or not np.all(np.isfinite(values)):
            raise ValueError("camera scales must be finite and positive")
        if not 0.0 < self.fov_half_angle_rad < 0.5 * np.pi:
            raise ValueError("camera half field of view must lie in (0, pi/2)")
        if points.shape != (5, 3) or not np.all(np.isfinite(points)):
            raise ValueError("A1 requires five finite target-frame feature points")

    @property
    def feature_points(self) -> FloatArray:
        return np.asarray(self.feature_points_target_m, dtype=np.float64)


@dataclass(frozen=True)
class FeatureMeasurement:
    """Visible feature identities and their noisy image coordinates."""

    feature_indices: IntArray
    image_points_px: FloatArray

    def __post_init__(self) -> None:
        indices = np.asarray(self.feature_indices, dtype=np.int64)
        pixels = np.asarray(self.image_points_px, dtype=np.float64)
        if indices.ndim != 1 or pixels.shape != (indices.size, 2):
            raise ValueError("feature measurement shapes are inconsistent")
        if not np.all(np.isfinite(pixels)):
            raise ValueError("feature pixels must be finite")
        object.__setattr__(self, "feature_indices", indices.copy())
        object.__setattr__(self, "image_points_px", pixels.copy())

    @property
    def visible_count(self) -> int:
        return int(self.feature_indices.size)


def project_feature_points(
    relative: RelativeState,
    feature_indices: IntArray,
    config: PerceptionConfig,
) -> FloatArray:
    """Project selected known target-frame features into the chaser camera."""

    indices = np.asarray(feature_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= 5):
        raise ValueError("feature indices are invalid")
    features_target = config.feature_points[indices]
    points_camera = (relative.rotation.T @ (
        features_target - relative.position
    ).T).T
    depth = points_camera[:, 0]
    if np.any(depth <= 0.0):
        raise ValueError("selected feature lies behind the camera")
    pixels = np.empty((indices.size, 2), dtype=np.float64)
    pixels[:, 0] = (
        config.principal_point_u_px
        + config.focal_length_px * points_camera[:, 1] / depth
    )
    pixels[:, 1] = (
        config.principal_point_v_px
        - config.focal_length_px * points_camera[:, 2] / depth
    )
    return pixels


def visible_feature_projection(
    relative: RelativeState,
    config: PerceptionConfig,
) -> FeatureMeasurement:
    """Return noise-free pixels for features satisfying every A1 visibility test."""

    features_target = config.feature_points
    points_camera = (relative.rotation.T @ (
        features_target - relative.position
    ).T).T
    depth = points_camera[:, 0]
    safe_depth = np.where(depth > 0.0, depth, 1.0)
    pixels = np.empty((5, 2), dtype=np.float64)
    pixels[:, 0] = (
        config.principal_point_u_px
        + config.focal_length_px * points_camera[:, 1] / safe_depth
    )
    pixels[:, 1] = (
        config.principal_point_v_px
        - config.focal_length_px * points_camera[:, 2] / safe_depth
    )
    ranges = np.linalg.norm(points_camera, axis=1)
    cone_cosine = depth / np.maximum(ranges, np.finfo(np.float64).eps)
    feature_to_camera_target = relative.position - features_target
    port_normal_target = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    front_facing = feature_to_camera_target @ port_normal_target > 0.0
    center_in_range = (
        float(np.linalg.norm(relative.position))
        <= config.max_measurement_distance_m
    )
    visible = (
        (depth > 0.0)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < config.image_width_px)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < config.image_height_px)
        & (cone_cosine >= np.cos(config.fov_half_angle_rad))
        & front_facing
        & center_in_range
    )
    indices = np.flatnonzero(visible).astype(np.int64)
    return FeatureMeasurement(indices, pixels[indices])


def measure_visible_features(
    relative: RelativeState,
    config: PerceptionConfig,
    rng: np.random.Generator,
) -> FeatureMeasurement:
    """Generate reproducible noisy pixels from the truth relative pose."""

    projection = visible_feature_projection(relative, config)
    noise = rng.normal(
        0.0, config.pixel_noise_std, size=projection.image_points_px.shape
    )
    return FeatureMeasurement(
        projection.feature_indices,
        projection.image_points_px + noise,
    )
