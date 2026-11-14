"""Aggregate camera-geometry acceptance test for A1."""

import numpy as np

from dynamics.lie import make_transform, so3_exp
from dynamics.relative import RelativeState
from env.perception import PerceptionConfig, visible_feature_projection


def test_a1_camera_projection_and_visibility_boundaries() -> None:
    config = PerceptionConfig()
    relative = RelativeState(
        make_transform(np.eye(3), np.array([-3.0, 0.0, 0.0])),
        np.zeros(6),
    )
    projection = visible_feature_projection(relative, config)
    assert projection.feature_indices.tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(
        projection.image_points_px,
        np.array(
            [
                [426.0, 598.0],
                [426.0, 426.0],
                [598.0, 598.0],
                [598.0, 426.0],
                [512.0 + 430.0 * 0.18 / 1.30, 512.0 + 430.0 * 0.10 / 1.30],
            ]
        ),
        atol=1.0e-12,
    )

    back_facing = RelativeState(
        make_transform(so3_exp([0.0, 0.0, np.pi]), np.zeros(3)),
        np.zeros(6),
    )
    outside_cone = RelativeState(
        make_transform(np.eye(3), np.array([-3.0, 3.0, 0.0])),
        np.zeros(6),
    )
    outside_range = RelativeState(
        make_transform(np.eye(3), np.array([-31.0, 0.0, 0.0])),
        np.zeros(6),
    )
    assert visible_feature_projection(back_facing, config).visible_count == 0
    assert visible_feature_projection(outside_cone, config).visible_count == 0
    assert visible_feature_projection(outside_range, config).visible_count == 0
