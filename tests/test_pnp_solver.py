import unittest

import numpy as np

from camera_reconstruction.core.camera_model import (
    CameraExtrinsics,
    CameraIntrinsics,
    camera_center_world,
    rotation_matrix_from_euler,
)
from camera_reconstruction.core.checkerboard import CheckerboardConfig, build_checkerboard_object_points
from camera_reconstruction.core.pnp_solver import normalize_distortion_coefficients, solve_pnp
from camera_reconstruction.core.projection import project_points


class PnPSolverTests(unittest.TestCase):
    def test_solve_pnp_recovers_synthetic_pose(self) -> None:
        intrinsics = CameraIntrinsics(fx=920.0, fy=910.0, cx=320.0, cy=240.0)
        R = rotation_matrix_from_euler(np.deg2rad(8.0), np.deg2rad(-12.0), np.deg2rad(18.0))
        t = np.array([0.15, -0.08, 4.2])
        extrinsics = CameraExtrinsics(R=R, t=t)

        object_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.2],
                [0.5, 0.4, 0.8],
                [1.4, 0.2, 0.5],
                [0.3, 1.3, 0.4],
                [1.2, 1.1, 0.9],
            ],
            dtype=np.float32,
        )
        image_points = project_points(object_points, intrinsics, extrinsics)

        result = solve_pnp(object_points, image_points, intrinsics)

        self.assertLess(result.reprojection_rmse, 1e-6)
        np.testing.assert_allclose(result.R, R, atol=1e-6)
        np.testing.assert_allclose(result.tvec, t, atol=1e-6)
        np.testing.assert_allclose(result.camera_center_world, camera_center_world(extrinsics), atol=1e-6)

    def test_checkerboard_object_points_shape(self) -> None:
        points = build_checkerboard_object_points(CheckerboardConfig(corners_x=9, corners_y=6, square_size=2.5))

        self.assertEqual(points.shape, (54, 3))
        np.testing.assert_allclose(points[0], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(points[1], np.array([2.5, 0.0, 0.0]))
        np.testing.assert_allclose(points[9], np.array([0.0, 2.5, 0.0]))

    def test_distortion_defaults_to_five_zero_coefficients(self) -> None:
        dist = normalize_distortion_coefficients(None)

        self.assertEqual(dist.shape, (5, 1))
        np.testing.assert_allclose(dist, np.zeros((5, 1)))


if __name__ == "__main__":
    unittest.main()

