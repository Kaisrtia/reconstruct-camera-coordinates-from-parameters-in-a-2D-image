import unittest

import numpy as np

from camera_reconstruction.camera_model import (
    CameraExtrinsics,
    CameraIntrinsics,
    camera_center_world,
    rotation_matrix_from_euler,
)
from camera_reconstruction.projection import project_points
from camera_reconstruction.reconstruction import backproject_ray, pixel_to_camera_point


class SmokeTests(unittest.TestCase):
    def test_project_shapes(self) -> None:
        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        R = rotation_matrix_from_euler(0.0, 0.0, 0.0)
        t = np.array([0.0, 0.0, 3.0])
        extrinsics = CameraExtrinsics(R=R, t=t)

        points_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
        points_2d = project_points(points_3d, intrinsics, extrinsics)

        self.assertEqual(points_2d.shape, (2, 2))

    def test_backproject_ray(self) -> None:
        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
        R = rotation_matrix_from_euler(0.0, 0.0, 0.0)
        t = np.array([0.0, 0.0, 3.0])
        extrinsics = CameraExtrinsics(R=R, t=t)

        origin, direction = backproject_ray((320.0, 240.0), intrinsics, extrinsics)

        self.assertEqual(origin.shape, (3,))
        self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=6)

    def test_camera_center(self) -> None:
        R = rotation_matrix_from_euler(0.0, 0.0, 0.0)
        t = np.array([0.0, 0.0, 3.0])
        extrinsics = CameraExtrinsics(R=R, t=t)

        center = camera_center_world(extrinsics)
        np.testing.assert_allclose(center, np.array([0.0, 0.0, -3.0]))

    def test_pixel_to_camera_point(self) -> None:
        intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)

        point = pixel_to_camera_point((320.0, 240.0), 3.0, intrinsics)

        np.testing.assert_allclose(point, np.array([0.0, 0.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
