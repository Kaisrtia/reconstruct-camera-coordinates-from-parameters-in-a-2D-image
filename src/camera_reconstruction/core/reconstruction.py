from __future__ import annotations

import numpy as np

from .camera_model import CameraExtrinsics, CameraIntrinsics, camera_center_world


def backproject_ray(
    pixel: np.ndarray | tuple[float, float],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a 2D pixel into a 3D world-space ray.

    A single 2D pixel cannot identify one unique 3D point. It defines a ray.

    The coordinate flow is:

    1. Image/Viewport pixel -> normalized camera ray:
       ray_camera = K^-1 [u, v, 1]^T
    2. Camera Space -> World Space direction:
       ray_world = R^T * ray_camera
    3. Ray origin is the camera center:
       C_world = -R^T * t
    """

    pixel = np.asarray(pixel, dtype=float).reshape(2,)
    K = intrinsics.matrix()
    K_inv = np.linalg.inv(K)
    R = np.asarray(extrinsics.R, dtype=float)

    ray_camera = K_inv @ np.array([pixel[0], pixel[1], 1.0])
    direction_world = R.T @ ray_camera
    direction_world = direction_world / np.linalg.norm(direction_world)

    origin_world = camera_center_world(extrinsics)
    return origin_world, direction_world


def pixel_to_camera_point(
    pixel: np.ndarray | tuple[float, float],
    depth: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Convert a pixel into one camera-space point when depth is known.

    With known camera-space depth Zc, the normalized camera ray can be scaled:

        [Xc, Yc, Zc]^T = Zc * K^-1 [u, v, 1]^T

    Without this depth, the result is only a ray, not a single point.
    """

    if depth <= 0:
        raise ValueError("depth must be positive")

    pixel = np.asarray(pixel, dtype=float).reshape(2,)
    K_inv = np.linalg.inv(intrinsics.matrix())
    ray_camera = K_inv @ np.array([pixel[0], pixel[1], 1.0])
    return ray_camera * depth

