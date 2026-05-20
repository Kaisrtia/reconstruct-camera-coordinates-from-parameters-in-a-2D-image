from __future__ import annotations

import numpy as np

from .camera_model import CameraExtrinsics, CameraIntrinsics, camera_center_world


def backproject_ray(
    pixel: np.ndarray | tuple[float, float],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> tuple[np.ndarray, np.ndarray]:
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
    if depth <= 0:
        raise ValueError("depth must be positive")

    pixel = np.asarray(pixel, dtype=float).reshape(2,)
    K_inv = np.linalg.inv(intrinsics.matrix())
    ray_camera = K_inv @ np.array([pixel[0], pixel[1], 1.0])
    return ray_camera * depth
