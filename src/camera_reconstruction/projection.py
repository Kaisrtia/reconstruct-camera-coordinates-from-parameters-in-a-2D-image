from __future__ import annotations

import numpy as np

from .camera_model import CameraExtrinsics, CameraIntrinsics


def project_points(
    points_3d: np.ndarray,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> np.ndarray:
    points_3d = np.asarray(points_3d, dtype=float)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")

    R = np.asarray(extrinsics.R, dtype=float)
    t = np.asarray(extrinsics.t, dtype=float).reshape(3, 1)
    K = intrinsics.matrix()

    camera_points = (R @ points_3d.T) + t
    uvw = (K @ camera_points).T
    zs = uvw[:, 2:3]
    if np.any(zs == 0):
        raise ValueError("points project to infinity (z == 0)")

    return uvw[:, :2] / zs


def project_points_with_P(points_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    points_3d = np.asarray(points_3d, dtype=float)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError("points_3d must have shape (N, 3)")

    P = np.asarray(P, dtype=float)
    if P.shape != (3, 4):
        raise ValueError("P must have shape (3, 4)")

    points_h = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    uvw = (P @ points_h.T).T
    zs = uvw[:, 2:3]
    if np.any(zs == 0):
        raise ValueError("points project to infinity (z == 0)")

    return uvw[:, :2] / zs
