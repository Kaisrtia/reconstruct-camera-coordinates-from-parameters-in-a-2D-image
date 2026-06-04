from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ensure_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters.

    The matrix K converts normalized camera coordinates into image/viewport
    pixel coordinates:

        [u, v, 1]^T ~ K [Xc/Zc, Yc/Zc, 1]^T

    `fx` and `fy` scale camera-plane units into pixels, while `cx` and `cy`
    move the origin from the optical axis to the image coordinate system whose
    top-left corner is pixel (0, 0).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0

    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, self.skew, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """Camera pose represented as the world-to-camera transform.

    OpenCV uses:

        X_camera = R * X_world + t

    Because `t` is part of the world-to-camera transform, it is not the camera
    position in world coordinates. The camera center is recovered with:

        C_world = -R^T * t
    """

    R: np.ndarray
    t: np.ndarray

    def matrix(self) -> np.ndarray:
        R = _ensure_shape("R", self.R, (3, 3))
        t = _ensure_shape("t", self.t, (3,))
        return np.hstack([R, t.reshape(3, 1)])


def projection_matrix(intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics) -> np.ndarray:
    """Return P = K [R | t], the full world-to-image projection matrix."""

    return intrinsics.matrix() @ extrinsics.matrix()


def camera_center_world(extrinsics: CameraExtrinsics) -> np.ndarray:
    """Return the camera center expressed in world coordinates."""

    R = _ensure_shape("R", extrinsics.R, (3, 3))
    t = _ensure_shape("t", extrinsics.t, (3,))
    return -R.T @ t


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    return Rz @ Ry @ Rx

