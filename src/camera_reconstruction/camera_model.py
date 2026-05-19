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
    R: np.ndarray
    t: np.ndarray

    def matrix(self) -> np.ndarray:
        R = _ensure_shape("R", self.R, (3, 3))
        t = _ensure_shape("t", self.t, (3,))
        return np.hstack([R, t.reshape(3, 1)])


def projection_matrix(intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics) -> np.ndarray:
    return intrinsics.matrix() @ extrinsics.matrix()


def camera_center_world(extrinsics: CameraExtrinsics) -> np.ndarray:
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
