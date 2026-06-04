from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import CameraExtrinsics, CameraIntrinsics, camera_center_world, projection_matrix


class PnPSolveError(ValueError):
    """Raised when OpenCV cannot estimate camera extrinsics from the input points."""


@dataclass(frozen=True)
class PnPResult:
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    extrinsics: CameraExtrinsics
    camera_center_world: np.ndarray
    projection_matrix: np.ndarray
    reprojected_points: np.ndarray
    reprojection_errors: np.ndarray
    reprojection_rmse: float
    reprojection_mean: float
    reprojection_max: float


def normalize_distortion_coefficients(coefficients: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray:
    if coefficients is None:
        return np.zeros((5, 1), dtype=float)

    dist = np.asarray(coefficients, dtype=float).reshape(-1, 1)
    if dist.size == 0:
        return np.zeros((5, 1), dtype=float)
    if dist.size not in {4, 5, 8, 12, 14}:
        raise ValueError("distortion coefficients must contain 4, 5, 8, 12, or 14 values")
    return dist


def _validate_points(object_points: np.ndarray, image_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    object_points = np.asarray(object_points, dtype=float)
    image_points = np.asarray(image_points, dtype=float)

    if object_points.ndim != 2 or object_points.shape[1] != 3:
        raise ValueError("object_points must have shape (N, 3)")
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise ValueError("image_points must have shape (N, 2)")
    if object_points.shape[0] != image_points.shape[0]:
        raise ValueError("object_points and image_points must contain the same number of points")
    if object_points.shape[0] < 4:
        raise ValueError("solvePnP requires at least 4 point correspondences")

    return object_points, image_points


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion_coefficients: np.ndarray | list[float] | tuple[float, ...] | None = None,
    flags: int = cv2.SOLVEPNP_ITERATIVE,
) -> PnPResult:
    """Estimate camera extrinsics and reprojection error with Perspective-n-Point.

    Inputs:

    - `object_points`: known points in World/Object Space.
    - `image_points`: corresponding 2D points in Image/Viewport pixel space.
    - `K`: intrinsic matrix from `CameraIntrinsics`.

    OpenCV solves the world-to-camera transform:

        X_camera = R * X_world + t

    Then reprojection applies the full graphics/CV pipeline:

        World Space -> Camera Space -> normalized image plane -> pixel viewport

    The final error compares original image points with the same 3D points
    projected back through the estimated camera.
    """

    object_points, image_points = _validate_points(object_points, image_points)
    K = intrinsics.matrix()
    dist = normalize_distortion_coefficients(distortion_coefficients)

    success, rvec, tvec = cv2.solvePnP(
        object_points.reshape(-1, 1, 3),
        image_points.reshape(-1, 1, 2),
        K,
        dist,
        flags=flags,
    )
    if not success:
        raise PnPSolveError("OpenCV solvePnP failed for the provided point correspondences")

    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    extrinsics = CameraExtrinsics(R=R, t=t)
    center = camera_center_world(extrinsics)
    P = projection_matrix(intrinsics, extrinsics)

    reprojected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        rvec,
        tvec,
        K,
        dist,
    )
    reprojected_points = reprojected.reshape(-1, 2)
    deltas = image_points - reprojected_points
    per_point_errors = np.linalg.norm(deltas, axis=1)
    rmse = float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))

    return PnPResult(
        rvec=rvec.reshape(3),
        tvec=t,
        R=R,
        extrinsics=extrinsics,
        camera_center_world=center,
        projection_matrix=P,
        reprojected_points=reprojected_points,
        reprojection_errors=per_point_errors,
        reprojection_rmse=rmse,
        reprojection_mean=float(np.mean(per_point_errors)),
        reprojection_max=float(np.max(per_point_errors)),
    )

