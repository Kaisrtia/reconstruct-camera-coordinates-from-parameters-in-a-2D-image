from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import CameraIntrinsics
from .pnp_solver import normalize_distortion_coefficients


@dataclass(frozen=True)
class AxisOverlay:
    image_bgr: np.ndarray
    axis_image_points: np.ndarray


def _point_tuple(point: np.ndarray) -> tuple[int, int]:
    rounded = np.rint(point).astype(int)
    return (int(rounded[0]), int(rounded[1]))


def _draw_label(image: np.ndarray, text: str, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    anchor = (x + 7, y - 7)
    cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def draw_axes_overlay(
    image_bgr: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    distortion_coefficients: np.ndarray | list[float] | tuple[float, ...] | None = None,
    axis_length: float = 1.0,
    image_points: np.ndarray | None = None,
    reprojected_points: np.ndarray | None = None,
) -> AxisOverlay:
    """Draw 3D coordinate axes after projecting them onto the 2D image.

    The axis endpoints are defined in checkerboard World/Object Space. OpenCV
    projects them with the estimated extrinsics:

        X_camera = R * X_world + t
        pixel = K * X_camera, followed by homogeneous divide

    The blue Z axis uses negative object-space Z because checkerboard examples
    normally treat that direction as emerging from the physical board toward the
    camera, which makes the overlay easier to inspect on a 2D photo.
    """

    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a color image with shape (H, W, 3)")
    if axis_length <= 0:
        raise ValueError("axis_length must be positive")

    K = intrinsics.matrix()
    dist = normalize_distortion_coefficients(distortion_coefficients)
    rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=float).reshape(3, 1)

    axis_points_3d = np.float32(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, -axis_length],
        ]
    )
    axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, K, dist)
    axis_points_2d = axis_points_2d.reshape(-1, 2)

    output = image_bgr.copy()

    if image_points is not None:
        for point in np.asarray(image_points, dtype=float).reshape(-1, 2):
            cv2.circle(output, _point_tuple(point), 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, _point_tuple(point), 3, (30, 30, 30), 1, cv2.LINE_AA)

    if reprojected_points is not None:
        for point in np.asarray(reprojected_points, dtype=float).reshape(-1, 2):
            center = _point_tuple(point)
            cv2.drawMarker(
                output,
                center,
                (0, 165, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=9,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

    origin = _point_tuple(axis_points_2d[0])
    x_end = _point_tuple(axis_points_2d[1])
    y_end = _point_tuple(axis_points_2d[2])
    z_end = _point_tuple(axis_points_2d[3])

    # OpenCV uses BGR colors. The convention here is X=red, Y=green, Z=blue.
    cv2.arrowedLine(output, origin, x_end, (0, 0, 255), 5, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(output, origin, y_end, (0, 170, 0), 5, cv2.LINE_AA, tipLength=0.12)
    cv2.arrowedLine(output, origin, z_end, (255, 0, 0), 5, cv2.LINE_AA, tipLength=0.12)
    cv2.circle(output, origin, 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(output, origin, 4, (255, 255, 255), -1, cv2.LINE_AA)

    _draw_label(output, "X", x_end, (0, 0, 255))
    _draw_label(output, "Y", y_end, (0, 170, 0))
    _draw_label(output, "Z", z_end, (255, 0, 0))

    return AxisOverlay(image_bgr=output, axis_image_points=axis_points_2d)

