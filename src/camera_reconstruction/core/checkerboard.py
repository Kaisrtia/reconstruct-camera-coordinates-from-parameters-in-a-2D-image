from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class CheckerboardDetectionError(ValueError):
    """Raised when the checkerboard pattern cannot be detected in the image."""


@dataclass(frozen=True)
class CheckerboardConfig:
    """Checkerboard geometry used as the known 3D object reference.

    `corners_x` and `corners_y` are the number of internal checkerboard corners,
    not the number of square cells. `square_size` is the real-world size of one
    square edge in any consistent unit such as cm, m, or arbitrary assignment
    units.
    """

    corners_x: int
    corners_y: int
    square_size: float

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.corners_x, self.corners_y)


@dataclass(frozen=True)
class CheckerboardDetection:
    object_points: np.ndarray
    image_points: np.ndarray


def _validate_config(config: CheckerboardConfig) -> None:
    if config.corners_x < 2 or config.corners_y < 2:
        raise ValueError("checkerboard corners_x and corners_y must both be at least 2")
    if config.square_size <= 0:
        raise ValueError("checkerboard square_size must be positive")


def build_checkerboard_object_points(config: CheckerboardConfig) -> np.ndarray:
    """Create world-space 3D points for a flat checkerboard on Z = 0.

    The checkerboard coordinate system is chosen as:

    - origin: first detected internal corner
    - +X: across a row of corners
    - +Y: down the next row of corners
    - Z = 0: all checkerboard corners lie on the same physical plane

    These points are the known 3D `Object Points` required by PnP.
    """

    _validate_config(config)
    grid = np.mgrid[0 : config.corners_x, 0 : config.corners_y].T.reshape(-1, 2)
    object_points = np.zeros((grid.shape[0], 3), dtype=np.float32)
    object_points[:, :2] = grid.astype(np.float32) * float(config.square_size)
    return object_points


def detect_checkerboard_corners(image_bgr: np.ndarray, config: CheckerboardConfig) -> CheckerboardDetection:
    """Detect 2D image points and pair them with generated 3D object points."""

    _validate_config(config)
    if image_bgr is None or image_bgr.ndim != 3:
        raise ValueError("image_bgr must be a color image with shape (H, W, 3)")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners = None
    found = False

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            config.pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, config.pattern_size, flags)
        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                40,
                0.001,
            )
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    if not found or corners is None:
        raise CheckerboardDetectionError(
            "Could not detect the checkerboard. Check the internal-corner counts and image clarity."
        )

    image_points = corners.reshape(-1, 2).astype(np.float32)
    object_points = build_checkerboard_object_points(config)
    return CheckerboardDetection(object_points=object_points, image_points=image_points)

