from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_reconstruction.core.camera_model import CameraIntrinsics
from camera_reconstruction.core.checkerboard import (
    CheckerboardConfig,
    CheckerboardDetection,
    CheckerboardDetectionError,
    detect_checkerboard_corners,
)
from camera_reconstruction.core.overlay import AxisOverlay, draw_axes_overlay
from camera_reconstruction.core.pnp_solver import PnPResult, normalize_distortion_coefficients, solve_pnp


class ReconstructionError(ValueError):
    """Raised for user-facing reconstruction failures."""


@dataclass(frozen=True)
class ReconstructionConfig:
    corners_x: int
    corners_y: int
    square_size: float
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    skew: float = 0.0
    distortion_coefficients: tuple[float, ...] = ()
    axis_length: float | None = None


@dataclass(frozen=True)
class ReconstructionResult:
    source_image_path: Path
    overlay_image_path: Path
    image_width: int
    image_height: int
    intrinsics: CameraIntrinsics
    distortion_coefficients: np.ndarray
    checkerboard: CheckerboardDetection
    pnp: PnPResult
    overlay: AxisOverlay


class ReconstructionService:
    """Application pipeline from uploaded image to rendered PnP result."""

    def reconstruct(
        self,
        image_path: Path,
        output_path: Path,
        config: ReconstructionConfig,
    ) -> ReconstructionResult:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ReconstructionError("The uploaded file could not be read as an image.")

        height, width = image.shape[:2]
        intrinsics = self._build_intrinsics(config, width=width, height=height)
        distortion = normalize_distortion_coefficients(config.distortion_coefficients)
        checkerboard_config = CheckerboardConfig(
            corners_x=config.corners_x,
            corners_y=config.corners_y,
            square_size=config.square_size,
        )

        try:
            detection = detect_checkerboard_corners(image, checkerboard_config)
            pnp = solve_pnp(
                detection.object_points,
                detection.image_points,
                intrinsics,
                distortion,
            )
        except CheckerboardDetectionError as exc:
            raise ReconstructionError(str(exc)) from exc
        except ValueError as exc:
            raise ReconstructionError(str(exc)) from exc

        axis_length = config.axis_length if config.axis_length is not None else config.square_size * 3.0
        overlay = draw_axes_overlay(
            image,
            intrinsics,
            pnp.rvec,
            pnp.tvec,
            distortion,
            axis_length=axis_length,
            image_points=detection.image_points,
            reprojected_points=pnp.reprojected_points,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), overlay.image_bgr):
            raise ReconstructionError("Could not write the rendered overlay image.")

        return ReconstructionResult(
            source_image_path=image_path,
            overlay_image_path=output_path,
            image_width=width,
            image_height=height,
            intrinsics=intrinsics,
            distortion_coefficients=distortion,
            checkerboard=detection,
            pnp=pnp,
            overlay=overlay,
        )

    @staticmethod
    def _build_intrinsics(config: ReconstructionConfig, width: int, height: int) -> CameraIntrinsics:
        default_focal = float(max(width, height))
        fx = config.fx if config.fx is not None else default_focal
        fy = config.fy if config.fy is not None else fx
        cx = config.cx if config.cx is not None else width / 2.0
        cy = config.cy if config.cy is not None else height / 2.0

        for name, value in {"fx": fx, "fy": fy}.items():
            if value <= 0:
                raise ReconstructionError(f"{name} must be positive")

        return CameraIntrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), skew=float(config.skew))

