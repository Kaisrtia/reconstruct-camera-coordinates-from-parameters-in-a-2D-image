from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from camera_reconstruction.services.reconstruction_service import ReconstructionError, ReconstructionResult, ReconstructionService
from camera_reconstruction.storage.files import save_upload_bytes, unique_image_path
from camera_reconstruction.web.schemas import DEFAULT_FORM_VALUES, ReconstructionForm, form_values


WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "form": form_values(),
            "error": None,
            "result": None,
        },
    )


@router.post("/reconstruct", response_class=HTMLResponse)
async def reconstruct(
    request: Request,
    image: UploadFile = File(...),
    corners_x: str = Form(DEFAULT_FORM_VALUES["corners_x"]),
    corners_y: str = Form(DEFAULT_FORM_VALUES["corners_y"]),
    square_size: str = Form(DEFAULT_FORM_VALUES["square_size"]),
    fx: str = Form(DEFAULT_FORM_VALUES["fx"]),
    fy: str = Form(DEFAULT_FORM_VALUES["fy"]),
    cx: str = Form(DEFAULT_FORM_VALUES["cx"]),
    cy: str = Form(DEFAULT_FORM_VALUES["cy"]),
    skew: str = Form(DEFAULT_FORM_VALUES["skew"]),
    distortion_coefficients: str = Form(DEFAULT_FORM_VALUES["distortion_coefficients"]),
    axis_length: str = Form(DEFAULT_FORM_VALUES["axis_length"]),
) -> HTMLResponse:
    raw_form = form_values(
        corners_x=corners_x,
        corners_y=corners_y,
        square_size=square_size,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        skew=skew,
        distortion_coefficients=distortion_coefficients,
        axis_length=axis_length,
    )

    try:
        parsed_form = ReconstructionForm.from_raw(raw_form)
        contents = await image.read()
        upload_path = save_upload_bytes(contents, request.app.state.uploads_dir, image.filename or "upload.png")
        result_path = unique_image_path(request.app.state.results_dir, "overlay.png")
        result = ReconstructionService().reconstruct(upload_path, result_path, parsed_form.to_config())
        view_model = _result_view_model(result)
    except (ValueError, ReconstructionError) as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "form": raw_form,
                "error": str(exc),
                "result": None,
            },
            status_code=400,
        )
    finally:
        await image.close()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "form": raw_form,
            "error": None,
            "result": view_model,
        },
    )


def _result_view_model(result: ReconstructionResult) -> dict[str, Any]:
    pnp = result.pnp
    return {
        "overlay_url": f"/results/{result.overlay_image_path.name}",
        "image_size": f"{result.image_width} x {result.image_height}",
        "point_count": int(result.checkerboard.image_points.shape[0]),
        "metrics": {
            "rmse": _format_number(pnp.reprojection_rmse),
            "mean": _format_number(pnp.reprojection_mean),
            "max": _format_number(pnp.reprojection_max),
        },
        "vectors": {
            "rvec": _format_vector(pnp.rvec),
            "tvec": _format_vector(pnp.tvec),
            "camera_center": _format_vector(pnp.camera_center_world),
            "distortion": _format_vector(result.distortion_coefficients.reshape(-1)),
        },
        "matrices": {
            "K": _format_matrix(result.intrinsics.matrix()),
            "R": _format_matrix(pnp.R),
            "P": _format_matrix(pnp.projection_matrix),
        },
        "axis_points": _format_matrix(result.overlay.axis_image_points),
    }


def _format_number(value: float) -> str:
    return f"{float(value):.6g}"


def _format_vector(values: np.ndarray) -> list[str]:
    return [_format_number(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def _format_matrix(values: np.ndarray) -> list[list[str]]:
    matrix = np.asarray(values, dtype=float)
    return [[_format_number(value) for value in row] for row in matrix]
