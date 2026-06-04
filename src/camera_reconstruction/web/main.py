from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from camera_reconstruction.storage.files import ensure_directory
from camera_reconstruction.web.routes import router


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = Path(__file__).resolve().parent


def create_app(var_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Camera Reconstruction PnP")

    runtime_dir = var_dir
    if runtime_dir is None:
        runtime_dir = Path(os.environ.get("CAMERA_RECONSTRUCTION_VAR_DIR", PROJECT_ROOT / "var"))

    uploads_dir = ensure_directory(runtime_dir / "uploads")
    results_dir = ensure_directory(runtime_dir / "results")

    app.state.uploads_dir = uploads_dir
    app.state.results_dir = results_dir

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    app.mount("/results", StaticFiles(directory=results_dir), name="results")
    app.include_router(router)

    return app


app = create_app()

