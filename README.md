# Reconstruct Camera Coordinates From Parameters in a 2D Image

This project demonstrates inverse perspective reconstruction for a 2D image. The current application solves a Perspective-n-Point (PnP) problem from checkerboard correspondences, estimates camera extrinsics, computes reprojection error, and renders 3D coordinate axes on top of the uploaded image.

## Core Theory

A 3D world point is projected to an image by:

```text
s [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T
```

Where:

- `K` is the intrinsic matrix.
- `R` and `t` are extrinsic parameters.
- `[u, v]` is the 2D image pixel.
- `s` is the homogeneous scale factor related to depth.

OpenCV represents extrinsics as a world-to-camera transform:

```text
X_camera = R * X_world + t
```

Therefore `t` is not the camera position. The camera center in world coordinates is:

```text
C_world = -R^T * t
```

## Web Application Flow

1. Upload a checkerboard image.
2. Enter checkerboard internal-corner counts and square size.
3. Enter camera intrinsic values `fx`, `fy`, `cx`, `cy`, and `skew`. If focal/principal-point values are blank, the app uses practical image-size defaults.
4. The backend detects checkerboard corners with OpenCV.
5. It builds matching 3D object points on the checkerboard plane `Z = 0`.
6. `cv2.solvePnP` estimates `rvec` and `tvec`.
7. The app reprojects object points to compute reprojection error.
8. The app projects a 3D XYZ axis and draws it over the original uploaded image.

## Project Structure

```text
src/camera_reconstruction/
  core/
    camera_model.py       camera intrinsics, extrinsics, projection matrix
    projection.py         forward projection, World Space -> Camera Space -> Image
    reconstruction.py     inverse projection from pixel to ray/camera point
    checkerboard.py       checkerboard 2D detection and 3D object-point generation
    pnp_solver.py         solvePnP, Rodrigues, camera center, reprojection error
    overlay.py            projected 3D axis rendering on uploaded images
  services/
    reconstruction_service.py
  storage/
    files.py
  web/
    main.py
    routes.py
    schemas.py
    templates/
    static/
scripts/
  run_demo.py
  run_web.py
tests/
```

Compatibility wrappers remain at `src/camera_reconstruction/camera_model.py`, `projection.py`, and `reconstruction.py` so the original CLI demo/tests keep working.

## Setup

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run The Web App

```powershell
python scripts\run_web.py
```

Then open:

```text
http://127.0.0.1:8000
```

Runtime uploads and rendered outputs are written under `var/`, which is ignored by git.

## Run The CLI Demo

```powershell
python scripts\run_demo.py
```

## Run Tests

```powershell
python -m unittest discover -s tests
```

## Evaluation Data

The result page displays:

- Intrinsic matrix `K`.
- Rotation matrix `R`.
- Translation vector `tvec`.
- Camera center `C_world`.
- Projection matrix `P = K [R | t]`.
- Reprojection RMSE, mean error, and max error in pixels.
- Axis endpoint pixels used for the overlay.

