# Reconstruct Camera Coordinates From Parameters in a 2D Image

This project demonstrates inverse perspective projection: reconstructing camera/world information from 2D image coordinates and known camera parameters.

The program includes:

- A pinhole camera model with intrinsic matrix `K`
- Camera extrinsic parameters `R` and `t`
- Forward projection from 3D world points to 2D image pixels
- Inverse projection from 2D pixels back to 3D rays
- Camera-coordinate reconstruction when depth is known
- Output images for evaluation and reporting

## Theory

A 3D world point is projected to the image by:

```text
s [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T
```

Where:

- `K` is the intrinsic matrix: focal length and principal point
- `R` is the rotation matrix
- `t` is the translation vector
- `[u, v]` is the 2D image pixel
- `s` is a scale factor related to depth

Forward projection is:

```text
3D point -> 2D image pixel
```

Inverse projection is:

```text
2D image pixel -> 3D ray
```

A single 2D pixel does not define one unique 3D point. It defines a ray that starts at the camera center and travels through the image plane. To recover one exact 3D camera coordinate, depth must be known or another geometric constraint must be provided.

For a known camera-space depth `Zc`, a pixel can be reconstructed in camera coordinates by:

```text
[Xc, Yc, Zc]^T = Zc * K^-1 [u, v, 1]^T
```

## Project Structure

```text
src/camera_reconstruction/
  camera_model.py      camera intrinsics, extrinsics, projection matrix
  projection.py        forward projection 3D -> 2D
  reconstruction.py    inverse projection 2D -> ray / camera point
  demo.py              runnable assignment demo
scripts/
  run_demo.py          script entry point
tests/
  test_smoke.py        basic correctness tests
```

## Setup

On Linux/macOS:

```bash
cd "/run/media/kaisrtia/Data/Project Github/reconstruct-camera-coordinates-from-parameters-in-a-2D-image"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run The Demo

```bash
python3 -m camera_reconstruction.demo
```

Or:

```bash
python3 scripts/run_demo.py
```

If your project folder is on a read-only mounted drive, choose another output directory:

```bash
python3 -m camera_reconstruction.demo --out-dir /tmp/camera-reconstruction-output
```

The program prints:

- Intrinsic matrix `K`
- Rotation matrix `R`
- Translation vector `t`
- Projection matrix `P = K [R | t]`
- Camera center in world coordinates
- Backprojected ray origin and direction for selected pixels
- Reconstructed camera coordinate for selected pixels when depth is known

The program saves:

```text
outputs/projection_2d.png
outputs/backprojected_rays_3d.png
```

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Evaluation And Comments

- The forward projection result shows how 3D grid points become 2D image pixels.
- The inverse projection result shows that each selected 2D pixel becomes a 3D ray.
- A unique 3D point cannot be reconstructed from only one 2D point without depth.
- If depth is supplied, the pixel can be converted to a concrete camera coordinate.
- Accuracy depends on the correctness of the intrinsic parameters, rotation, translation, and depth value.
