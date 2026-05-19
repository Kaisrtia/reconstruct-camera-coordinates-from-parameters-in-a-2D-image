from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .camera_model import CameraExtrinsics, CameraIntrinsics, camera_center_world, rotation_matrix_from_euler
from .projection import project_points


def main(out_path: str = "outputs/demo.png") -> None:
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    R = rotation_matrix_from_euler(
        np.deg2rad(5.0),
        np.deg2rad(-10.0),
        np.deg2rad(25.0),
    )
    t = np.array([0.2, -0.1, 3.0])
    extrinsics = CameraExtrinsics(R=R, t=t)

    grid = np.linspace(-1.0, 1.0, 10)
    xs, ys = np.meshgrid(grid, grid)
    zs = np.zeros_like(xs)
    points_3d = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)

    points_2d = project_points(points_3d, intrinsics, extrinsics)
    center = camera_center_world(extrinsics)

    print("camera_center_world:", center)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10)
    plt.gca().invert_yaxis()
    plt.title("Projected points")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print("saved plot:", out_path)


if __name__ == "__main__":
    main()
