from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from .camera_model import CameraExtrinsics, CameraIntrinsics, camera_center_world, rotation_matrix_from_euler
from .projection import project_points
from .reconstruction import backproject_ray, pixel_to_camera_point


def _print_matrix(name: str, value: np.ndarray) -> None:
    print(f"{name}:")
    print(np.array2string(value, precision=4, suppress_small=True))


def _make_plane_points() -> np.ndarray:
    grid = np.linspace(-1.0, 1.0, 10)
    xs, ys = np.meshgrid(grid, grid)
    zs = np.zeros_like(xs)
    return np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)


def _plot_projection(points_2d: np.ndarray, selected_pixels: list[tuple[float, float]], out_path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(points_2d[:, 0], points_2d[:, 1], s=16, label="Projected 3D grid")

    selected = np.array(selected_pixels, dtype=float)
    plt.scatter(selected[:, 0], selected[:, 1], marker="x", s=90, color="crimson", label="Backprojected pixels")
    for index, pixel in enumerate(selected_pixels, start=1):
        plt.annotate(f"p{index}", pixel, xytext=(6, 6), textcoords="offset points", color="crimson")

    plt.gca().invert_yaxis()
    plt.title("Forward projection: 3D plane points to 2D image")
    plt.xlabel("u pixel")
    plt.ylabel("v pixel")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_rays(
    points_3d: np.ndarray,
    ray_origins: list[np.ndarray],
    ray_directions: list[np.ndarray],
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=12, label="World plane z=0")

    for index, (origin, direction) in enumerate(zip(ray_origins, ray_directions), start=1):
        length = 3.5
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            length=length,
            normalize=True,
            label=f"Backprojected ray p{index}",
        )

    camera = ray_origins[0]
    ax.scatter([camera[0]], [camera[1]], [camera[2]], marker="^", s=80, color="black", label="Camera center")
    ax.set_title("Inverse projection: 2D pixels to 3D rays")
    ax.set_xlabel("X world")
    ax.set_ylabel("Y world")
    ax.set_zlabel("Z world")
    ax.legend()
    ax.view_init(elev=24, azim=-58)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(out_dir: str = "outputs") -> None:
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=320.0, cy=240.0)
    R = rotation_matrix_from_euler(
        np.deg2rad(5.0),
        np.deg2rad(-10.0),
        np.deg2rad(25.0),
    )
    t = np.array([0.2, -0.1, 3.0])
    extrinsics = CameraExtrinsics(R=R, t=t)

    points_3d = _make_plane_points()
    selected_pixels = [(320.0, 240.0), (420.0, 260.0), (250.0, 180.0)]
    depth = 3.0

    points_2d = project_points(points_3d, intrinsics, extrinsics)
    center = camera_center_world(extrinsics)

    os.makedirs(out_dir, exist_ok=True)
    projection_path = os.path.join(out_dir, "projection_2d.png")
    rays_path = os.path.join(out_dir, "backprojected_rays_3d.png")

    _print_matrix("Intrinsic matrix K", intrinsics.matrix())
    _print_matrix("Rotation matrix R", R)
    _print_matrix("Translation vector t", t)
    _print_matrix("Projection matrix P = K [R | t]", intrinsics.matrix() @ extrinsics.matrix())
    print("Camera center in world coordinates:", np.array2string(center, precision=4, suppress_small=True))
    print()

    ray_origins = []
    ray_directions = []
    for index, pixel in enumerate(selected_pixels, start=1):
        origin, direction = backproject_ray(pixel, intrinsics, extrinsics)
        point_camera = pixel_to_camera_point(pixel, depth, intrinsics)
        ray_origins.append(origin)
        ray_directions.append(direction)

        print(f"Pixel p{index}: {pixel}")
        print("  Ray origin world:", np.array2string(origin, precision=4, suppress_small=True))
        print("  Ray direction world:", np.array2string(direction, precision=4, suppress_small=True))
        print(
            f"  Camera coordinate at depth z={depth}:",
            np.array2string(point_camera, precision=4, suppress_small=True),
        )

    _plot_projection(points_2d, selected_pixels, projection_path)
    _plot_rays(points_3d, ray_origins, ray_directions, rays_path)

    print()
    print("Saved 2D projection image:", projection_path)
    print("Saved 3D inverse-projection image:", rays_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the camera reconstruction demo.")
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory where output images are saved. Default: outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(out_dir=args.out_dir)
