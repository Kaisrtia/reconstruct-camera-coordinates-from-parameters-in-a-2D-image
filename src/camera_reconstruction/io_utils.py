from __future__ import annotations

import numpy as np


def load_points_csv(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def save_points_csv(path: str, points: np.ndarray) -> None:
    np.savetxt(path, points, delimiter=",")
