"""Pitch rasters and the single-cell success loss."""

from __future__ import annotations

import numpy as np

GRID_X = 104
GRID_Y = 68
SIGMA_CELLS = 2.0


def cell_index(x_m: float, y_m: float, pitch_x: float, pitch_y: float) -> tuple[int, int]:
    ix = int(np.clip(np.rint(x_m / pitch_x * (GRID_X - 1)), 0, GRID_X - 1))
    iy = int(np.clip(np.rint(y_m / pitch_y * (GRID_Y - 1)), 0, GRID_Y - 1))
    return ix, iy


def raster(attack, defense, ball, pitch: tuple[float, float]) -> np.ndarray:
    """Three channels: attacking players, defending players, ball."""
    image = np.zeros((3, GRID_Y, GRID_X), dtype=np.float32)
    _paint(image[0], attack, pitch)
    _paint(image[1], defense, pitch)
    _paint(image[2], [ball], pitch)
    return image


def _paint(channel: np.ndarray, points, pitch: tuple[float, float]) -> None:
    pitch_x, pitch_y = pitch
    radius = int(3 * SIGMA_CELLS)
    for x_m, y_m in points:
        cx = x_m / pitch_x * (GRID_X - 1)
        cy = y_m / pitch_y * (GRID_Y - 1)
        x0 = max(int(cx) - radius, 0)
        x1 = min(int(cx) + radius + 1, GRID_X)
        y0 = max(int(cy) - radius, 0)
        y1 = min(int(cy) + radius + 1, GRID_Y)
        if x0 >= x1 or y0 >= y1:
            continue
        xs = np.arange(x0, x1) - cx
        ys = np.arange(y0, y1) - cy
        kernel = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (2 * SIGMA_CELLS**2))
        channel[y0:y1, x0:x1] += kernel.astype(np.float32)
