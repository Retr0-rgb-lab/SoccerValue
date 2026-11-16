"""Ground-roll ball flight versus constant velocity. One drag coefficient."""

from __future__ import annotations

import numpy as np

from soccervalue.world_segments import DT, roll_position


def path_error(segment: dict, drag: float, at_s: float | None = None) -> float | None:
    future = segment["future"]
    if at_s is None:
        index = len(future) - 1
    else:
        index = int(round(at_s / segment["dt"]))
        if index >= len(future):
            return None
    times = np.arange(index + 1) * segment["dt"]
    predicted = roll_position(segment["origin"], segment["velocity"], times, drag)
    return float(np.linalg.norm(predicted[index] - future[index]))


def mean_error(segments: list[dict], drag: float, at_s: float | None) -> float | None:
    errors = [path_error(segment, drag, at_s) for segment in segments]
    usable = [error for error in errors if error is not None]
    if not usable:
        return None
    return float(np.mean(usable))


def fit_drag(segments: list[dict]) -> float:
    """Pick the decay rate that minimises train error along the path, not on the val match."""

    def cost(drag: float) -> float:
        total = 0.0
        count = 0
        for segment in segments:
            steps = min(len(segment["future"]), int(round(2.0 / segment["dt"])) + 1)
            times = np.arange(steps) * segment["dt"]
            predicted = roll_position(segment["origin"], segment["velocity"], times, drag)
            total += float(np.linalg.norm(predicted - segment["future"][:steps], axis=1).mean())
            count += 1
        return total / max(count, 1)

    grid = np.linspace(0.0, 1.5, 16)
    return float(min(grid, key=cost))


def summarise(segments: list[dict], drag: float) -> dict:
    return {
        "n": len(segments),
        "n_reaching_1s": sum(len(segment["future"]) > int(round(1.0 / DT)) for segment in segments),
        "constant_velocity_ade_1s": mean_error(segments, 0.0, 1.0),
        "drag_ade_1s": mean_error(segments, drag, 1.0),
        "constant_velocity_at_end": mean_error(segments, 0.0, None),
        "drag_at_end": mean_error(segments, drag, None),
    }
