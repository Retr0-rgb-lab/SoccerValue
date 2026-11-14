"""Geometric and hand-built spatial features for the same logistic model.

Model A sees only the ball and the recorded pass segment. Model B adds
distances and counts taken from the same segment and the player lists.
Both go through one logistic regression, so a gain from A to B is the
hand-built spatial information, not a change of model family.
"""

from __future__ import annotations

import math

import numpy as np

LANE_RADIUS_M = 3.0
PASSER_EXCLUDE_M = 1.5

GEOMETRY_NAMES = (
    "ball_x",
    "ball_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "distance",
    "sin_angle",
    "cos_angle",
)
SPATIAL_NAMES = GEOMETRY_NAMES + (
    "nearest_defender_to_ball",
    "nearest_defender_to_end",
    "nearest_teammate_to_end",
    "defenders_within_3m_of_lane",
    "nearest_defender_to_lane",
)


def geometry_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return _stack(rows, _geometry)


def spatial_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return _stack(rows, _spatial)


def describe_space(row: dict) -> dict:
    """Meter values used when reading a single wrong prediction."""
    ball = row["ball"]
    end = row["end"]
    defense = row["defense"]
    attack = row["attack"]
    return {
        "pass_distance_m": _hypot(end[0] - ball[0], end[1] - ball[1]),
        "nearest_defender_to_ball_m": _nearest(ball, defense),
        "nearest_defender_to_end_m": _nearest(end, defense),
        "nearest_teammate_to_end_m": _nearest(end, _others(attack, ball)),
        "defenders_within_3m_of_lane": _lane_count(ball, end, defense),
        "nearest_defender_to_lane_m": _lane_clearance(ball, end, defense),
    }


def _stack(rows: list[dict], builder) -> tuple[np.ndarray, np.ndarray]:
    features = [builder(row) for row in rows]
    labels = [row["success"] for row in rows]
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _geometry(row: dict) -> list[float]:
    pitch_x, pitch_y = row["pitch"]
    bx, by = row["ball"]
    ex, ey = row["end"]
    dx = ex - bx
    dy = ey - by
    distance = _hypot(dx, dy)
    diagonal = _hypot(pitch_x, pitch_y)
    angle = math.atan2(dy, dx)
    return [
        bx / pitch_x,
        by / pitch_y,
        ex / pitch_x,
        ey / pitch_y,
        dx / pitch_x,
        dy / pitch_y,
        distance / diagonal,
        math.sin(angle),
        math.cos(angle),
    ]


def _spatial(row: dict) -> list[float]:
    pitch_x, pitch_y = row["pitch"]
    diagonal = _hypot(pitch_x, pitch_y)
    ball = row["ball"]
    end = row["end"]
    defense = row["defense"]
    others = _others(row["attack"], ball)
    return _geometry(row) + [
        _nearest(ball, defense) / diagonal,
        _nearest(end, defense) / diagonal,
        _nearest(end, others) / diagonal,
        float(_lane_count(ball, end, defense)),
        _lane_clearance(ball, end, defense) / diagonal,
    ]


def _others(attack, ball) -> list[tuple[float, float]]:
    return [point for point in attack if _hypot(point[0] - ball[0], point[1] - ball[1]) > PASSER_EXCLUDE_M]


def _nearest(origin, points) -> float:
    if not points:
        return _hypot(105.0, 68.0)
    return min(_hypot(point[0] - origin[0], point[1] - origin[1]) for point in points)


def _lane_count(start, end, defense) -> int:
    return sum(_segment_distance(point, start, end) <= LANE_RADIUS_M for point in defense)


def _lane_clearance(start, end, defense) -> float:
    if not defense:
        return _hypot(105.0, 68.0)
    return min(_segment_distance(point, start, end) for point in defense)


def _segment_distance(point, start, end) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    abx, aby = bx - ax, by - ay
    length_sq = abx * abx + aby * aby
    if length_sq < 1e-8:
        return _hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / length_sq))
    return _hypot(px - (ax + t * abx), py - (ay + t * aby))


def _hypot(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5
