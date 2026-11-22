"""Cut tracking into ground-ball segments without using pass-event ends.

A touch is a sudden change in ball velocity while a player is within 2 m.
Segments start at that touch, use the outgoing velocity, and stop at the next
touch, the sideline, or 3 seconds.
"""

from __future__ import annotations

import numpy as np

DT = 0.1
MAX_FUTURE_S = 3.0
TOUCH_RADIUS_M = 2.0
SPEED_JUMP = 5.0
MIN_TOUCH_GAP_S = 0.4
MIN_OUT_SPEED = 2.0
SAMPLE_TOLERANCE_S = 0.08


def in_play_window(events: list[dict]) -> tuple[float, float] | None:
    kicks = [item["time"].timestamp() for item in events if item.get("event_tag") == "KickOff"]
    whistles = [item["time"].timestamp() for item in events if item.get("event_tag") == "FinalWhistle"]
    if not kicks or not whistles:
        return None
    return min(kicks), max(whistles)


def touch_indices(
    ball_xy: np.ndarray,
    nearest_player_m: np.ndarray,
    dt: float = DT,
    speed_jump: float = SPEED_JUMP,
    radius_m: float = TOUCH_RADIUS_M,
    min_gap_s: float = MIN_TOUCH_GAP_S,
) -> np.ndarray:
    """Indices where the ball's velocity jumps and a player is close."""
    velocity = np.diff(ball_xy, axis=0) / dt
    if len(velocity) < 5:
        return np.zeros(0, dtype=int)
    before = velocity[:-2]
    after = velocity[2:]
    jump = np.linalg.norm(after - before, axis=1)
    centers = np.arange(2, 2 + len(jump))
    close = nearest_player_m[centers] <= radius_m
    hits = centers[np.isfinite(jump) & (jump >= speed_jump) & close]
    if len(hits) == 0:
        return hits
    kept = [int(hits[0])]
    for index in hits[1:]:
        if index - kept[-1] >= int(round(min_gap_s / dt)):
            kept.append(int(index))
    return np.asarray(kept, dtype=int)


def roll_position(origin: np.ndarray, velocity: np.ndarray, times: np.ndarray, drag: float) -> np.ndarray:
    """Constant direction with exponentially decaying speed. drag 0 is constant velocity."""
    times = np.asarray(times, dtype=np.float64)
    if drag < 1e-4:
        return origin + velocity * times[:, None]
    scale = (1.0 - np.exp(-drag * times)) / drag
    return origin + velocity * scale[:, None]


def constant_velocity(origin: np.ndarray, velocity: np.ndarray, times: np.ndarray) -> np.ndarray:
    return roll_position(origin, velocity, times, 0.0)


def sample_series(stamps: np.ndarray, values: np.ndarray, grid: np.ndarray, tolerance: float = SAMPLE_TOLERANCE_S) -> np.ndarray:
    """Nearest sample on `grid`. Points farther than `tolerance` are NaN."""
    if len(stamps) == 0:
        return np.full((len(grid), values.shape[1]), np.nan)
    index = np.searchsorted(stamps, grid)
    index = np.clip(index, 1, len(stamps) - 1)
    left = index - 1
    take_right = np.abs(stamps[index] - grid) < np.abs(stamps[left] - grid)
    picked = np.where(take_right, index, left)
    gap = np.abs(stamps[picked] - grid)
    out = values[picked].astype(np.float64).copy()
    out[gap > tolerance] = np.nan
    return out


def build_segments(ball, players: dict, pitch: tuple[float, float], t_start: float, t_end: float) -> list[dict]:
    """Return segments in tracking coordinates. `ball` rows are (time, x, y)."""
    if t_end - t_start < 5 or len(ball) < 10:
        return []
    grid = np.arange(t_start, t_end, DT)
    ball_t = np.asarray([row[0] for row in ball], dtype=np.float64)
    ball_xy = np.asarray([[row[1], row[2]] for row in ball], dtype=np.float64)
    sampled = sample_series(ball_t, ball_xy, grid)
    names = []
    teams = []
    columns = []
    for person, rows in players.items():
        if len(rows) < 2:
            continue
        stamps = np.asarray([row[0] for row in rows], dtype=np.float64)
        xy = np.asarray([[row[1], row[2]] for row in rows], dtype=np.float64)
        columns.append(sample_series(stamps, xy, grid))
        names.append(person)
        teams.append(rows[0][3])
    if not columns:
        return []
    player_xy = np.stack(columns, axis=1)
    distances = np.linalg.norm(player_xy - sampled[:, None, :], axis=2)
    nearest = np.nanmin(distances, axis=1)
    touches = touch_indices(sampled, nearest)
    half_x, half_y = pitch[0] / 2.0, pitch[1] / 2.0
    oob = (np.abs(sampled[:, 0]) > half_x + 1.0) | (np.abs(sampled[:, 1]) > half_y + 1.0)
    segments = []
    horizon = int(round(MAX_FUTURE_S / DT))
    for start, stop in zip(touches[:-1], touches[1:]):
        end = min(int(stop), int(start) + horizon)
        if end - start < 3 or not np.isfinite(sampled[start]).all() or not np.isfinite(sampled[start + 2]).all():
            continue
        oob_at = np.flatnonzero(oob[start + 1 : end + 1])
        reason = "touch"
        if len(oob_at):
            end = int(start) + 1 + int(oob_at[0])
            reason = "oob"
        elif end < int(stop):
            reason = "horizon"
        velocity = (sampled[start + 2] - sampled[start]) / (2 * DT)
        if not np.isfinite(velocity).all() or float(np.linalg.norm(velocity)) < MIN_OUT_SPEED:
            continue
        future = sampled[start : end + 1]
        if not np.isfinite(future).all():
            continue
        earlier = player_xy[start - 2] if start >= 2 else player_xy[start]
        roster = _roster(player_xy[start], earlier, names, teams)
        if len(roster) < 4:
            continue
        kept = [index for index in range(len(names)) if np.isfinite(player_xy[start, index]).all()]
        player_future = player_xy[start : end + 1, kept, :]
        toucher = None
        if reason == "touch" and end < len(distances):
            order = int(np.nanargmin(distances[end]))
            if np.isfinite(distances[end, order]) and distances[end, order] <= TOUCH_RADIUS_M + 0.5:
                toucher = {"id": names[order], "team": teams[order]}
        segments.append(
            {
                "origin": future[0],
                "velocity": velocity,
                "future": future,
                "dt": DT,
                "reason": reason,
                "roster": roster,
                "players": player_future,
                "toucher": toucher,
                "start_index": start,
            }
        )
    return segments


def _roster(now: np.ndarray, earlier: np.ndarray, names: list[str], teams: list[str]) -> list[dict]:
    roster = []
    for index, name in enumerate(names):
        if not np.isfinite(now[index]).all():
            continue
        velocity = np.zeros(2) if not np.isfinite(earlier[index]).all() else (now[index] - earlier[index]) / (2 * DT)
        roster.append({"id": name, "team": teams[index], "xy": now[index], "velocity": velocity})
    return roster
