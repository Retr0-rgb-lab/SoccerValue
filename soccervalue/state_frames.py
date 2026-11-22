"""Attach a tracking frame at or before the event, then face team T's goal."""

from __future__ import annotations

import bisect
import math
from pathlib import Path

from soccervalue.align import SYNC_WINDOW_S, TRACKING_TO_EVENT, convert, frame_cost
from soccervalue.tracking import _at, _read_trajectories


def attach_state_frame(
    tracking_path: Path,
    samples: list[dict],
    pitch: tuple[float, float],
    trajectories: tuple | None = None,
) -> None:
    """Write the best frame in [t - 6 s, t]. Frames after the event are ignored."""
    if not samples:
        return
    if trajectories is None:
        ball, players = _read_trajectories(tracking_path)
    else:
        ball, players = trajectories
    ball_times = [item[0] for item in ball]
    player_times = {person: [item[0] for item in rows] for person, rows in players.items()}
    pitch_x, pitch_y = pitch
    for item in samples:
        chosen = _past_frame(item, ball, ball_times, players, player_times, pitch_x, pitch_y)
        if chosen is None:
            continue
        item["track_ball"] = chosen["ball"]
        item["track_players"] = chosen["players"]
        item["frame_signed_dt"] = chosen["signed_dt"]


def orient_state(sample: dict, transform: str, pitch_x: float, pitch_y: float) -> dict | None:
    """Return T / opponent / ball with T attacking toward +x."""
    ball = sample.get("track_ball")
    players = sample.get("track_players")
    if ball is None or not players or sample.get("team_left") is None:
        return None
    bx, by = convert(transform, ball[0], ball[1], pitch_x, pitch_y)
    if math.hypot(bx - sample["x"], by - sample["y"]) > 8:
        return None
    attack_positive = sample["team_id"] == sample["team_left"]
    team = []
    opponent = []
    for _person, (x, y, side) in players.items():
        point = _flip(convert(transform, x, y, pitch_x, pitch_y), pitch_x, pitch_y, attack_positive)
        if side == sample["team_id"]:
            team.append(point)
        else:
            opponent.append(point)
    return {
        "match_id": sample.get("match_id"),
        "event_id": sample.get("event_id"),
        "ball": _flip((bx, by), pitch_x, pitch_y, attack_positive),
        "team": team,
        "opponent": opponent,
        "possession_by_t": sample["possession_by_t"],
        "y_for": sample["y_for"],
        "y_against": sample["y_against"],
        "xg_for_sum": sample.get("xg_for_sum", 0.0),
        "xg_against_sum": sample.get("xg_against_sum", 0.0),
        "xg_for_product": sample.get("xg_for_product", 0.0),
        "xg_against_product": sample.get("xg_against_product", 0.0),
        "pitch": (pitch_x, pitch_y),
        "shots_in_window": sample.get("shots_in_window") or [],
    }


def _past_frame(item, ball, ball_times, players, player_times, pitch_x, pitch_y):
    if not ball_times:
        return None
    target = item["time"].timestamp()
    start = bisect.bisect_left(ball_times, target - SYNC_WINDOW_S)
    stop = bisect.bisect_right(ball_times, target)
    passer_rows = players.get(item.get("player_id"), [])
    passer_times = player_times.get(item.get("player_id"), [])
    best = None
    best_cost = float("inf")
    for index in range(start, stop):
        stamp, x, y = ball[index]
        if stamp - target > 0:
            continue
        bx, by = convert(TRACKING_TO_EVENT, x, y, pitch_x, pitch_y)
        ball_event = ((bx - item["x"]) ** 2 + (by - item["y"]) ** 2) ** 0.5
        passer_xy = _at(passer_rows, passer_times, stamp)
        if passer_xy is None:
            passer_ball = 15.0
        else:
            px, py = convert(TRACKING_TO_EVENT, passer_xy[1], passer_xy[2], pitch_x, pitch_y)
            passer_ball = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        cost = frame_cost(ball_event, passer_ball, stamp - target)
        if cost < best_cost:
            best_cost = cost
            best = (stamp, x, y)
    if best is None:
        return None
    stamp, x, y = best
    snapshot = {}
    for person, rows in players.items():
        point = _at(rows, player_times[person], stamp)
        if point is not None:
            snapshot[person] = (point[1], point[2], point[3])
    return {"ball": (x, y), "players": snapshot, "signed_dt": stamp - target}


def _flip(point: tuple[float, float], pitch_x: float, pitch_y: float, attack_positive: bool) -> tuple[float, float]:
    x, y = point
    if attack_positive:
        return x, y
    return pitch_x - x, pitch_y - y
