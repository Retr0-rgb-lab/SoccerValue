"""Read tracking and move each pass to the frame that matches the kick."""

from __future__ import annotations

import bisect
from datetime import datetime
from pathlib import Path

from lxml import etree

from soccervalue.align import SYNC_WINDOW_S, TRACKING_TO_EVENT, convert, frame_cost
from soccervalue.intent import IntentTracker


def attach_tracking(tracking_path: Path, passes: list[dict], pitch: tuple[float, float]) -> None:
    """Store the tracking frame inside a few seconds that best matches each pass.

    DFL event timestamps are not the optical kick instant. The dataset paper
    measured an average 9 m gap until each event is moved to a nearby frame.

    The same pass also gets an `intent` entry: the teammate the passer aimed
    at, read from the kick rather than from the recorded result.
    """
    if not passes:
        return
    ball, players = _read_trajectories(tracking_path)
    ball_times = [item[0] for item in ball]
    player_times = {person: [item[0] for item in rows] for person, rows in players.items()}
    pitch_x, pitch_y = pitch
    tracker = IntentTracker(ball, pitch_x, pitch_y)
    for item in passes:
        target = item["time"].timestamp()
        chosen = _best_frame(item, target, ball, ball_times, players, player_times, pitch_x, pitch_y)
        if chosen is None:
            continue
        item["track_ball"] = chosen["ball"]
        item["track_passer"] = chosen["passer"]
        item["track_players"] = chosen["players"]
        item["frame_dt"] = chosen["dt"]
        item["frame_signed_dt"] = chosen["signed_dt"]
        item["intent"] = tracker.estimate(item, chosen["index"], chosen["players"])


def _read_trajectories(path: Path) -> tuple[list[tuple[float, float, float]], dict]:
    ball: list[tuple[float, float, float]] = []
    players: dict[str, list[tuple[float, float, float, str]]] = {}
    current = {"team": "", "person": ""}
    for phase, elem in etree.iterparse(str(path), events=("start", "end")):
        if phase == "start" and elem.tag == "FrameSet":
            current = {"team": elem.get("TeamId") or "", "person": elem.get("PersonId") or ""}
        elif phase == "end" and elem.tag == "Frame":
            team = current["team"]
            if team.lower() != "referee":
                stamp = datetime.fromisoformat(elem.get("T")).timestamp()
                x = float(elem.get("X"))
                y = float(elem.get("Y"))
                if team.lower() == "ball":
                    ball.append((stamp, x, y))
                else:
                    players.setdefault(current["person"], []).append((stamp, x, y, team))
            elem.clear()
        elif phase == "end" and elem.tag == "FrameSet":
            elem.clear()
    ball.sort(key=lambda row: row[0])
    for rows in players.values():
        rows.sort(key=lambda row: row[0])
    return ball, players


def _best_frame(item, target, ball, ball_times, players, player_times, pitch_x, pitch_y):
    if not ball_times:
        return None
    start = bisect.bisect_left(ball_times, target - SYNC_WINDOW_S)
    stop = bisect.bisect_right(ball_times, target + SYNC_WINDOW_S)
    passer_rows = players.get(item["player_id"], [])
    passer_times = player_times.get(item["player_id"], [])
    best = None
    best_index = None
    best_cost = float("inf")
    for index in range(start, stop):
        stamp, x, y = ball[index]
        dt = stamp - target
        bx, by = convert(TRACKING_TO_EVENT, x, y, pitch_x, pitch_y)
        ball_event = ((bx - item["start_x"]) ** 2 + (by - item["start_y"]) ** 2) ** 0.5
        passer_xy = _at(passer_rows, passer_times, stamp)
        if passer_xy is None:
            passer_ball = 15.0
        else:
            px, py = convert(TRACKING_TO_EVENT, passer_xy[1], passer_xy[2], pitch_x, pitch_y)
            passer_ball = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        cost = frame_cost(ball_event, passer_ball, dt)
        if cost < best_cost:
            best_cost = cost
            best = (stamp, x, y)
            best_index = index
    if best is None:
        return None
    stamp, x, y = best
    snapshot = {}
    for person, rows in players.items():
        point = _at(rows, player_times[person], stamp)
        if point is not None:
            snapshot[person] = (point[1], point[2], point[3])
    passer = snapshot.get(item["player_id"])
    return {
        "ball": (x, y, None),
        "passer": (passer[0], passer[1]) if passer else None,
        "players": snapshot,
        "dt": abs(stamp - target),
        "signed_dt": stamp - target,
        "index": best_index,
    }


def _at(rows, times, stamp: float, tolerance: float = 0.03):
    if not times:
        return None
    index = bisect.bisect_left(times, stamp)
    best = None
    best_gap = tolerance
    for candidate in (index - 1, index):
        if not 0 <= candidate < len(times):
            continue
        gap = abs(times[candidate] - stamp)
        if gap <= best_gap:
            best = rows[candidate]
            best_gap = gap
    return best
