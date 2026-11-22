"""The state after a pass, and what the value model says about it.

A recorded pass has three states that matter: the state the passer is in, the
state the ball is in when it reaches the next player, and the state the world
model thinks it will be in.

The world model gives the ball path and the next toucher. The value model
gives a number for a state: the chance of a shot in the next ten seconds for
the team being evaluated minus the chance for the other team.

This module links a pass to the ground-ball segment that starts at its kick,
then builds the three states in the frame the value model was trained in.

Nothing here reads a label. The pass outcome is used only to report what the
value model says about states it did not see.
"""

from __future__ import annotations

import math

import numpy as np

from soccervalue.align import convert
from soccervalue.touch_head import player_features, predict_index
from soccervalue.world_segments import roll_position


def orient(point: tuple[float, float], pitch: tuple[float, float], attack_positive: bool) -> tuple[float, float]:
    """Flip a point so that the team being evaluated attacks toward +x."""
    x, y = point
    if attack_positive:
        return (x, y)
    return (pitch[0] - x, pitch[1] - y)


def link_segment(
    pass_row: dict,
    segments: list[dict],
    transform: str,
    pitch: tuple[float, float],
    limit_m: float = 3.0,
) -> dict | None:
    """The ground-ball segment that starts at this pass.

    The kick of a pass is a touch, so the pass flight is the segment that
    begins there. The link is made on the ball position rather than on the
    frame index, because the two touch detectors do not fire on the same frame.
    """
    intent = pass_row.get("intent") or {}
    if intent.get("status") != "ok" or intent.get("ball_x") is None:
        return None
    origin = convert(transform, intent["ball_x"], intent["ball_y"], pitch[0], pitch[1])
    best = None
    best_distance = limit_m
    for segment in segments:
        if segment.get("toucher") is None:
            continue
        start = convert(transform, segment["origin"][0], segment["origin"][1], pitch[0], pitch[1])
        distance = math.hypot(start[0] - origin[0], start[1] - origin[1])
        if distance < best_distance:
            best_distance = distance
            best = segment
    return best


def simulated_ball(segment: dict, drag: float) -> tuple[np.ndarray, float]:
    """The rolled-out ball at the end of the segment, and the time it took."""
    dt = segment["dt"]
    steps = len(segment["future"]) - 1
    times = np.asarray([steps * dt], dtype=np.float64)
    rolled = roll_position(segment["origin"], segment["velocity"], times, drag)
    return rolled[0], float(times[0])


def predicted_toucher(segment: dict, model, mean, std) -> dict | None:
    """Who the model thinks touches the ball next, and for which team."""
    if model is None:
        return None
    features, roster = player_features(segment, None)
    index = predict_index(model, features, mean, std)
    if index < 0 or index >= len(roster):
        return None
    return {"id": roster[index]["id"], "team": roster[index]["team"]}


def value_row(ball: tuple[float, float], possession: int, pitch: tuple[float, float]) -> dict:
    """A row the value model can score. The labels are placeholders."""
    return {
        "ball": (float(ball[0]), float(ball[1])),
        "possession_by_t": int(possession),
        "pitch": pitch,
        "y_for": 0,
        "y_against": 0,
    }


def build_states(
    pass_row: dict,
    segment: dict,
    touch_model,
    touch_mean,
    touch_std,
    drag: float,
    pitch: tuple[float, float],
    transform: str,
) -> dict | None:
    """The before state, the real after state and the simulated after state."""
    intent = pass_row.get("intent") or {}
    if segment.get("toucher") is None:
        return None
    team = pass_row.get("team_id")
    team_left = pass_row.get("team_left")
    if team is None or team_left is None or intent.get("ball_x") is None:
        return None
    attack_positive = team == team_left

    ball_before = convert(transform, intent["ball_x"], intent["ball_y"], pitch[0], pitch[1])

    end_raw = segment["future"][-1]
    end_event = convert(transform, end_raw[0], end_raw[1], pitch[0], pitch[1])

    rolled, elapsed = simulated_ball(segment, drag)
    rolled_event = convert(transform, rolled[0], rolled[1], pitch[0], pitch[1])

    real_toucher = segment["toucher"]
    real_team = real_toucher.get("team")
    predicted = predicted_toucher(segment, touch_model, touch_mean, touch_std)
    predicted_team = predicted.get("team") if predicted else None

    return {
        "match_id": pass_row.get("match_id"),
        "event_id": pass_row.get("event_id"),
        "success": pass_row.get("success"),
        "team_id": team,
        "reason": segment.get("reason"),
        "elapsed_s": elapsed,
        "ball_before": orient(ball_before, pitch, attack_positive),
        "ball_after_real": orient(end_event, pitch, attack_positive),
        "ball_after_sim": orient(rolled_event, pitch, attack_positive),
        "ball_position_error_m": float(np.linalg.norm(np.asarray(end_event) - np.asarray(rolled_event))),
        "possession_before": 1,
        "possession_after_real": 1 if real_team == team else 0,
        "possession_after_sim": 1 if predicted_team == team else 0,
        "real_toucher": real_toucher.get("id"),
        "real_toucher_team": real_team,
        "predicted_toucher": predicted.get("id") if predicted else None,
        "predicted_toucher_team": predicted_team,
        "toucher_correct": (predicted.get("id") == real_toucher.get("id")) if predicted else None,
        "pitch": pitch,
    }
