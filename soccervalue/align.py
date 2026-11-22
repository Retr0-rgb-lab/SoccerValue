"""Put tracking coordinates into the event coordinate system, then face one goal."""

from __future__ import annotations

import math

TRANSFORMS = ("identity", "y_center", "both_center")
# Tracking X/Y are centered on the pitch. Event X/Y start at a corner.
TRACKING_TO_EVENT = "both_center"
SYNC_WINDOW_S = 6.0
# Meters added to the cost for each second away from the annotated timestamp.
SYNC_TIME_WEIGHT = 1.5

# After synchronization the public dataset itself reports about 2.6 m average
# ball error. The raw timestamps average about 9 m, so 3 m before sync is not
# a meaningful stop.
ALIGN_MEDIAN_LIMIT_M = 4.0
PASSER_MEDIAN_LIMIT_M = 3.0


def convert(name: str, x: float, y: float, pitch_x: float, pitch_y: float) -> tuple[float, float]:
    if name == "identity":
        return x, y
    if name == "y_center":
        return x, y + pitch_y / 2
    if name == "both_center":
        return x + pitch_x / 2, y + pitch_y / 2
    raise KeyError(name)


def frame_cost(ball_event_m: float, passer_ball_m: float, dt_s: float) -> float:
    """Prefer a nearby frame where the ball matches the event and the passer is on it."""
    return ball_event_m + passer_ball_m + SYNC_TIME_WEIGHT * abs(dt_s)


def choose_transform(samples: list[dict], pitch_x: float, pitch_y: float) -> str:
    """Pick the tracking-to-event shift with the smallest median error on these samples."""
    best_name = "identity"
    best_error = math.inf
    for name in TRANSFORMS:
        errors = []
        for sample in samples:
            ball = sample.get("track_ball")
            if ball is None:
                continue
            bx, by = convert(name, ball[0], ball[1], pitch_x, pitch_y)
            errors.append(math.hypot(bx - sample["start_x"], by - sample["start_y"]))
        if not errors:
            continue
        errors.sort()
        median = errors[len(errors) // 2]
        if median < best_error:
            best_name = name
            best_error = median
    return best_name


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def alignment_report(samples: list[dict], transform: str, pitch_x: float, pitch_y: float) -> dict:
    ball_errors = []
    passer_errors = []
    missing = 0
    for sample in samples:
        ball = sample.get("track_ball")
        if ball is None:
            missing += 1
            continue
        bx, by = convert(transform, ball[0], ball[1], pitch_x, pitch_y)
        ball_errors.append(math.hypot(bx - sample["start_x"], by - sample["start_y"]))
        passer = sample.get("track_passer")
        if passer is not None:
            px, py = convert(transform, passer[0], passer[1], pitch_x, pitch_y)
            passer_errors.append(math.hypot(px - bx, py - by))
    signed = [float(sample["frame_signed_dt"]) for sample in samples if sample.get("frame_signed_dt") is not None]
    ball_median = median(ball_errors)
    passer_median = median(passer_errors)
    return {
        "transform": transform,
        "n": len(samples),
        "n_with_ball": len(ball_errors),
        "n_missing_frame": missing,
        "ball_vs_event_median_m": ball_median,
        "passer_vs_ball_median_m": passer_median,
        "ball_vs_event_over_2m": _fraction_over(ball_errors, 2.0),
        "ball_vs_event_over_5m": _fraction_over(ball_errors, 5.0),
        "frame_signed_dt_median_s": median(signed),
        "fraction_frame_after_event": _fraction_over(signed, 0.0) if signed else None,
        "fraction_frame_more_than_1s_after": _fraction_over(signed, 1.0) if signed else None,
        "usable": ball_median is not None
        and passer_median is not None
        and ball_median <= ALIGN_MEDIAN_LIMIT_M
        and passer_median <= PASSER_MEDIAN_LIMIT_M,
    }


def _fraction_over(values: list[float], limit: float) -> float | None:
    if not values:
        return None
    return sum(value > limit for value in values) / len(values)


def orient_samples(samples: list[dict], transform: str, pitch_x: float, pitch_y: float, end_mode: str = "recorded") -> list[dict]:
    """Return model rows in a frame where the passing team attacks toward +x.

    `end_mode` is "recorded" for the point the event file gives, or "intent" for
    the teammate the passer aimed at. Both rows carry the recorded point as well,
    so the two arms can be compared on the same passes.
    """
    rows = []
    for sample in samples:
        ball = sample.get("track_ball")
        players = sample.get("track_players")
        if ball is None or not players:
            continue
        bx, by = convert(transform, ball[0], ball[1], pitch_x, pitch_y)
        ball_event_m = math.hypot(bx - sample["start_x"], by - sample["start_y"])
        if ball_event_m > 8:
            continue
        passer = sample.get("track_passer")
        if passer is not None:
            px, py = convert(transform, passer[0], passer[1], pitch_x, pitch_y)
            if math.hypot(px - bx, py - by) > 5:
                continue
        attack_positive = sample["team_id"] == sample["team_left"]
        ball_xy = _orient(convert(transform, ball[0], ball[1], pitch_x, pitch_y), pitch_x, pitch_y, attack_positive)
        start = _orient((sample["start_x"], sample["start_y"]), pitch_x, pitch_y, attack_positive)
        recorded_end = _orient((sample["end_x"], sample["end_y"]), pitch_x, pitch_y, attack_positive)
        intent = sample.get("intent") or {}
        if end_mode == "intent":
            if intent.get("status") != "ok":
                continue
            end = _orient((intent["target_x"], intent["target_y"]), pitch_x, pitch_y, attack_positive)
            end_source = "intent"
        else:
            end = recorded_end
            end_source = sample.get("end_source")
        if intent.get("annotated_x") is not None:
            annotated_xy = _orient((intent["annotated_x"], intent["annotated_y"]), pitch_x, pitch_y, attack_positive)
        else:
            annotated_xy = None
        attack = []
        defense = []
        for _person, (x, y, team) in players.items():
            point = _orient(convert(transform, x, y, pitch_x, pitch_y), pitch_x, pitch_y, attack_positive)
            if team == sample["team_id"]:
                attack.append(point)
            else:
                defense.append(point)
        rows.append(
            {
                "match_id": sample["match_id"],
                "event_id": sample.get("event_id"),
                "success": sample["success"],
                "start": start,
                "end": end,
                "recorded_end": recorded_end,
                "ball": ball_xy,
                "attack": attack,
                "defense": defense,
                "pitch": (pitch_x, pitch_y),
                "end_source": end_source,
                "recorded_end_source": sample.get("end_source"),
                "recipient": sample.get("recipient"),
                "next_event_relation": sample.get("next_event_relation"),
                "next_event_tag": sample.get("next_event_tag"),
                "end_gap_s": sample.get("end_gap_s"),
                "frame_signed_dt": sample.get("frame_signed_dt"),
                "ball_event_m": ball_event_m,
                "intent_player": intent.get("player"),
                "intent_perp_m": intent.get("perp_m"),
                "intent_along_m": intent.get("along_m"),
                "intent_status": intent.get("status"),
                "intent_candidates": intent.get("candidates") or [],
                "annotated_xy": annotated_xy,
            }
        )
    return rows


def _orient(point: tuple[float, float], pitch_x: float, pitch_y: float, attack_positive: bool) -> tuple[float, float]:
    x, y = point
    if attack_positive:
        return x, y
    return pitch_x - x, pitch_y - y
