"""Player level differences between a receiver and the other candidates.

The position surface values a zone and a zone cannot pick a player. This module
asks what the player who received the ball has that the other candidates do
not, and it adds the opponent's defensive lines as a feature.

The defensive lines are read from the tracking. The goalkeeper comes from the
match information file, which lists the playing position of every player, so
nobody has to be guessed as the deepest player on the pitch.

Everything here is measured at the kick. No outcome is read.
"""

from __future__ import annotations

import math

import numpy as np
from lxml import etree

from soccervalue.intent import FEATURE_NAMES as INTENT_FEATURE_NAMES

# A gap between two defenders bigger than this starts a new line.
LINE_GAP_M = 10.0
# The goal is this wide.
GOAL_WIDTH_M = 7.32

FEATURE_NAMES = (
    "along_m",
    "perp_m",
    "distance_to_ball_m",
    "nearest_defender_to_candidate_m",
    "lane_clearance_m",
    "defenders_within_3m_of_lane",
    "candidate_speed_mps",
    "velocity_along_ray_mps",
    "distance_to_opponent_goal_m",
    "goal_angle_rad",
    "lateral_from_centre_m",
    "distance_to_last_line_m",
    "lines_in_front",
    "lines_crossed_by_the_pass",
    "gap_to_next_line_m",
    "offside_line_distance_m",
)


def read_goalkeepers(meta_path) -> dict[str, str]:
    """The person id of each team's goalkeeper, from the match information."""
    root = etree.parse(str(meta_path)).getroot()
    keepers: dict[str, str] = {}
    for team in root.iter("Team"):
        team_id = team.get("TeamId")
        for player in team.iter("Player"):
            if (player.get("PlayingPosition") or "").upper() == "TW" and team_id:
                keepers[team_id] = player.get("PersonId")
                break
    return keepers


def defensive_lines(points: list, ids: list, goalkeeper: str | None) -> tuple[list[dict], float | None]:
    """Group the defending outfield players into lines.

    The points are in the attacking frame, so the biggest x is the defender
    closest to his own goal. The offside line is the second deepest defender,
    which is the rule the game itself uses.
    """
    pairs = [
        (float(point[0]), person)
        for point, person in zip(points, ids)
        if person != goalkeeper
    ]
    if not pairs:
        return ([], None)
    all_pairs = sorted(
        ((float(point[0]), person) for point, person in zip(points, ids)), key=lambda item: -item[0]
    )
    offside = all_pairs[1][0] if len(all_pairs) > 1 else None

    pairs.sort(key=lambda item: -item[0])
    lines: list[list[float]] = [[pairs[0][0]]]
    for x, _person in pairs[1:]:
        if lines[-1][0] - x <= LINE_GAP_M:
            lines[-1].append(x)
        else:
            lines.append([x])
    described = [
        {"x": float(np.mean(members)), "deepest": max(members), "size": len(members)}
        for members in lines
    ]
    return (described, offside)


def line_features(candidate: tuple[float, float], ball: tuple[float, float], lines: list[dict], offside: float | None) -> dict:
    """Where a candidate stands relative to the opponent's lines."""
    x = candidate[0]
    in_front = [line for line in lines if x > line["x"]]
    crossed = [
        line
        for line in lines
        if min(x, ball[0]) < line["deepest"] < max(x, ball[0])
    ]
    gaps = [line["x"] - x for line in in_front]
    return {
        "distance_to_last_line_m": (x - lines[0]["x"]) if lines else 0.0,
        "lines_in_front": float(len(in_front)),
        "lines_crossed_by_the_pass": float(len(crossed)),
        "gap_to_next_line_m": min(gaps) if gaps else 0.0,
        "offside_line_distance_m": (x - offside) if offside is not None else 0.0,
    }


def goal_geometry(candidate: tuple[float, float], pitch: tuple[float, float]) -> tuple[float, float]:
    """The distance to the opponent's goal and the angle the goal subtends."""
    pitch_x, pitch_y = pitch
    goal_x = pitch_x
    goal_y = pitch_y / 2.0
    half = GOAL_WIDTH_M / 2.0
    dx = goal_x - candidate[0]
    upper = (dx, goal_y + half - candidate[1])
    lower = (dx, goal_y - half - candidate[1])
    distance = math.hypot(dx, goal_y - candidate[1])
    cross = upper[0] * lower[1] - upper[1] * lower[0]
    dot = upper[0] * lower[0] + upper[1] * lower[1]
    angle = abs(math.atan2(cross, dot))
    return (distance, angle)


def candidate_features(candidate: dict, row: dict, lines: list[dict], offside: float | None) -> list[float]:
    """One feature row per candidate, all of it measured at the kick.

    The columns of the candidate table the intent module builds are looked up
    by name, so the order of that table cannot silently shift the features.
    """
    pitch = row["pitch"]
    diagonal = math.hypot(pitch[0], pitch[1])
    table = candidate.get("features") or []
    point = candidate.get("xy")
    ball = row["ball"]
    if point is None or len(table) != len(INTENT_FEATURE_NAMES):
        return []
    column = {name: index for index, name in enumerate(INTENT_FEATURE_NAMES)}

    def metres(name: str, scale: float = 1.0) -> float:
        return float(table[column[name]]) * scale

    relative = line_features(point, ball, lines, offside)
    distance, angle = goal_geometry(point, pitch)
    return [
        metres("along_m", diagonal),
        metres("perp_m", diagonal),
        metres("distance_to_ball_m", diagonal),
        metres("nearest_defender_to_candidate_m", diagonal),
        metres("lane_clearance_m", diagonal),
        float(table[column["defenders_within_3m_of_lane"]]),
        metres("candidate_speed_mps", 10.0),
        metres("candidate_velocity_along_mps", 10.0),
        distance,
        angle,
        abs(point[1] - pitch[1] / 2.0),
        relative["distance_to_last_line_m"],
        relative["lines_in_front"],
        relative["lines_crossed_by_the_pass"],
        relative["gap_to_next_line_m"],
        relative["offside_line_distance_m"],
    ]
