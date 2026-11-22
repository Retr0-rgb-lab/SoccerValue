"""The aimed receiver, read from the kick instead of from the recorded result.

The recorded destination of a pass is often the place where the ball was won.
When the event repeats the kick coordinate, the next event supplies the point,
and on an unsuccessful pass that next event belongs to the opponent. The point
then encodes the outcome: a defender standing on the scored cell gives the
answer away.

This module estimates the target the passer aimed at. It uses the state at the
kick and the first moments of the ball flight only, so the estimate does not
depend on whether the pass was completed.

The rule. The ball velocity right after the kick gives a direction. Among the
teammates ahead of the ball along that direction, the one with the smallest
perpendicular distance to the ray is the aimed receiver.

Nothing here reads the label. The passer's team comes from the event, the
direction comes from the tracking, and the candidates are the same team as the
passer.

The module also writes out the candidate table for every pass: each eligible
teammate with the geometry a learned model needs. Every feature is relative,
so the table does not depend on which way the team attacks.
"""

from __future__ import annotations

import bisect

import numpy as np

from soccervalue.align import TRACKING_TO_EVENT, convert

# A kick is a jump in ball speed. Both thresholds are in metres per second.
KICK_SPEED_MIN = 5.0
KICK_JUMP_MIN = 3.0
# Frames scanned forward from the aligned frame while looking for the kick.
KICK_SEARCH_FRAMES = 45
# Frames averaged after the kick to get the direction. At 25 Hz this is 0.24 s,
# short enough to stay on the passer's aim and long enough to damp the contact.
DIRECTION_FRAMES = 6
# A receiver must be in front of the ball and within reach of the pass.
AHEAD_MIN_M = 0.5
MAX_AIM_M = 60.0
MAX_PERP_M = 15.0
# The candidate table for a learned model is wider than the rule's own cut, so
# the model can overrule the rule and the rule can still be scored on its own.
CANDIDATE_PERP_M = 25.0
# Seconds used to read a player's speed around the aligned frame.
VELOCITY_WINDOW_S = 0.2
# A lane is the segment from the ball to the candidate. A defender inside this
# radius of the lane is on the way.
LANE_RADIUS_M = 3.0
# The ball must move faster than this for the averaged direction to mean anything.
MIN_DIRECTION_SPEED = 3.0

STATUSES = ("ok", "no_frame", "no_kick", "slow_ball", "no_future", "no_candidate")

# The columns of the candidate table. Every one of them is relative to the ball
# and to the aim, so the table reads the same whichever way the team attacks.
FEATURE_NAMES = (
    "along_m",
    "perp_m",
    "along_over_distance",
    "distance_to_ball_m",
    "distance_to_passer_m",
    "candidate_speed_mps",
    "candidate_velocity_along_mps",
    "candidate_velocity_perp_mps",
    "nearest_defender_to_candidate_m",
    "nearest_defender_to_ball_m",
    "defenders_within_3m_of_lane",
    "lane_clearance_m",
    "candidates_closer_to_ball",
    "perp_rank",
    "is_closest_to_the_ray",
    "kick_speed_mps",
    "passer_speed_mps",
)


class IntentTracker:
    """Ball flight for one match, and the aimed receiver for one pass."""

    def __init__(
        self,
        ball: list[tuple[float, float, float]],
        pitch_x: float,
        pitch_y: float,
        players: dict | None = None,
    ) -> None:
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.players = players or {}
        self.player_times = {
            person: [row[0] for row in rows] for person, rows in self.players.items()
        }
        self.points = np.asarray(
            [convert(TRACKING_TO_EVENT, row[1], row[2], pitch_x, pitch_y) for row in ball],
            dtype=np.float64,
        )
        self.times = np.asarray([row[0] for row in ball], dtype=np.float64)
        if len(self.times) > 1:
            self.velocity = np.diff(self.points, axis=0) / np.diff(self.times)[:, None]
        else:
            self.velocity = np.zeros((0, 2), dtype=np.float64)
        self.speed = np.linalg.norm(self.velocity, axis=1)

    def kick_direction(self, frame_index: int) -> tuple[np.ndarray | None, int | None, float | None]:
        """The direction the ball left in, the kick frame, and the speed."""
        if frame_index is None or frame_index < 1 or frame_index >= len(self.speed) - 1:
            return None, None, None
        kick = None
        limit = min(frame_index + KICK_SEARCH_FRAMES, len(self.speed) - 1)
        for index in range(frame_index, limit):
            if index >= 3 and self.speed[index] - self.speed[index - 3] > KICK_JUMP_MIN and self.speed[index] > KICK_SPEED_MIN:
                kick = index
                break
        if kick is None:
            return None, None, None
        if kick + DIRECTION_FRAMES + 1 >= len(self.points):
            return None, kick, None
        direction = self.velocity[kick + 1 : kick + 1 + DIRECTION_FRAMES].mean(axis=0)
        speed = float(np.linalg.norm(direction))
        if speed < MIN_DIRECTION_SPEED:
            return None, kick, speed
        return direction / speed, kick, speed

    def estimate(self, item: dict, frame_index: int | None, snapshot: dict) -> dict:
        """Return the aimed receiver for one pass.

        `snapshot` maps person id to (tracking x, tracking y, team) at the
        aligned frame. The result never contains the pass outcome.
        """
        result = {
            "status": "no_frame",
            "target_x": None,
            "target_y": None,
            "player": None,
            "perp_m": None,
            "along_m": None,
            "speed_mps": None,
            "kick_index": None,
            "annotated_x": None,
            "annotated_y": None,
            "candidates": [],
        }
        if frame_index is None:
            return result
        direction, kick, speed = self.kick_direction(frame_index)
        if direction is None:
            if kick is None:
                result["status"] = "no_kick"
            elif speed is not None:
                result["status"] = "slow_ball"
            else:
                result["status"] = "no_future"
            return result

        ball_point = self.points[frame_index]
        passer = snapshot.get(item.get("player_id"))
        if passer is None:
            result["status"] = "no_candidate"
            return result
        team = item.get("team_id") or passer[2]

        # Where the annotated recipient stood at the kick. On a successful pass
        # that is the teammate who received the ball, on an unsuccessful one the
        # opponent who intercepted it. It is the result, not the aim, and it is
        # only recorded so that a round can measure how far a scored endpoint
        # sits from where the ball actually ended up.
        annotated = snapshot.get(item.get("recipient"))
        if annotated is not None:
            ax, ay = convert(TRACKING_TO_EVENT, annotated[0], annotated[1], self.pitch_x, self.pitch_y)
            result["annotated_x"] = ax
            result["annotated_y"] = ay

        best = None
        defenders = [
            convert(TRACKING_TO_EVENT, x, y, self.pitch_x, self.pitch_y)
            for person, (x, y, person_team) in snapshot.items()
            if person_team != team
        ]
        table = []
        for person, (x, y, person_team) in snapshot.items():
            if person == item.get("player_id") or person_team != team:
                continue
            px, py = convert(TRACKING_TO_EVENT, x, y, self.pitch_x, self.pitch_y)
            offset_x = px - ball_point[0]
            offset_y = py - ball_point[1]
            along = float(offset_x * direction[0] + offset_y * direction[1])
            perp = float(abs(offset_x * direction[1] - offset_y * direction[0]))
            if along <= AHEAD_MIN_M or along > MAX_AIM_M or perp > CANDIDATE_PERP_M:
                continue
            table.append({"player": person, "along_m": along, "perp_m": perp, "xy": (px, py)})
            if perp > MAX_PERP_M:
                continue
            if best is None or perp < best[1] - 1e-9 or (abs(perp - best[1]) <= 1e-9 and along < best[2]):
                best = (person, perp, along, px, py)

        result["candidates"] = self._feature_rows(
            item, table, direction, speed, snapshot, ball_point, defenders, float(self.times[frame_index])
        )

        if best is None:
            result["status"] = "no_candidate"
            return result
        result.update(
            {
                "status": "ok",
                "target_x": best[3],
                "target_y": best[4],
                "player": best[0],
                "perp_m": best[1],
                "along_m": best[2],
                "speed_mps": speed,
                "kick_index": kick,
            }
        )
        return result

    def _feature_rows(
        self,
        item: dict,
        table: list[dict],
        direction: np.ndarray,
        speed: float,
        snapshot: dict,
        ball_point: np.ndarray,
        defenders: list[tuple[float, float]],
        stamp: float,
    ) -> list[dict]:
        """One row per candidate teammate, with the geometry a model needs."""
        if not table:
            return []
        diagonal = float(np.hypot(self.pitch_x, self.pitch_y))
        passer = snapshot.get(item.get("player_id"))
        passer_xy = None
        if passer is not None:
            passer_xy = (
                convert(TRACKING_TO_EVENT, passer[0], passer[1], self.pitch_x, self.pitch_y),
                self._velocity(item.get("player_id"), stamp),
            )
        nearest_defender_to_ball = min(
            (float(np.hypot(x - ball_point[0], y - ball_point[1])) for x, y in defenders),
            default=diagonal,
        )
        by_distance = sorted(
            table,
            key=lambda row: float(np.hypot(row["xy"][0] - ball_point[0], row["xy"][1] - ball_point[1])),
        )
        rank_by_distance = {row["player"]: index for index, row in enumerate(by_distance)}
        by_perp = sorted(table, key=lambda row: row["perp_m"])
        rank_by_perp = {row["player"]: index for index, row in enumerate(by_perp)}
        closest = by_perp[0]["player"] if by_perp else None

        rows = []
        for row in table:
            player = row["player"]
            px, py = row["xy"]
            offset_x = px - ball_point[0]
            offset_y = py - ball_point[1]
            distance = float(np.hypot(offset_x, offset_y))
            along = row["along_m"]
            perp = row["perp_m"]
            velocity = self._velocity(player, stamp)
            along_velocity = float(velocity[0] * direction[0] + velocity[1] * direction[1])
            perp_velocity = float(abs(velocity[0] * direction[1] - velocity[1] * direction[0]))
            nearest_defender = min(
                (float(np.hypot(x - px, y - py)) for x, y in defenders),
                default=diagonal,
            )
            lane = _lane(ball_point[0], ball_point[1], px, py, defenders)
            distance_to_passer = diagonal
            passer_speed = 0.0
            if passer_xy is not None:
                distance_to_passer = float(np.hypot(px - passer_xy[0][0], py - passer_xy[0][1]))
                passer_speed = float(np.hypot(*passer_xy[1]))
            rows.append(
                {
                    "player": player,
                    "features": [
                        along / diagonal,
                        perp / diagonal,
                        along / max(distance, 1e-6),
                        distance / diagonal,
                        distance_to_passer / diagonal,
                        float(np.hypot(*velocity)) / 10.0,
                        along_velocity / 10.0,
                        perp_velocity / 10.0,
                        nearest_defender / diagonal,
                        nearest_defender_to_ball / diagonal,
                        float(lane["count"]),
                        lane["clearance"] / diagonal,
                        float(rank_by_distance[player]),
                        rank_by_perp[player] / max(len(table), 1),
                        1.0 if player == closest else 0.0,
                        speed / 20.0,
                        passer_speed / 10.0,
                    ],
                }
            )
        return rows

    def _velocity(self, person: str | None, stamp: float | None) -> tuple[float, float]:
        """Speed of one player around the aligned frame, in metres per second."""
        if person is None or stamp is None:
            return (0.0, 0.0)
        rows = self.players.get(person)
        if not rows:
            return (0.0, 0.0)
        times = self.player_times[person]
        now = _at_time(rows, times, stamp)
        before = _at_time(rows, times, stamp - VELOCITY_WINDOW_S)
        if now is None or before is None:
            return (0.0, 0.0)
        span = max(stamp - before[0], 1e-3)
        return ((now[1] - before[1]) / span, (now[2] - before[2]) / span)


def _at_time(rows: list, times: list, stamp: float, tolerance: float = 0.06):
    index = bisect.bisect_left(times, stamp)
    best = None
    gap = tolerance
    for candidate in (index - 1, index):
        if 0 <= candidate < len(times):
            distance = abs(times[candidate] - stamp)
            if distance <= gap:
                best = rows[candidate]
                gap = distance
    return best


def _lane(ball_x: float, ball_y: float, end_x: float, end_y: float, defenders: list[tuple[float, float]]) -> dict:
    """How many defenders sit on the lane and how much clearance it has."""
    count = 0
    clearance = float("inf")
    for x, y in defenders:
        distance = _segment_distance(x, y, ball_x, ball_y, end_x, end_y)
        if distance <= LANE_RADIUS_M:
            count += 1
        clearance = min(clearance, distance)
    return {"count": count, "clearance": clearance if clearance < float("inf") else 1e3}


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    ab_x = bx - ax
    ab_y = by - ay
    length_sq = ab_x * ab_x + ab_y * ab_y
    if length_sq < 1e-8:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * ab_x + (py - ay) * ab_y) / length_sq))
    return float(np.hypot(px - (ax + t * ab_x), py - (ay + t * ab_y)))


def perp_distance(point: tuple[float, float], origin: tuple[float, float], direction: tuple[float, float]) -> float:
    """Distance from a point to the line through the origin along the direction."""
    offset_x = point[0] - origin[0]
    offset_y = point[1] - origin[1]
    return abs(offset_x * direction[1] - offset_y * direction[0])


def along_distance(point: tuple[float, float], origin: tuple[float, float], direction: tuple[float, float]) -> float:
    """Distance from the origin to the projection of the point on the direction."""
    offset_x = point[0] - origin[0]
    offset_y = point[1] - origin[1]
    return offset_x * direction[0] + offset_y * direction[1]
