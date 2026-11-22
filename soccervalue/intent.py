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
"""

from __future__ import annotations

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
# The ball must move faster than this for the averaged direction to mean anything.
MIN_DIRECTION_SPEED = 3.0

STATUSES = ("ok", "no_frame", "no_kick", "slow_ball", "no_future", "no_candidate")


class IntentTracker:
    """Ball flight for one match, and the aimed receiver for one pass."""

    def __init__(self, ball: list[tuple[float, float, float]], pitch_x: float, pitch_y: float) -> None:
        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
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
        for person, (x, y, person_team) in snapshot.items():
            if person == item.get("player_id") or person_team != team:
                continue
            px, py = convert(TRACKING_TO_EVENT, x, y, self.pitch_x, self.pitch_y)
            offset_x = px - ball_point[0]
            offset_y = py - ball_point[1]
            along = float(offset_x * direction[0] + offset_y * direction[1])
            perp = float(abs(offset_x * direction[1] - offset_y * direction[0]))
            if along <= AHEAD_MIN_M or along > MAX_AIM_M or perp > MAX_PERP_M:
                continue
            if best is None or perp < best[1] - 1e-9 or (abs(perp - best[1]) <= 1e-9 and along < best[2]):
                best = (person, perp, along, px, py)

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
