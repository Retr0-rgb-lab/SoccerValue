"""A position value surface over coarse zones.

The shot label is sparse. A shot happens in about one state in fifty, and the
side that rarely shoots has too few events for the relation to be identified.
A surface that only says where the ball is does not have that problem: every
state contributes its window xG to the zone it sits in, and the value of a zone
is the average of what happens from there.

The surface is a function of the ball position and of who holds it, and of
nothing else. It never reads whether a pass was completed, so it can value a
pass that was never played. That is the point of building it.

Eight zones: four bands along the length and two across it. The lateral split
puts the two half spaces in the middle band, because that is where a pass
carries more threat for the same forward distance.
"""

from __future__ import annotations

import numpy as np

# The length is cut at the thirds and at the edge of the penalty area.
DEPTH_EDGES = (35.0, 70.0, 88.5)
# The width is cut so that the middle third is the two half spaces.
LATERAL_EDGE = 11.333333333333334

DEPTH_NAMES = ("own_third", "middle_third", "attacking_third", "opponent_box")
LATERAL_NAMES = ("wide", "centre")
ZONE_COUNT = len(DEPTH_NAMES) * len(LATERAL_NAMES)


def depth_band(x: float, pitch_x: float) -> int:
    """0 is the team's own end, 3 is the opponent's penalty area."""
    fraction = x / pitch_x
    if fraction * pitch_x < DEPTH_EDGES[0]:
        return 0
    if fraction * pitch_x < DEPTH_EDGES[1]:
        return 1
    if fraction * pitch_x < DEPTH_EDGES[2]:
        return 2
    return 3


def lateral_band(y: float, pitch_y: float) -> int:
    """0 is wide, 1 is the centre band that holds the two half spaces."""
    return 1 if abs(y - pitch_y / 2.0) <= LATERAL_EDGE else 0


def zone_of(x: float, y: float, pitch: tuple[float, float]) -> int:
    pitch_x, pitch_y = pitch
    return depth_band(x, pitch_x) * len(LATERAL_NAMES) + lateral_band(y, pitch_y)


def zone_name(index: int) -> str:
    depth = index // len(LATERAL_NAMES)
    lateral = index % len(LATERAL_NAMES)
    return f"{DEPTH_NAMES[depth]}_{LATERAL_NAMES[lateral]}"


class ZoneSurface:
    """The average window value of every zone, for each holder."""

    def __init__(self) -> None:
        self.values: dict[tuple[int, int], float] = {}
        self.counts: dict[tuple[int, int], int] = {}

    def fit(self, rows: list[dict], pitch: tuple[float, float]) -> "ZoneSurface":
        totals: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        for row in rows:
            zone = zone_of(row["ball"][0], row["ball"][1], pitch)
            key = (zone, int(row["possession_by_t"]))
            value = float(row.get("xg_for_sum", 0.0)) - float(row.get("xg_against_sum", 0.0))
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
        self.values = {key: totals[key] / counts[key] for key in totals}
        self.counts = counts
        return self

    def value(self, x: float, y: float, possession: int, pitch: tuple[float, float]) -> float:
        key = (zone_of(x, y, pitch), int(possession))
        if key in self.values:
            return self.values[key]
        # fall back to the other holder before falling back to nothing
        other = (key[0], 1 - key[1])
        return self.values.get(other, 0.0)

    def delta(self, before: tuple[float, float], after: tuple[float, float], possession_after: int, pitch: tuple[float, float]) -> float:
        """The value a pass changes, from one ball position to another."""
        return self.value(after[0], after[1], possession_after, pitch) - self.value(
            before[0], before[1], 1, pitch
        )

    def shape(self, pitch: tuple[float, float]) -> dict:
        """The surface by depth band and by lateral band, for reading."""
        out = {"by_zone": {}, "by_depth": {}, "by_lateral": {}}
        for index in range(ZONE_COUNT):
            key = (index, 1)
            out["by_zone"][zone_name(index)] = {
                "value_with_possession": self.values.get(key),
                "states": self.counts.get(key, 0),
            }
        for depth, name in enumerate(DEPTH_NAMES):
            values = [
                self.values.get((depth * len(LATERAL_NAMES) + lateral, 1))
                for lateral in range(len(LATERAL_NAMES))
            ]
            kept = [value for value in values if value is not None]
            out["by_depth"][name] = {
                "wide": values[0],
                "centre": values[1],
                "centre_minus_wide": (values[1] - values[0]) if kept and values[0] is not None and values[1] is not None else None,
            }
        for lateral, name in enumerate(LATERAL_NAMES):
            values = [
                self.values.get((depth * len(LATERAL_NAMES) + lateral, 1))
                for depth in range(len(DEPTH_NAMES))
            ]
            kept = [value for value in values if value is not None]
            out["by_lateral"][name] = {
                "mean_over_depth": float(np.mean(kept)) if kept else None,
            }
        return out


def surface_from_rows(rows: list[dict], pitch: tuple[float, float]) -> ZoneSurface:
    return ZoneSurface().fit(rows, pitch)
