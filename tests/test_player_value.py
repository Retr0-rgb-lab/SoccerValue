import unittest
from pathlib import Path

from soccervalue.player_value import (
    candidate_features,
    defensive_lines,
    goal_geometry,
    line_features,
    read_goalkeepers,
)

PITCH = (105.0, 68.0)


def write_meta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<Document>
  <Teams>
    <Team TeamId="home" LineUp="4-2-3-1">
      <Players>
        <Player PersonId="h1" PlayingPosition="TW"/>
        <Player PersonId="h2" PlayingPosition="IV"/>
      </Players>
    </Team>
    <Team TeamId="away" LineUp="4-3-3">
      <Players>
        <Player PersonId="a1" PlayingPosition="TW"/>
        <Player PersonId="a2" PlayingPosition="IV"/>
      </Players>
    </Team>
  </Teams>
</Document>
""",
        encoding="utf-8",
    )


class GoalkeeperTests(unittest.TestCase):
    def test_the_goalkeeper_of_each_team_is_read_from_the_match_information(self):
        path = Path("data/cache/keeper_fixture.xml")
        write_meta(path)
        keepers = read_goalkeepers(path)
        self.assertEqual(keepers["home"], "h1")
        self.assertEqual(keepers["away"], "a1")


class DefensiveLineTests(unittest.TestCase):
    def test_players_close_together_form_one_line(self):
        points = [(95.0, 34.0), (90.0, 30.0), (88.0, 38.0), (60.0, 34.0), (55.0, 30.0)]
        ids = ["gk", "d1", "d2", "m1", "m2"]
        lines, offside = defensive_lines(points, ids, "gk")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["size"], 2)
        self.assertEqual(lines[1]["size"], 2)
        # the offside line is the second deepest defender
        self.assertAlmostEqual(offside, 90.0)

    def test_the_goalkeeper_is_left_out(self):
        points = [(95.0, 34.0), (90.0, 30.0)]
        ids = ["gk", "d1"]
        lines, _offside = defensive_lines(points, ids, "gk")
        self.assertEqual(lines[0]["size"], 1)
        self.assertAlmostEqual(lines[0]["x"], 90.0)

    def test_a_candidate_in_front_of_the_back_line(self):
        points = [(95.0, 34.0), (90.0, 30.0), (60.0, 34.0)]
        ids = ["gk", "d1", "m1"]
        lines, offside = defensive_lines(points, ids, "gk")
        features = line_features((95.0, 34.0), (50.0, 34.0), lines, offside)
        self.assertGreater(features["distance_to_last_line_m"], 0.0)
        # the candidate stands beyond both the back line and the midfield line
        self.assertEqual(features["lines_in_front"], 2.0)
        self.assertEqual(features["lines_crossed_by_the_pass"], 2.0)

    def test_a_candidate_behind_every_line(self):
        points = [(95.0, 34.0), (90.0, 30.0), (60.0, 34.0)]
        ids = ["gk", "d1", "m1"]
        lines, offside = defensive_lines(points, ids, "gk")
        features = line_features((40.0, 34.0), (30.0, 34.0), lines, offside)
        self.assertEqual(features["lines_in_front"], 0.0)
        self.assertLess(features["distance_to_last_line_m"], 0.0)


class GoalGeometryTests(unittest.TestCase):
    def test_a_candidate_in_the_box_is_close_and_sees_a_wide_angle(self):
        distance, angle = goal_geometry((100.0, 34.0), PITCH)
        self.assertLess(distance, 10.0)
        self.assertGreater(angle, 0.3)

    def test_a_candidate_far_away_sees_a_narrow_angle(self):
        _distance, angle = goal_geometry((10.0, 34.0), PITCH)
        self.assertLess(angle, 0.1)


class CandidateFeatureTests(unittest.TestCase):
    def test_the_feature_row_has_one_value_per_name(self):
        import math

        from soccervalue.intent import FEATURE_NAMES as INTENT_FEATURE_NAMES

        diagonal = math.hypot(PITCH[0], PITCH[1])
        table = [0.0] * len(INTENT_FEATURE_NAMES)
        column = {name: index for index, name in enumerate(INTENT_FEATURE_NAMES)}
        table[column["along_m"]] = 20.0 / diagonal
        table[column["perp_m"]] = 1.0 / diagonal
        table[column["distance_to_ball_m"]] = 15.0 / diagonal
        table[column["nearest_defender_to_candidate_m"]] = 4.0 / diagonal
        table[column["lane_clearance_m"]] = 2.0 / diagonal
        table[column["defenders_within_3m_of_lane"]] = 1.0
        table[column["candidate_speed_mps"]] = 5.0 / 10.0
        table[column["candidate_velocity_along_mps"]] = 3.0 / 10.0
        candidate = {"player": "t1", "xy": (80.0, 34.0), "features": table}
        row = {
            "ball": (60.0, 34.0),
            "pitch": PITCH,
            "defense": [(95.0, 34.0), (90.0, 30.0)],
            "defense_ids": ["gk", "d1"],
        }
        lines, offside = defensive_lines(row["defense"], row["defense_ids"], "gk")
        features = candidate_features(candidate, row, lines, offside)
        self.assertEqual(len(features), 16)
        self.assertAlmostEqual(features[0], 20.0, places=6)
        self.assertAlmostEqual(features[1], 1.0, places=6)
        self.assertAlmostEqual(features[2], 15.0, places=6)
        self.assertAlmostEqual(features[3], 4.0, places=6)
        self.assertAlmostEqual(features[5], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
