import unittest

import numpy as np

from soccervalue.pass_value import build_states, link_segment, orient, simulated_ball, value_row

PITCH = (105.0, 68.0)
TRANSFORM = "both_center"


def raw(event_x: float, event_y: float) -> tuple[float, float]:
    return (event_x - PITCH[0] / 2.0, event_y - PITCH[1] / 2.0)


def make_segment(toucher_team: str = "home", toucher: str = "t1") -> dict:
    origin = raw(30.0, 34.0)
    future = np.asarray(
        [raw(30.0 + 0.6 * step, 34.0) for step in range(11)],
        dtype=np.float64,
    )
    return {
        "origin": origin,
        "velocity": np.asarray([15.0, 0.0]),
        "future": future,
        "dt": 0.04,
        "reason": "touch",
        "roster": [
            {"id": "p1", "team": "home", "xy": origin, "velocity": np.zeros(2)},
            {"id": "t1", "team": "home", "xy": raw(50.0, 35.0), "velocity": np.zeros(2)},
            {"id": "d1", "team": "away", "xy": raw(40.0, 34.0), "velocity": np.zeros(2)},
        ],
        "players": future[:, None, :].repeat(3, axis=1),
        "toucher": {"id": toucher, "team": toucher_team},
        "start_index": 10,
    }


def make_pass(**extra) -> dict:
    base = {
        "match_id": "M1",
        "event_id": "e1",
        "success": 1,
        "team_id": "home",
        "team_left": "home",
        "intent": {"status": "ok", "ball_x": raw(30.0, 34.0)[0], "ball_y": raw(30.0, 34.0)[1]},
    }
    base.update(extra)
    return base


class PassValueTests(unittest.TestCase):
    def test_orient_flips_a_team_that_attacks_the_other_way(self):
        self.assertEqual(orient((20.0, 10.0), PITCH, True), (20.0, 10.0))
        self.assertEqual(orient((20.0, 10.0), PITCH, False), (85.0, 58.0))

    def test_link_segment_picks_the_flight_that_starts_at_the_kick(self):
        segment = make_segment()
        other = make_segment()
        other["origin"] = raw(70.0, 20.0)
        linked = link_segment(make_pass(), [other, segment], TRANSFORM, PITCH)
        self.assertIs(linked, segment)

    def test_link_segment_gives_up_when_no_flight_starts_near_the_kick(self):
        far = make_segment()
        far["origin"] = raw(70.0, 20.0)
        self.assertIsNone(link_segment(make_pass(), [far], TRANSFORM, PITCH))

    def test_the_simulated_ball_slows_down(self):
        segment = make_segment()
        rolled, elapsed = simulated_ball(segment, 0.3)
        self.assertAlmostEqual(elapsed, 0.4, places=6)
        # drag pulls the ball back from the constant velocity distance
        self.assertLess(rolled[0] - segment["origin"][0], 15.0 * 0.4)

    def test_the_states_carry_the_holder_of_the_ball(self):
        built = build_states(
            make_pass(), make_segment(toucher_team="home"), None, None, None, 0.3, PITCH, TRANSFORM
        )
        self.assertIsNotNone(built)
        self.assertEqual(built["possession_before"], 1)
        self.assertEqual(built["possession_after_real"], 1)
        self.assertAlmostEqual(built["ball_before"][0], 30.0, places=6)

    def test_an_opponent_touch_means_the_team_lost_the_ball(self):
        built = build_states(
            make_pass(), make_segment(toucher_team="away"), None, None, None, 0.3, PITCH, TRANSFORM
        )
        self.assertEqual(built["possession_after_real"], 0)

    def test_a_value_row_has_what_the_value_model_needs(self):
        row = value_row((30.0, 34.0), 1, PITCH)
        self.assertEqual(row["ball"], (30.0, 34.0))
        self.assertEqual(row["possession_by_t"], 1)
        self.assertIn("y_for", row)
        self.assertIn("y_against", row)


if __name__ == "__main__":
    unittest.main()
