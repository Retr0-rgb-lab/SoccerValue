import unittest

import numpy as np

from soccervalue.value_model import SimpleStateValue, SymmetricStateValue

PITCH = (105.0, 68.0)


def make_rows(n: int = 240) -> list[dict]:
    """States where the team that holds the ball is the one that shoots."""
    rows = []
    for index in range(n):
        possession = index % 2
        ball = (30.0 + possession * 20.0, 34.0)
        rows.append(
            {
                "ball": ball,
                "possession_by_t": possession,
                "pitch": PITCH,
                "y_for": possession,
                "y_against": 1 - possession,
            }
        )
    return rows


class SymmetricValueTests(unittest.TestCase):
    def test_the_ownership_slope_is_positive(self):
        model = SymmetricStateValue("slopes")
        model.fit(make_rows())
        coefficients = model.coefficients()
        self.assertIn("possession_own", coefficients)
        self.assertGreater(coefficients["possession_own"], 0.0)

    def test_the_opponent_head_reads_the_mirrored_frame(self):
        model = SymmetricStateValue("all")
        model.fit(make_rows())
        here = {"ball": (30.0, 34.0), "possession_by_t": 1, "pitch": PITCH, "y_for": 0, "y_against": 0}
        mirrored = {"ball": (75.0, 34.0), "possession_by_t": 0, "pitch": PITCH, "y_for": 0, "y_against": 0}
        predicted = model.predict([here, mirrored])
        self.assertAlmostEqual(predicted[0][1], predicted[1][0], places=10)

    def test_the_shared_slopes_keep_one_level_per_team(self):
        model = SymmetricStateValue("slopes")
        model.fit(make_rows())
        here = {"ball": (30.0, 34.0), "possession_by_t": 1, "pitch": PITCH, "y_for": 0, "y_against": 0}
        mirrored = {"ball": (75.0, 34.0), "possession_by_t": 0, "pitch": PITCH, "y_for": 0, "y_against": 0}
        predicted = model.predict([here, mirrored])
        level = model.coefficients()["team_level"]
        difference = np.log(predicted[0][1] / (1 - predicted[0][1])) - np.log(
            predicted[1][0] / (1 - predicted[1][0])
        )
        self.assertAlmostEqual(float(difference), level, places=6)

    def test_the_team_holds_the_ball_means_a_higher_value(self):
        model = SymmetricStateValue("slopes")
        model.fit(make_rows())
        held = model.predict(
            [{"ball": (30.0, 34.0), "possession_by_t": 1, "pitch": PITCH, "y_for": 0, "y_against": 0}]
        )[0]
        lost = model.predict(
            [{"ball": (30.0, 34.0), "possession_by_t": 0, "pitch": PITCH, "y_for": 0, "y_against": 0}]
        )[0]
        self.assertGreater(held[0] - held[1], lost[0] - lost[1])

    def test_the_shared_mode_keeps_one_level(self):
        model = SymmetricStateValue("all")
        model.fit(make_rows())
        self.assertNotIn("team_level", model.coefficients())
        self.assertIn("possession_own", model.coefficients())

    def test_the_separate_heads_can_weigh_ownership_differently(self):
        """The reason this round exists: two heads fitted apart may disagree."""
        rows = make_rows()
        rows += [
            {
                "ball": (80.0, 60.0),
                "possession_by_t": 0,
                "pitch": PITCH,
                "y_for": 0,
                "y_against": 0,
            }
            for _ in range(60)
        ]
        separate = SimpleStateValue()
        separate.fit(rows)
        symmetric = SymmetricStateValue("slopes")
        symmetric.fit(rows)
        self.assertEqual(len(separate.for_model.coef_[0]), 5)
        self.assertEqual(len(symmetric.coefficients()), 4)


if __name__ == "__main__":
    unittest.main()
