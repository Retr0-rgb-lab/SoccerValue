import unittest

import numpy as np

from soccervalue.run_candidate_pass import candidate_scores, rank_hit, zone_hit
from soccervalue.surface import ZoneSurface

PITCH = (105.0, 68.0)


class FakeSuccessModel:
    """Returns a fixed probability per candidate, in the order given."""

    def __init__(self, values):
        self.values = values
        self.seen = []

    def predict(self, rows):
        self.seen.extend(rows)
        return np.asarray(self.values, dtype=np.float64)


def make_row():
    return {
        "ball": (50.0, 34.0),
        "attack": [(50.0, 34.0), (90.0, 34.0)],
        "defense": [(60.0, 10.0)],
        "pitch": PITCH,
        "success": 1,
        "intent_candidates": [
            {"player": "t1", "xy": (90.0, 34.0)},
            {"player": "t2", "xy": (60.0, 5.0)},
        ],
    }


def make_surface():
    rows = [
        {"ball": (90.0, 34.0), "possession_by_t": 1, "xg_for_sum": 0.5, "xg_against_sum": 0.0},
        {"ball": (60.0, 5.0), "possession_by_t": 1, "xg_for_sum": 0.05, "xg_against_sum": 0.0},
    ]
    return ZoneSurface().fit(rows, PITCH)


class CandidateScoreTests(unittest.TestCase):
    def test_every_candidate_gets_the_three_scores(self):
        surface = make_surface()
        model = FakeSuccessModel([0.8, 0.2])
        scored = candidate_scores(make_row(), surface, model, PITCH)
        self.assertEqual(len(scored), 2)
        self.assertEqual(scored[0]["player"], "t1")
        self.assertAlmostEqual(scored[0]["probability"], 0.8)
        self.assertGreater(scored[0]["value"], scored[1]["value"])
        self.assertGreater(scored[0]["combined"], scored[1]["combined"])

    def test_the_combination_is_the_probability_weighted_value(self):
        surface = make_surface()
        model = FakeSuccessModel([1.0, 0.0])
        scored = candidate_scores(make_row(), surface, model, PITCH)
        first = scored[0]
        expected = (
            first["probability"] * first["value"]
            + (1.0 - first["probability"]) * first["value_against"]
            - (first["value"] - first["delta_value"])
        )
        self.assertAlmostEqual(first["combined"], expected)

    def test_a_candidate_without_a_position_is_skipped(self):
        surface = make_surface()
        model = FakeSuccessModel([0.8])
        row = make_row()
        row["intent_candidates"][1]["xy"] = None
        scored = candidate_scores(row, surface, model, PITCH)
        self.assertEqual(len(scored), 1)


class RankTests(unittest.TestCase):
    def test_the_truth_is_first_when_it_has_the_top_score(self):
        scored = [
            {"player": "a", "value": 0.1, "probability": 0.2, "combined": 0.3},
            {"player": "b", "value": 0.9, "probability": 0.4, "combined": 0.1},
        ]
        hit, tied = rank_hit(scored, "b", "value")
        self.assertTrue(hit)
        self.assertEqual(tied, 1)

    def test_ties_are_counted(self):
        scored = [
            {"player": "a", "value": 0.5, "probability": 0.2, "combined": 0.3},
            {"player": "b", "value": 0.5, "probability": 0.4, "combined": 0.1},
        ]
        hit, tied = rank_hit(scored, "a", "value")
        self.assertTrue(hit)
        self.assertEqual(tied, 2)

    def test_the_zone_hit_looks_at_the_best_valued_zone(self):
        scored = [
            {"player": "a", "zone": "attacking_third_centre", "value": 0.9, "probability": 0.2, "combined": 0.3},
            {"player": "b", "zone": "own_third_wide", "value": 0.1, "probability": 0.4, "combined": 0.1},
        ]
        self.assertTrue(zone_hit(scored, "attacking_third_centre"))
        self.assertFalse(zone_hit(scored, "own_third_wide"))


if __name__ == "__main__":
    unittest.main()
