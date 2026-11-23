import unittest

import numpy as np

from soccervalue.run_two_stage import METHODS, pick

PITCH = (105.0, 68.0)


def candidates():
    return [
        {"player": "a", "zone_value": 0.03, "ranker_score": 1.0, "perp": 0.5},
        {"player": "b", "zone_value": 0.01, "ranker_score": 2.0, "perp": 1.0},
    ]


class PickTests(unittest.TestCase):
    def test_the_zone_value_alone_picks_the_best_zone(self):
        self.assertEqual(pick(candidates(), "zone_only"), "a")

    def test_the_zone_value_alone_breaks_ties_towards_the_ray(self):
        tied = [
            {"player": "a", "zone_value": 0.03, "ranker_score": 1.0, "perp": 2.0},
            {"player": "b", "zone_value": 0.03, "ranker_score": 2.0, "perp": 1.0},
        ]
        self.assertEqual(pick(tied, "zone_only"), "b")

    def test_the_ranker_alone_picks_the_highest_score(self):
        self.assertEqual(pick(candidates(), "ranker_only"), "b")

    def test_the_lexicographic_reading_lets_the_zone_decide(self):
        self.assertEqual(pick(candidates(), "lexicographic"), "a")

    def test_the_weighted_reading_follows_the_ranker_at_a_small_weight(self):
        self.assertEqual(pick(candidates(), "weighted", 1.0), "b")
        self.assertEqual(pick(candidates(), "weighted", 100.0), "a")

    def test_the_ray_rule_picks_the_nearest_to_the_ray(self):
        self.assertEqual(pick(candidates(), "ray_rule"), "a")

    def test_every_method_is_defined(self):
        self.assertEqual(set(METHODS), {"zone_only", "ranker_only", "lexicographic", "weighted", "ray_rule"})


if __name__ == "__main__":
    unittest.main()
