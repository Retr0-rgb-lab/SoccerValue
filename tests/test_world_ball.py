import unittest

import numpy as np

from soccervalue.ball_flight import fit_drag, mean_error
import torch
from torch import nn

from soccervalue.player_motion import constant_player, nearest_to_path, rollout
from soccervalue.touch_head import baseline_index, player_features
from soccervalue.world_segments import DT, constant_velocity, roll_position, touch_indices


class WorldSegmentTests(unittest.TestCase):
    def test_touch_requires_a_nearby_player_and_a_speed_jump(self):
        steps = 30
        xy = np.zeros((steps, 2))
        xy[:, 0] = np.minimum(np.arange(steps) * DT * 10.0, 10.0)
        nearest = np.full(steps, 8.0)
        nearest[10] = 1.0
        hits = touch_indices(xy, nearest)
        self.assertIn(10, hits.tolist())
        nearest[10] = 8.0
        self.assertEqual(len(touch_indices(xy, nearest)), 0)

    def test_drag_stays_behind_constant_velocity_on_a_slowing_ball(self):
        times = np.arange(11) * DT
        origin = np.zeros(2)
        velocity = np.array([10.0, 0.0])
        future = roll_position(origin, velocity, times, 0.8)
        segment = {"origin": origin, "velocity": velocity, "future": future, "dt": DT}
        self.assertLess(mean_error([segment], 0.8, 1.0), mean_error([segment], 0.0, 1.0))
        self.assertAlmostEqual(fit_drag([segment]), 0.8, delta=0.15)

    def test_baseline_picks_the_player_who_meets_the_ball_first(self):
        segment = {
            "origin": np.zeros(2),
            "velocity": np.array([5.0, 0.0]),
            "dt": DT,
            "future": np.stack([np.array([index * DT * 5.0, 0.0]) for index in range(11)]),
            "roster": [
                {"id": "near", "team": "A", "xy": np.array([0.0, 3.0]), "velocity": np.zeros(2)},
                {"id": "meet", "team": "B", "xy": np.array([4.0, 0.0]), "velocity": np.zeros(2)},
            ],
            "toucher": {"id": "meet", "team": "B"},
        }
        features, roster = player_features(segment)
        self.assertEqual(roster[baseline_index(features)]["id"], "meet")

    def test_constant_velocity_is_drag_zero(self):
        origin = np.array([1.0, 2.0])
        velocity = np.array([3.0, -1.0])
        times = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(constant_velocity(origin, velocity, times), roll_position(origin, velocity, times, 0.0))

    def test_nearest_players_are_chosen_from_the_kick_not_the_future(self):
        players = np.array([[0.0, 8.0], [2.0, 0.2], [30.0, 20.0]])
        ball = np.stack([np.array([index * 1.0, 0.0]) for index in range(6)])
        chosen = nearest_to_path(players, ball, 1)
        self.assertEqual(int(chosen[0]), 1)

    def test_velocity_copy_rollout_matches_constant_velocity(self):
        class CopyVelocity(nn.Module):
            def forward(self, tokens, pad_mask):
                return tokens[..., 2:4]

        player = {"xy": np.array([1.0, 2.0]), "velocity": np.array([4.0, -2.0]), "team": "A", "id": "p"}
        steps = 5
        segment = {
            "origin": np.zeros(2),
            "dt": DT,
            "roster": [player],
        }
        ball = np.zeros((steps, 2))
        predicted = rollout(CopyVelocity(), segment, ball, torch.device("cpu"))
        np.testing.assert_allclose(predicted[:, 0, :], constant_player(player, steps), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
