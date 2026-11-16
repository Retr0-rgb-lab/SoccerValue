import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from soccervalue.events import load_timeline
from soccervalue.metrics import binary_scores
from soccervalue.state_frames import attach_state_frame, orient_state
from soccervalue.state_value import build_state_samples
from soccervalue.value_model import (
    StateValueNet,
    constant_rates,
    simple_features,
    spatial_features,
    two_head_loss,
)

ZONE = timezone(timedelta(hours=2))
BASE = datetime(2023, 5, 27, 15, 30, tzinfo=ZONE)


def write_timeline(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:30:00+02:00" EventId="kick" X-Position="52.5" Y-Position="34">
  <KickOff TeamLeft="home" GameSection="firstHalf"/>
</Event>
<Event EventTime="2023-05-27T15:30:10+02:00" EventId="play" X-Position="40" Y-Position="20">
  <Play Team="home" Player="p1" Evaluation="successfullyCompleted"><Pass/></Play>
</Event>
<Event EventTime="2023-05-27T15:30:20+02:00" EventId="shot" X-Position="90" Y-Position="34">
  <ShotAtGoal Team="home" Player="p1" xG="0.2"><SuccessfulShot/></ShotAtGoal>
</Event>
<Event EventTime="2023-05-27T15:30:30+02:00" EventId="tackle" X-Position="50" Y-Position="30">
  <TacklingGame WinnerTeam="away" PossessionChange="true"/>
</Event>
<Event EventTime="2023-05-27T16:15:00+02:00" EventId="whistle">
  <FinalWhistle GameSection="firstHalf"/>
</Event>
</PutDataRequest>
""",
        encoding="utf-8",
    )


def at(seconds: float) -> datetime:
    return BASE + timedelta(seconds=seconds)


def event(event_id: str, seconds: float, tag: str, **extra) -> dict:
    row = {
        "time": at(seconds),
        "event_id": event_id,
        "event_tag": tag,
        "game_section": "firstHalf",
        "team_left": "home",
        "team_id": "home",
        "player_id": "p",
        "x": 10.0,
        "y": 20.0,
        "is_shot": False,
        "is_goal": False,
        "xg": None,
        "possession_change": False,
        "evaluation": None,
        "match_id": "M1",
    }
    row.update(extra)
    return row


class TimelineTests(unittest.TestCase):
    def test_timeline_keeps_shot_goal_and_half_boundary(self):
        path = Path("data/cache/timeline_fixture.xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_timeline(path)
        events = load_timeline(path, "M1")
        by_id = {item["event_id"]: item for item in events}
        self.assertTrue(by_id["shot"]["is_shot"])
        self.assertTrue(by_id["shot"]["is_goal"])
        self.assertAlmostEqual(by_id["shot"]["xg"], 0.2)
        self.assertEqual(by_id["whistle"]["event_tag"], "FinalWhistle")
        self.assertEqual(by_id["shot"]["game_section"], "firstHalf")
        self.assertTrue(by_id["tackle"]["possession_change"])


class SampleTests(unittest.TestCase):
    def timeline(self) -> list[dict]:
        return [
            event("pass0", 0, "Play", evaluation="successfullyCompleted"),
            event("shot_for", 6, "ShotAtGoal", is_shot=True, team_id="home"),
            event("shot_against", 12, "ShotAtGoal", is_shot=True, team_id="away"),
            event("opp", 30, "Play", team_id="away", evaluation="successfullyCompleted"),
            event("late", 47, "Play", evaluation="successfullyCompleted"),
            event("whistle", 50, "FinalWhistle", team_id=None, x=None, y=None),
        ]

    def test_labels_use_the_fixed_team_and_drop_censored_windows(self):
        samples = build_state_samples(self.timeline(), window_s=10)
        by_id = {item["event_id"]: item for item in samples}
        self.assertEqual((by_id["pass0"]["y_for"], by_id["pass0"]["y_against"]), (1, 0))
        self.assertEqual((by_id["opp"]["y_for"], by_id["opp"]["y_against"]), (0, 0))
        self.assertNotIn("late", by_id)

    def test_failed_pass_keeps_the_original_team_after_a_turnover(self):
        events = [
            event("bad", 0, "Play", team_id="home", evaluation="unsuccessful"),
            event("their_shot", 8, "ShotAtGoal", is_shot=True, team_id="away"),
            event("whistle", 30, "FinalWhistle", team_id=None, x=None, y=None),
        ]
        row = build_state_samples(events, window_s=10)[0]
        self.assertEqual(row["possession_by_t"], 0)
        self.assertEqual(row["team_id"], "home")
        self.assertEqual((row["y_for"], row["y_against"]), (0, 1))


class FrameTests(unittest.TestCase):
    def test_state_frame_does_not_look_after_the_event(self):
        path = Path("data/cache/state_frame_fixture.xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<FrameSet TeamId="ball" PersonId="ball">
  <Frame T="2023-05-27T15:30:19.600+02:00" X="2" Y="0"/>
  <Frame T="2023-05-27T15:30:20.100+02:00" X="0.2" Y="0"/>
</FrameSet>
<FrameSet TeamId="home" PersonId="p1">
  <Frame T="2023-05-27T15:30:19.600+02:00" X="2" Y="0"/>
  <Frame T="2023-05-27T15:30:20.100+02:00" X="0.2" Y="0"/>
</FrameSet>
</PutDataRequest>
""",
            encoding="utf-8",
        )
        sample = {
            "time": at(20),
            "x": 52.5,
            "y": 34.0,
            "player_id": "p1",
            "team_id": "home",
        }
        attach_state_frame(path, [sample], (105.0, 68.0))
        self.assertLessEqual(sample["frame_signed_dt"], 0.0)
        self.assertAlmostEqual(sample["track_ball"][0], 2.0)

    def test_orient_faces_team_t_toward_positive_x(self):
        sample = {
            "match_id": "M1",
            "event_id": "e",
            "x": 20.0,
            "y": 10.0,
            "team_id": "away",
            "team_left": "home",
            "possession_by_t": 1,
            "y_for": 0,
            "y_against": 1,
            "track_ball": (20.0, 10.0),
            "track_players": {"a": (20.0, 10.0, "away"), "h": (40.0, 10.0, "home")},
            "shots_in_window": [],
        }
        row = orient_state(sample, "identity", 105.0, 68.0)
        self.assertAlmostEqual(row["ball"][0], 85.0)
        self.assertEqual(len(row["team"]), 1)
        self.assertEqual(len(row["opponent"]), 1)


class ValueModelTests(unittest.TestCase):
    def test_simple_features_measure_both_goals(self):
        row = {
            "ball": (10.0, 34.0),
            "pitch": (105.0, 68.0),
            "possession_by_t": 1,
            "y_for": 0,
            "y_against": 0,
        }
        features, _for, _against = simple_features([row])
        self.assertAlmostEqual(features[0, 2], 10 / 105)
        self.assertAlmostEqual(features[0, 3], 95 / 105)

    def test_spatial_features_count_pressure_and_players_ahead(self):
        row = {
            "ball": (10.0, 34.0),
            "pitch": (105.0, 68.0),
            "possession_by_t": 1,
            "y_for": 0,
            "y_against": 0,
            "team": [(5.0, 34.0), (20.0, 34.0)],
            "opponent": [(12.0, 34.0), (10.0, 40.0)],
        }
        features, _for, _against = spatial_features([row])
        diagonal = (105.0**2 + 68.0**2) ** 0.5
        self.assertAlmostEqual(features[0, 5], 2.0 / diagonal)
        self.assertAlmostEqual(features[0, 6], 95 / 105)
        self.assertEqual(features[0, 7], 1.0)
        self.assertEqual(features[0, 8], 1.0)

    def test_constant_rate_is_the_training_frequency(self):
        rows = [{"y_for": value, "y_against": 0} for value in (1, 0, 0, 0)]
        self.assertAlmostEqual(constant_rates(rows)[0], 0.25)

    def test_network_outputs_two_probabilities(self):
        model = StateValueNet()
        frames = torch.zeros(2, 3, 68, 104)
        context = torch.zeros(2, 3)
        out = model(frames, context)
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out <= 1))

    def test_loss_reads_both_heads(self):
        labels_for = torch.tensor([1.0])
        labels_against = torch.tensor([1.0])
        low = two_head_loss(torch.tensor([[0.2, 0.2]]), labels_for, labels_against)
        high = two_head_loss(torch.tensor([[0.2, 0.9]]), labels_for, labels_against)
        self.assertNotAlmostEqual(float(low), float(high))

    def test_constant_scores_accept_state_labels(self):
        samples = build_state_samples(
            [
                event("pass0", 0, "Play", evaluation="successfullyCompleted"),
                event("shot_for", 6, "ShotAtGoal", is_shot=True),
                event("whistle", 40, "FinalWhistle", team_id=None, x=None, y=None),
            ],
            window_s=10,
        )
        labels = np.asarray([item["y_for"] for item in samples])
        rate = float(labels.mean())
        scores = binary_scores(labels, np.full(labels.shape, rate))
        self.assertIn("brier", scores)


if __name__ == "__main__":
    unittest.main()
