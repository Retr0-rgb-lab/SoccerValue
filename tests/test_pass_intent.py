import unittest
from pathlib import Path

import numpy as np

from soccervalue.events import load_open_flat_passes
from soccervalue.intent import FEATURE_NAMES, IntentTracker, along_distance, perp_distance
from soccervalue.intent_model import build_dataset, fit_intent_model, label_rows, predict_pick, rule_picks, score_top1

PITCH = (105.0, 68.0)


def raw(event_x: float, event_y: float) -> tuple[float, float]:
    """Event coordinates into tracking coordinates, which are pitch centred."""
    return (event_x - PITCH[0] / 2.0, event_y - PITCH[1] / 2.0)


def ball_flight() -> list[tuple[float, float, float]]:
    """A slow dribble, then a kick towards +x at 15 m/s.

    Frame 5 is the kick. The ball sits at (30, 34) there, so the passer and the
    ball share a point.
    """
    rows = []
    for index in range(24):
        stamp = index * 0.04
        if index <= 5:
            event_x = 30.0 - 0.04 * (5 - index)
        else:
            event_x = 30.0 + 0.6 * (index - 5)
        x, y = raw(event_x, 34.0)
        rows.append((stamp, x, y))
    return rows


def tracker() -> IntentTracker:
    return IntentTracker(ball_flight(), PITCH[0], PITCH[1])


def snapshot(**players):
    return {name: (raw(x, y)[0], raw(x, y)[1], team) for name, (x, y, team) in players.items()}


def item(**extra):
    base = {"player_id": "p1", "team_id": "home", "success": 1}
    base.update(extra)
    return base


class IntentRuleTests(unittest.TestCase):
    def test_the_teammate_on_the_ray_is_the_target(self):
        people = snapshot(
            p1=(30.0, 34.0, "home"),
            t1=(50.0, 35.0, "home"),
            t2=(45.0, 44.0, "home"),
            d1=(40.0, 34.0, "away"),
        )
        result = tracker().estimate(item(), 5, people)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["player"], "t1")
        self.assertAlmostEqual(result["perp_m"], 1.0, places=6)
        self.assertAlmostEqual(result["along_m"], 20.0, places=6)
        self.assertAlmostEqual(result["target_x"], 50.0, places=6)
        self.assertAlmostEqual(result["target_y"], 35.0, places=6)

    def test_an_opponent_closer_to_the_ray_is_not_a_candidate(self):
        people = snapshot(
            p1=(30.0, 34.0, "home"),
            d1=(34.0, 34.0, "away"),
            t1=(50.0, 36.0, "home"),
        )
        result = tracker().estimate(item(), 5, people)
        self.assertEqual(result["player"], "t1")
        self.assertGreater(result["perp_m"], 0.5)

    def test_a_teammate_behind_the_ball_is_not_a_candidate(self):
        people = snapshot(
            p1=(30.0, 34.0, "home"),
            behind=(20.0, 34.0, "home"),
            t1=(50.0, 36.0, "home"),
        )
        result = tracker().estimate(item(), 5, people)
        self.assertEqual(result["player"], "t1")

    def test_a_teammate_far_from_the_ray_is_not_a_candidate(self):
        people = snapshot(
            p1=(30.0, 34.0, "home"),
            wide=(45.0, 60.0, "home"),
            t1=(50.0, 36.0, "home"),
        )
        result = tracker().estimate(item(), 5, people)
        self.assertEqual(result["player"], "t1")

    def test_no_eligible_teammate_reports_no_candidate(self):
        people = snapshot(p1=(30.0, 34.0, "home"), behind=(20.0, 34.0, "home"))
        result = tracker().estimate(item(), 5, people)
        self.assertEqual(result["status"], "no_candidate")
        self.assertIsNone(result["player"])

    def test_the_estimate_does_not_read_the_label(self):
        people = snapshot(p1=(30.0, 34.0, "home"), t1=(50.0, 35.0, "home"))
        success = tracker().estimate(item(success=1), 5, people)
        failure = tracker().estimate(item(success=0), 5, people)
        self.assertEqual(success, failure)

    def test_a_ball_that_never_is_kicked_has_no_direction(self):
        slow = [(index * 0.04, *raw(30.0 + 0.04 * index, 34.0)) for index in range(24)]
        result = IntentTracker(slow, PITCH[0], PITCH[1]).estimate(
            item(), 5, snapshot(p1=(30.0, 34.0, "home"), t1=(50.0, 35.0, "home"))
        )
        self.assertEqual(result["status"], "no_kick")

    def test_the_helpers_measure_the_ray(self):
        direction = (1.0, 0.0)
        self.assertAlmostEqual(perp_distance((10.0, 3.0), (0.0, 0.0), direction), 3.0)
        self.assertAlmostEqual(along_distance((10.0, 3.0), (0.0, 0.0), direction), 10.0)


class CandidateTableTests(unittest.TestCase):
    def people(self, **extra):
        base = {
            "p1": (30.0, 34.0, "home"),
            "t1": (50.0, 35.0, "home"),
            "t2": (45.0, 44.0, "home"),
            "d1": (40.0, 34.0, "away"),
        }
        base.update(extra)
        return snapshot(**base)

    def test_the_table_holds_the_teammates_and_not_the_opponent(self):
        result = tracker().estimate(item(), 5, self.people())
        players = [row["player"] for row in result["candidates"]]
        self.assertIn("t1", players)
        self.assertIn("t2", players)
        self.assertNotIn("d1", players)
        self.assertNotIn("p1", players)

    def test_a_teammate_far_off_the_ray_is_not_a_candidate(self):
        result = tracker().estimate(item(), 5, self.people(wide=(45.0, 60.0, "home")))
        self.assertNotIn("wide", [row["player"] for row in result["candidates"]])

    def test_every_candidate_carries_the_feature_table(self):
        result = tracker().estimate(item(), 5, self.people())
        for row in result["candidates"]:
            self.assertEqual(len(row["features"]), len(FEATURE_NAMES))

    def test_the_features_do_not_depend_on_which_way_the_team_attacks(self):
        """The aim direction turns around with the frame, so nothing in the
        table may reveal it."""
        people = self.people()
        forward = tracker().estimate(item(), 5, people)
        turned_people = {name: (-x, -y, team) for name, (x, y, team) in people.items()}
        turned = IntentTracker(
            [(stamp, -x, -y) for stamp, x, y in ball_flight()], PITCH[0], PITCH[1]
        ).estimate(item(), 5, turned_people)
        self.assertEqual(forward["player"], turned["player"])
        for left, right in zip(forward["candidates"], turned["candidates"]):
            for one, other in zip(left["features"], right["features"]):
                self.assertAlmostEqual(one, other, places=6)


class IntentModelTests(unittest.TestCase):
    def make_row(self, recipient="t1"):
        candidates = [
            {"player": "t1", "features": [0.1] * len(FEATURE_NAMES)},
            {"player": "t2", "features": [0.4] * len(FEATURE_NAMES)},
        ]
        return {
            "match_id": "M1",
            "event_id": "e1",
            "success": 1,
            "recipient": recipient,
            "intent_player": "t2",
            "intent_candidates": candidates,
        }

    def test_a_row_without_the_recipient_is_skipped(self):
        dataset, skipped = build_dataset([self.make_row(recipient="nobody")])
        self.assertEqual(len(dataset), 0)
        self.assertEqual(skipped, 1)

    def test_the_labelled_candidate_is_found(self):
        features, players, index = label_rows(self.make_row())
        self.assertEqual(players[index], "t1")
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))

    def test_a_trained_model_picks_the_labelled_candidate(self):
        dataset, _ = build_dataset([self.make_row() for _ in range(12)])
        net, mean, std = fit_intent_model(dataset, model="linear", steps=300)
        picks = {item["event_id"]: predict_pick(net, item, mean, std) for item in dataset}
        scored = score_top1(dataset, picks)
        self.assertEqual(scored["top1"], 1.0)

    def test_the_rule_pick_is_read_from_the_row(self):
        dataset, _ = build_dataset([self.make_row()])
        self.assertEqual(rule_picks(dataset)["e1"], "t2")


class FailedPassReportTests(unittest.TestCase):
    def test_the_report_counts_agreement_and_the_distance_to_the_ray(self):
        from soccervalue.run_pass_intent_model import pick_report

        def make(event_id, recipient):
            return {
                "match_id": "M1",
                "event_id": event_id,
                "success": 0,
                "recipient": recipient,
                "intent_player": "t1",
                "intent_candidates": [
                    {"player": "t1", "features": [0.1] * len(FEATURE_NAMES)},
                    {"player": "t2", "features": [0.3] * len(FEATURE_NAMES)},
                ],
            }

        failures, _ = build_dataset([make("f1", None), make("f2", None)], require_label=False)
        training, _ = build_dataset(
            [
                {
                    "match_id": "M1",
                    "event_id": f"t{index}",
                    "success": 1,
                    "recipient": "t1",
                    "intent_player": "t1",
                    "intent_candidates": [
                        {"player": "t1", "features": [0.1] * len(FEATURE_NAMES)},
                        {"player": "t2", "features": [0.3] * len(FEATURE_NAMES)},
                    ],
                }
                for index in range(12)
            ]
        )
        net, mean, std = fit_intent_model(training, model="linear", steps=50)
        report = pick_report(net, mean, std, failures)
        self.assertEqual(report["n"], 2)
        self.assertIsNotNone(report["same_pick_as_the_rule"])
        self.assertIn("model_pick", report)
        self.assertIn("where_they_differ", report)


class NextEventAttributionTests(unittest.TestCase):
    def test_a_tackle_by_the_opponent_is_named_as_the_opponent(self):
        path = Path("data/cache/intent_relation_fixture.xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:30:12.230+02:00" EventId="kick" X-Source-Position="52.5" Y-Source-Position="34" X-Position="52.5" Y-Position="34">
  <KickOff TeamLeft="home" GameSection="firstHalf"/>
</Event>
<Event EventTime="2023-05-27T15:31:00.000+02:00" EventId="pass" X-Source-Position="20" Y-Source-Position="20" X-Position="20" Y-Position="20">
  <Play FromOpenPlay="true" Height="flat" Team="home" Player="p1" Recipient="p2" Evaluation="unsuccessful">
    <Pass/>
  </Play>
</Event>
<Event EventTime="2023-05-27T15:31:02.000+02:00" EventId="tackle" X-Position="35" Y-Position="25">
  <TacklingGame WinnerTeam="away" Winner="d1" PossessionChange="true"/>
</Event>
</PutDataRequest>
""",
            encoding="utf-8",
        )
        passes = load_open_flat_passes(path, "M1")
        by_id = {row["event_id"]: row for row in passes}
        self.assertEqual(by_id["pass"]["end_source"], "next_event")
        self.assertEqual((by_id["pass"]["end_x"], by_id["pass"]["end_y"]), (35.0, 25.0))
        self.assertEqual(by_id["pass"]["next_event_relation"], "opponent")
        self.assertEqual(by_id["pass"]["next_event_tag"], "TacklingGame")
        self.assertEqual(by_id["pass"]["recipient"], "p2")


if __name__ == "__main__":
    unittest.main()
