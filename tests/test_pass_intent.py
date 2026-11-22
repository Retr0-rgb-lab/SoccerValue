import unittest
from pathlib import Path

from soccervalue.events import load_open_flat_passes
from soccervalue.intent import IntentTracker, along_distance, perp_distance

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
