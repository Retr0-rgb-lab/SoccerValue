import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from soccervalue.align import alignment_report, choose_transform, convert, frame_cost, orient_samples
from soccervalue.events import label_audit, load_open_flat_passes
from soccervalue.features import geometry_matrix, spatial_matrix
from soccervalue.matches import TEST_MATCHES, TRAIN_MATCHES, VAL_MATCHES, assert_split_disjoint
from soccervalue.metrics import binary_scores
from soccervalue.model import SoccerMapSuccess, destination_loss
from soccervalue.raster import cell_index, raster

ZONE = timezone(timedelta(hours=2))


def write_events(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:30:12.230+02:00" EventId="kick" X-Source-Position="52.5" Y-Source-Position="34" X-Position="52.5" Y-Position="34">
  <KickOff TeamLeft="home" GameSection="firstHalf"/>
</Event>
<Event EventTime="2023-05-27T15:30:20.000+02:00" EventId="flat" X-Source-Position="40" Y-Source-Position="20" X-Position="55" Y-Position="22">
  <Play FromOpenPlay="true" Height="flat" Team="home" Player="p1" Evaluation="successfullyCompleted">
    <Pass/>
  </Play>
</Event>
<Event EventTime="2023-05-27T15:30:30.000+02:00" EventId="high" X-Source-Position="10" Y-Source-Position="10" X-Position="30" Y-Position="12">
  <Play FromOpenPlay="true" Height="high" Team="home" Player="p1" Evaluation="unsuccessful">
    <Pass/>
  </Play>
</Event>
<Event EventTime="2023-05-27T15:30:40.000+02:00" EventId="throw" X-Source-Position="5" Y-Source-Position="5" X-Position="15" Y-Position="8">
  <ThrowIn>
    <Play FromOpenPlay="false" Height="flat" Team="away" Player="p2" Evaluation="successfullyCompleted">
      <Pass/>
    </Play>
  </ThrowIn>
</Event>
<Event EventTime="2023-05-27T15:31:00.000+02:00" EventId="same" X-Source-Position="20" Y-Source-Position="20" X-Position="20" Y-Position="20">
  <Play FromOpenPlay="true" Height="flat" Team="away" Player="p2" Evaluation="unsuccessful">
    <Pass/>
  </Play>
</Event>
<Event EventTime="2023-05-27T15:31:02.000+02:00" EventId="arrival" X-Position="28" Y-Position="24"/>
</PutDataRequest>
""",
        encoding="utf-8",
    )


class PassAndSplitTests(unittest.TestCase):
    def test_split_is_fixed_and_disjoint(self):
        assert_split_disjoint()
        self.assertEqual(len(TRAIN_MATCHES), 5)
        self.assertEqual(len(VAL_MATCHES), 1)
        self.assertEqual(len(TEST_MATCHES), 1)

    def test_parser_keeps_open_flat_passes_and_finds_a_missing_end(self):
        path = Path("data/cache/parser_fixture.xml")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_events(path)
        passes = load_open_flat_passes(path, "M1")
        by_id = {item["event_id"]: item for item in passes}
        self.assertEqual(set(by_id), {"flat", "same"})
        self.assertEqual(by_id["flat"]["success"], 1)
        self.assertEqual((by_id["flat"]["end_x"], by_id["flat"]["end_y"]), (55.0, 22.0))
        self.assertEqual(by_id["same"]["success"], 0)
        self.assertEqual((by_id["same"]["end_x"], by_id["same"]["end_y"]), (28.0, 24.0))
        self.assertEqual(by_id["same"]["end_source"], "next_event")
        self.assertEqual(by_id["flat"]["end_source"], "event")
        self.assertEqual(by_id["same"]["team_left"], "home")
        audit = label_audit(passes)
        self.assertEqual(audit["n_failure"], 1)
        self.assertEqual(audit["fraction_failures_end_from_next_event"], 1.0)

    def test_sync_prefers_the_frame_where_the_ball_matches_the_event(self):
        near_in_time = frame_cost(ball_event_m=18.0, passer_ball_m=8.0, dt_s=0.1)
        matching_kick = frame_cost(ball_event_m=1.5, passer_ball_m=0.8, dt_s=1.2)
        self.assertLess(matching_kick, near_in_time)

    def test_transform_choice_uses_the_shift_that_matches_the_event(self):
        samples = [
            {"start_x": 52.5, "start_y": 34.0, "track_ball": (52.5, 0.0, 0.2)},
            {"start_x": 20.0, "start_y": 10.0, "track_ball": (20.0, -24.0, 0.2)},
        ]
        self.assertEqual(choose_transform(samples, 105.0, 68.0), "y_center")
        self.assertEqual(convert("y_center", 20.0, -24.0, 105.0, 68.0), (20.0, 10.0))

    def test_attacking_team_faces_positive_x(self):
        sample = {
            "match_id": "M1",
            "success": 1,
            "team_id": "away",
            "team_left": "home",
            "start_x": 20.0,
            "start_y": 10.0,
            "end_x": 30.0,
            "end_y": 12.0,
            "track_ball": (20.0, 10.0, 0.1),
            "track_passer": (20.0, 10.0),
            "track_players": {"p": (20.0, 10.0, "away"), "d": (40.0, 10.0, "home")},
        }
        row = orient_samples([sample], "identity", 105.0, 68.0)[0]
        self.assertAlmostEqual(row["start"][0], 85.0)
        self.assertAlmostEqual(row["ball"][0], 85.0)
        self.assertEqual(len(row["attack"]), 1)
        self.assertEqual(len(row["defense"]), 1)

    def test_alignment_gate_rejects_a_large_gap(self):
        samples = [
            {
                "start_x": 0.0,
                "start_y": 0.0,
                "track_ball": (10.0, 0.0, 0.0),
                "track_passer": (10.0, 0.0),
            }
        ]
        report = alignment_report(samples, "identity", 105.0, 68.0)
        self.assertFalse(report["usable"])
        self.assertAlmostEqual(report["ball_vs_event_median_m"], 10.0)

    def test_raster_peaks_on_the_ball_cell(self):
        image = raster([], [], (52.5, 34.0), (105.0, 68.0))
        ix, iy = cell_index(52.5, 34.0, 105.0, 68.0)
        peak_y, peak_x = np.unravel_index(int(image[2].argmax()), image[2].shape)
        self.assertLessEqual(abs(int(peak_x) - ix), 1)
        self.assertLessEqual(abs(int(peak_y) - iy), 1)
        self.assertGreater(image[2, iy, ix], 0.5)

    def test_loss_reads_only_the_destination_cell(self):
        row = {
            "success": 1,
            "end": (0.0, 0.0),
            "pitch": (105.0, 68.0),
        }
        predicted = torch.full((1, 1, 68, 104), 0.2)
        ix, iy = cell_index(0.0, 0.0, 105.0, 68.0)
        predicted[0, 0, iy, ix] = 0.9
        loss_close = float(destination_loss(predicted, [row]))
        predicted[0, 0, iy, min(ix + 5, 103)] = 0.01
        loss_elsewhere = float(destination_loss(predicted, [row]))
        self.assertAlmostEqual(loss_close, loss_elsewhere)
        predicted[0, 0, iy, ix] = 0.1
        self.assertGreater(float(destination_loss(predicted, [row])), loss_close)

    def test_spatial_features_count_a_defender_on_the_lane(self):
        clear = {
            "success": 1,
            "ball": (20.0, 34.0),
            "end": (60.0, 34.0),
            "attack": [(20.0, 34.0), (70.0, 34.0)],
            "defense": [(80.0, 10.0)],
            "pitch": (105.0, 68.0),
        }
        blocked = dict(clear)
        blocked["defense"] = [(40.0, 34.0)]
        clear_x, _ = spatial_matrix([clear])
        blocked_x, _ = spatial_matrix([blocked])
        lane_count = -2
        clearance = -1
        self.assertEqual(clear_x[0, lane_count], 0.0)
        self.assertEqual(blocked_x[0, lane_count], 1.0)
        self.assertLess(blocked_x[0, clearance], clear_x[0, clearance])
        geometry, _ = geometry_matrix([clear])
        self.assertEqual(geometry.shape[1], 9)
        self.assertEqual(clear_x.shape[1], 14)

    def test_brier_of_a_perfect_prediction_is_zero(self):
        scores = binary_scores(np.array([0, 1]), np.array([0.0, 1.0]))
        self.assertEqual(scores["brier"], 0.0)
        self.assertEqual(scores["auroc"], 1.0)

    def test_network_output_shape(self):
        model = SoccerMapSuccess()
        out = model(torch.zeros(2, 3, 68, 104))
        self.assertEqual(tuple(out.shape), (2, 1, 68, 104))


if __name__ == "__main__":
    unittest.main()
