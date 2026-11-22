import unittest

import numpy as np

from soccervalue.state_value import window_xg
from soccervalue.surface import DEPTH_NAMES, LATERAL_NAMES, ZONE_COUNT, ZoneSurface, zone_name, zone_of

PITCH = (105.0, 68.0)


def state(x, y, possession, value):
    return {
        "ball": (x, y),
        "possession_by_t": possession,
        "xg_for_sum": value,
        "xg_against_sum": 0.0,
    }


class ZoneTests(unittest.TestCase):
    def test_eight_zones_cover_the_pitch(self):
        self.assertEqual(ZONE_COUNT, 8)
        self.assertEqual(len(DEPTH_NAMES), 4)
        self.assertEqual(len(LATERAL_NAMES), 2)

    def test_the_centre_band_holds_the_two_half_spaces(self):
        centre = zone_of(90.0, 34.0, PITCH)
        wide = zone_of(90.0, 5.0, PITCH)
        self.assertNotEqual(centre, wide)
        self.assertTrue(zone_name(centre).endswith("centre"))
        self.assertTrue(zone_name(wide).endswith("wide"))

    def test_the_depth_bands_follow_the_thirds_and_the_box(self):
        self.assertTrue(zone_name(zone_of(10.0, 34.0, PITCH)).startswith("own_third"))
        self.assertTrue(zone_name(zone_of(50.0, 34.0, PITCH)).startswith("middle_third"))
        self.assertTrue(zone_name(zone_of(80.0, 34.0, PITCH)).startswith("attacking_third"))
        self.assertTrue(zone_name(zone_of(95.0, 34.0, PITCH)).startswith("opponent_box"))

    def test_a_zone_is_identified_by_depth_and_side(self):
        self.assertEqual(zone_of(95.0, 34.0, PITCH), 7)
        self.assertEqual(zone_of(95.0, 5.0, PITCH), 6)
        self.assertEqual(zone_of(10.0, 34.0, PITCH), 1)
        self.assertEqual(zone_of(10.0, 5.0, PITCH), 0)


class SurfaceTests(unittest.TestCase):
    def test_a_zone_takes_the_average_of_its_states(self):
        rows = [state(95.0, 34.0, 1, 0.2), state(95.0, 34.0, 1, 0.4), state(95.0, 34.0, 1, 0.3)]
        surface = ZoneSurface().fit(rows, PITCH)
        self.assertAlmostEqual(surface.value(95.0, 34.0, 1, PITCH), 0.3)
        self.assertEqual(surface.counts[(7, 1)], 3)

    def test_an_empty_cell_falls_back_to_the_other_holder(self):
        rows = [state(95.0, 34.0, 1, 0.5)]
        surface = ZoneSurface().fit(rows, PITCH)
        self.assertAlmostEqual(surface.value(95.0, 34.0, 0, PITCH), 0.5)

    def test_the_change_is_the_difference_between_two_positions(self):
        rows = [state(95.0, 34.0, 1, 0.5), state(10.0, 34.0, 1, 0.1)]
        surface = ZoneSurface().fit(rows, PITCH)
        change = surface.delta((10.0, 34.0), (95.0, 34.0), 1, PITCH)
        self.assertAlmostEqual(change, 0.4)

    def test_the_shape_reports_the_centre_against_the_wide_band(self):
        rows = [state(80.0, 34.0, 1, 0.2), state(80.0, 5.0, 1, 0.05)]
        surface = ZoneSurface().fit(rows, PITCH)
        shape = surface.shape(PITCH)
        self.assertGreater(shape["by_depth"]["attacking_third"]["centre_minus_wide"], 0.0)


class WindowXgTests(unittest.TestCase):
    def test_an_empty_window_is_zero(self):
        self.assertEqual(window_xg([]), (0.0, 0.0))

    def test_the_sum_and_the_product_form(self):
        total, product = window_xg([0.1, 0.2])
        self.assertAlmostEqual(total, 0.3)
        self.assertAlmostEqual(product, 1.0 - 0.9 * 0.8)

    def test_a_single_shot_gives_the_same_number_in_both_forms(self):
        total, product = window_xg([0.25])
        self.assertAlmostEqual(total, 0.25)
        self.assertAlmostEqual(product, 0.25)


if __name__ == "__main__":
    unittest.main()
