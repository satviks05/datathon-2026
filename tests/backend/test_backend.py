"""Meaningful backend checks against synthetic edge cases and real input records."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src import config
from src.api import get_analysis, get_area_evidence, get_data_quality, get_historical_evaluation
from src.engine import AnalysisEngine
from src.loaders import input_version, load_events
from src.ranking import score_candidates
from src.regression import fit_ols, regression_metrics
from src.spatial import RadiusIndex, haversine_m


def north(meters):
    return 40.7 + math.degrees(meters / config.EARTH_RADIUS_M)


def report(identifier, meters, created, descriptor="Controller"):
    return dict(unique_key=identifier, created_date=created, closed_date=None, agency="DOT",
                complaint_type="Traffic Signal Condition", descriptor=descriptor, status="Closed",
                borough=None, latitude=north(meters), longitude=-74.0)


def crash(identifier, meters, date, time="00:00", injured=0, killed=0):
    return dict(collision_id=identifier, crash_date=date, crash_time=time, borough=None,
                latitude=north(meters), longitude=-74.0, number_of_persons_injured=injured,
                number_of_persons_killed=killed, number_of_cyclist_injured=0,
                has_injury=injured > 0, has_fatality=killed > 0)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.reports = pd.DataFrame([
            report("0001", 0, "2025-08-01T23:59:59"),
            report("0002", 300, "2025-08-01T00:00:00", "Pedestrian Signal"),
            report("0003", 0, "2025-08-31T23:59:59"),
            report("0004", 900, "2025-09-01T00:00:00"),
        ])
        self.crashes = pd.DataFrame([
            crash("0001", 150, "2025-08-01", "23:59", injured=1),
            crash("0002", 150, "2025-08-01", "00:01", killed=1),
            crash("0003", 0, "2025-08-02"),
            crash("0004", 150, "2025-09-01", injured=9),
        ])
        self.write()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self):
        self.reports.to_csv(self.directory / "signal_complaints_clean.csv", index=False)
        self.crashes.to_csv(self.directory / "collisions_clean.csv", index=False)

    def analysis(self, **kwargs):
        controls = dict(start_date="2025-08-01", end_date="2025-08-31", strategy="crash_first")
        controls.update(kwargs)
        return get_analysis(data_dir=self.directory, **controls)

    def test_end_date_includes_entire_day_and_actual_crash_time_orders_events(self):
        result = self.analysis(start_date="2025-08-01", end_date="2025-08-01")
        self.assertEqual(set(result["reports"]["unique_key"]), {"0001", "0002"})
        self.assertEqual(result["collisions"]["collision_id"].tolist(), ["0002", "0001"])
        self.assertEqual(result["metadata"]["harm_crashes_total"], 2)
        self.assertFalse(result["metadata"]["model_available"])

    def test_fatal_only_crash_is_included_and_strings_preserve_leading_zeros(self):
        result = self.analysis()
        self.assertEqual(result["targets_scored"]["harm_crash_count"].tolist(), [2, 2])
        evidence = get_area_evidence(result["selected_targets"].iloc[0]["target_id"], 250,
                                     data_dir=self.directory)
        self.assertEqual(set(evidence["collisions"]["collision_id"]), {"0001", "0002", "0003"})
        fatal_only = evidence["collisions"].loc[evidence["collisions"]["collision_id"].eq("0002")].iloc[0]
        self.assertTrue(fatal_only["has_harm"])
        self.assertFalse(fatal_only["has_injury"])

    def test_union_coverage_counts_each_event_once_across_overlapping_circles(self):
        result = self.analysis()
        self.assertEqual(len(result["selected_targets"]), 2)
        self.assertEqual(int(result["selected_targets"]["harm_crash_count"].sum()), 4)
        engine = AnalysisEngine(self.directory, input_version(self.directory))
        reports, harm = engine.coverage(result["selected_targets"], 250, "2025-08-01", "2025-08-31")
        self.assertEqual((reports, harm), (3, 2))
        for row in result["comparison"].itertuples():
            self.assertEqual(row.future_harm_crashes_covered, 1)
            self.assertEqual(row.history_signal_reports_covered, 3)
            self.assertEqual(row.future_harm_coverage_pct, 100)

    def test_september_changes_cannot_change_august_candidates_counts_or_scores(self):
        before = self.analysis()
        self.reports.loc[3, "latitude"] = north(0)
        self.crashes.loc[3, "latitude"] = north(1200)
        self.crashes.loc[3, "number_of_persons_injured"] = 500
        self.write()
        after = self.analysis()
        assert_frame_equal(before["targets_scored"], after["targets_scored"])
        self.assertEqual(before["model"]["coefficients"], after["model"]["coefficients"])
        self.assertNotEqual(before["metadata"]["input_version"], after["metadata"]["input_version"])
        self.assertEqual(len(before["targets_scored"]), 2)

    def test_date_changes_recalculate_counts_evidence_and_comparison_together(self):
        day = self.analysis(start_date="2025-08-31", end_date="2025-08-31")
        self.assertEqual(day["metadata"]["signal_reports_total"], 1)
        self.assertTrue(day["targets_scored"]["harm_crash_count"].eq(0).all())
        self.assertTrue(day["comparison"]["future_harm_crashes_covered"].eq(1).all())
        evidence = get_area_evidence(day["selected_targets"].iloc[0]["target_id"], 250,
                                     start_date="2025-08-31", end_date="2025-08-31", data_dir=self.directory)
        self.assertEqual(evidence["reports"]["unique_key"].tolist(), ["0003"])
        self.assertTrue(evidence["collisions"].empty)

    def test_empty_window_does_not_manufacture_recommendations(self):
        result = self.analysis(start_date="2025-08-03", end_date="2025-08-03")
        self.assertTrue(result["selected_targets"].empty)
        self.assertTrue(result["comparison"]["selected_count"].eq(0).all())
        self.assertTrue(result["comparison"]["future_harm_coverage_pct"].isna().all())

    def test_public_results_do_not_mutate_cached_results(self):
        result = self.analysis()
        result["selected_targets"].loc[:, "score"] = -100.0
        result["metadata"]["radius_m"] = -1
        fresh = self.analysis()
        self.assertTrue(fresh["selected_targets"]["score"].ge(0).all())
        self.assertEqual(fresh["metadata"]["radius_m"], 250)

    def test_invalid_controls_and_partial_evidence_dates_are_rejected(self):
        for kwargs in [dict(radius_m=99), dict(budget=0), dict(strategy="unknown"),
                       dict(start_date="2025-09-02", end_date="2025-09-01"),
                       dict(start_date="2025-07-31"), dict(end_date="2025-10-01"),
                       dict(start_date="2025-08-01T12:00:00")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.analysis(**kwargs)
        with self.assertRaises(ValueError):
            get_area_evidence("unknown", 250, start_date="2025-08-01", data_dir=self.directory)

    def test_duplicate_ids_and_inconsistent_crash_flags_are_rejected(self):
        self.reports.loc[1, "unique_key"] = "0001"
        self.write()
        with self.assertRaises(ValueError):
            load_events(self.directory)
        self.reports.loc[1, "unique_key"] = "0002"
        self.crashes.loc[1, "has_fatality"] = False
        self.write()
        with self.assertRaises(ValueError):
            load_events(self.directory)

    def test_radius_and_spacing_boundaries(self):
        self.reports = pd.DataFrame([report("1", 0, "2025-08-01"), report("2", 250, "2025-08-01")])
        self.crashes = pd.DataFrame([crash("1", 100, "2025-08-01", injured=1)])
        self.write()
        result = self.analysis(radius_m=100)
        self.assertEqual(result["targets_scored"]["harm_crash_count"].max(), 1)
        self.assertEqual(len(result["selected_targets"]), 2, "Exactly 250 m separation is allowed")


class SpatialAndScoringTests(unittest.TestCase):
    def test_spatial_bucket_queries_match_brute_force_at_all_radii(self):
        rng = np.random.default_rng(2026)
        points = np.column_stack([rng.uniform(40.65, 40.75, 1000), rng.uniform(-74.05, -73.95, 1000)])
        index = RadiusIndex(points)
        for origin in points[::29]:
            distances = haversine_m(*origin, points)
            for radius in config.RADII:
                actual, _ = index.query(*origin, radius)
                np.testing.assert_array_equal(actual, np.flatnonzero(distances <= radius + 1e-7))

    def test_equal_counts_receive_equal_percentiles_and_target_id_breaks_order(self):
        features = pd.DataFrame(dict(target_id=["b", "a", "c"], signal_report_count=[2, 2, 1], harm_crash_count=[0, 0, 0]))
        crash_scores = score_candidates(features, "crash_first")
        self.assertEqual(crash_scores["target_id"].tolist(), ["a", "b", "c"])
        self.assertEqual(crash_scores["score"].nunique(), 1)
        report_scores = score_candidates(features, "complaint_first").set_index("target_id")
        self.assertEqual(report_scores.loc["a", "score"], report_scores.loc["b", "score"])


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AnalysisEngine(config.DATA_DIR, input_version(config.DATA_DIR))

    def test_real_data_audit_matches_snapshots(self):
        quality = get_data_quality()
        self.assertEqual(quality["signal_reports"], 6119)
        self.assertEqual(quality["collisions"], 14066)
        self.assertEqual(quality["harm_crashes"], 6536)
        self.assertEqual(quality["fatal_only_crashes"], 32)
        self.assertEqual(quality["months"]["september"]["harm_crashes"], 3400)
        self.assertEqual(quality["signal_reports_excluded"], 415)
        self.assertEqual(quality["collisions_excluded"], 460)

    def test_candidates_are_fixed_and_neighborhood_counts_are_monotonic(self):
        previous = None
        for radius in config.RADII:
            features = self.engine.features("2025-08-01", "2025-08-31", radius)
            if previous is not None:
                assert_frame_equal(features[["target_id", "latitude", "longitude"]], previous[["target_id", "latitude", "longitude"]])
                self.assertTrue(features["signal_report_count"].ge(previous["signal_report_count"]).all())
                self.assertTrue(features["harm_crash_count"].ge(previous["harm_crash_count"]).all())
            previous = features

    def test_default_features_and_evidence_match_independent_full_scan(self):
        result = get_analysis()
        for candidate in result["selected_targets"].head(5).itertuples():
            evidence = get_area_evidence(candidate.target_id, 250,
                                         start_date=config.DEFAULT_START_DATE, end_date=config.DEFAULT_END_DATE)
            for name, source, date_column in (("reports", self.engine.reports, "created_date"),
                                              ("collisions", self.engine.collisions, "occurred_at")):
                distance = haversine_m(candidate.latitude, candidate.longitude, source[["latitude", "longitude"]].to_numpy())
                expected = ((distance <= 250 + 1e-7)
                            & source[date_column].ge(config.DEFAULT_START_DATE).to_numpy()
                            & source[date_column].lt("2025-09-01").to_numpy())
                id_column = "unique_key" if name == "reports" else "collision_id"
                self.assertEqual(set(evidence[name][id_column]), set(source.loc[expected, id_column]))
            self.assertEqual(len(evidence["reports"]), candidate.signal_report_count)
            self.assertEqual(int(evidence["collisions"]["has_harm"].sum()), candidate.harm_crash_count)

    def test_all_252_weekly_settings_have_valid_spacing_and_unique_coverage(self):
        configurations = 0
        for radius in config.RADII:
            for budget in config.BUDGETS:
                evaluation = self.engine.historical_evaluation(radius, budget)
                for row in evaluation.itertuples():
                    _, selected = self.engine.scored_and_selected(row.start_date, row.end_date, radius, row.strategy, budget)
                    points = selected[["latitude", "longitude"]].to_numpy()
                    for index, point in enumerate(points):
                        self.assertTrue((haversine_m(*point, points[index + 1:]) >= 250 - 1e-7).all())
                    self.assertEqual(row.selected_count, len(selected))
                    self.assertLessEqual(row.selected_count, budget)
                    # Independent OR masks over all source points, not neighborhood caches.
                    reports = self.engine.reports
                    crashes = self.engine.collisions
                    report_union = np.zeros(len(reports), dtype=bool)
                    crash_union = np.zeros(len(crashes), dtype=bool)
                    for point in points:
                        report_union |= haversine_m(*point, reports[["latitude", "longitude"]].to_numpy()) <= radius + 1e-7
                        crash_union |= haversine_m(*point, crashes[["latitude", "longitude"]].to_numpy()) <= radius + 1e-7
                    report_time_mask, _ = self.engine._time_masks(row.start_date, row.end_date)
                    _, crash_time_mask = self.engine._time_masks(row.forecast_start_date, row.forecast_end_date)
                    reports_covered = int((report_union & report_time_mask).sum())
                    crashes_covered = int((crash_union & crash_time_mask & self.engine.harm).sum())
                    denominator = int((crash_time_mask & self.engine.harm).sum())
                    self.assertEqual(row.history_signal_reports_covered, reports_covered)
                    self.assertEqual(row.future_harm_crashes_covered, crashes_covered)
                    self.assertAlmostEqual(row.future_harm_coverage_pct, 100 * crashes_covered / denominator)
                    configurations += 1
        self.assertEqual(configurations, 252)

    def test_regression_predicts_next_seven_days_and_reports_actual_out_of_time_metrics(self):
        result = get_analysis()
        self.assertEqual(result["metadata"]["forecast_start_date"], "2025-09-01")
        self.assertEqual(result["metadata"]["forecast_end_date"], "2025-09-07")
        self.assertEqual(result["model"]["training_last_feature_end_date"], "2025-08-24")
        self.assertEqual(result["model"]["training_labels_end_date"], "2025-08-31")
        self.assertTrue(result["targets_scored"]["predicted_harm_crashes_next_7d"].ge(0).all())
        predictions = self.engine.regression.weekly_predictions(250)
        self.assertEqual(len(predictions), 4 * len(self.engine.candidates))
        self.assertEqual(predictions["forecast_end_date"].max(), "2025-09-28")
        actual = predictions["actual_harm_crashes_next_7d"].to_numpy()
        predicted = predictions["predicted_harm_crashes_next_7d"].to_numpy()
        metrics = self.engine.regression.test_metrics(250)
        measured = metrics.loc[(metrics.estimator == "linear_regression") & (metrics.test_period == "all_complete_september_weeks")].iloc[0]
        self.assertAlmostEqual(measured.mae, np.abs(predicted - actual).mean())
        self.assertAlmostEqual(measured.rmse, np.sqrt(np.mean((predicted - actual) ** 2)))

    def test_forecast_after_available_data_is_not_presented_as_zero_observed_crashes(self):
        result = get_analysis("2025-09-24", "2025-09-30")
        self.assertTrue(result["metadata"]["model_available"])
        self.assertFalse(result["metadata"]["outcomes_available"])
        self.assertEqual(result["metadata"]["forecast_end_date"], "2025-10-07")
        self.assertTrue(result["comparison"]["future_harm_crashes_covered"].isna().all())


class RegressionMathTests(unittest.TestCase):
    def test_ols_recovers_known_intercept_and_two_coefficients(self):
        x = np.array([[0, 0], [1, 0], [0, 1], [2, 3], [4, 2]], dtype=float)
        y = 0.25 + 0.5 * x[:, 0] + 2 * x[:, 1]
        coefficients, rank = fit_ols(x, y)
        np.testing.assert_allclose(coefficients, [0.25, 0.5, 2], rtol=1e-12)
        self.assertEqual(rank, 3)

    def test_constant_target_r2_is_undefined_and_singular_features_are_supported(self):
        coefficients, rank = fit_ols(np.zeros((4, 2)), np.ones(4))
        self.assertEqual(rank, 1)
        self.assertTrue(np.isfinite(coefficients).all())
        self.assertIsNone(regression_metrics([0, 0], [0, 0])["r2"])


if __name__ == "__main__":
    unittest.main()
