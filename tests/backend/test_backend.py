"""Temporal isolation, pure August training, spatial checks and real evaluation."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src import config
from src.api import get_analysis, get_area_evidence, get_data_quality, get_model_metrics, get_september_forecast
from src.data_split import split_inputs
from src.engine import AnalysisEngine
from src.loaders import input_version, load_events
from src.ranking import score_candidates
from src.regression import AugustOnlyRegression, OUTCOMES, fit_ols, regression_metrics
from src.spatial import RadiusIndex, build_candidates, haversine_m
from src.train import train_and_forecast


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
        self.temp_root = (config.ROOT / "tmp" / "backend-tests").resolve()
        self.temp_root.relative_to(config.ROOT.resolve())
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.temp_root)
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
        # Verify the absolute cleanup target remains inside this task's scratch tree.
        self.directory.resolve().relative_to(self.temp_root)
        self.temporary.cleanup()

    def write(self):
        self.reports.to_csv(self.directory / "signal_complaints_clean.csv", index=False)
        self.crashes.to_csv(self.directory / "collisions_clean.csv", index=False)

    def analysis(self, **kwargs):
        return get_analysis(data_dir=self.directory, **kwargs)

    def test_august_layers_keep_actual_times_and_do_not_show_september_forecasts(self):
        result = self.analysis(start_date="2025-08-01", end_date="2025-08-01")
        self.assertEqual(set(result["reports"]["unique_key"]), {"0001", "0002"})
        self.assertEqual(result["collisions"]["collision_id"].tolist(), ["0002", "0001"])
        self.assertFalse(result["metadata"]["model_available"])
        self.assertTrue(result["selected_targets"].empty)

    def test_fatal_only_crash_and_original_string_ids_are_preserved(self):
        result = self.analysis()
        self.assertTrue(result["targets_scored"]["harm_crash_count"].eq(2).all())
        target = result["selected_targets"].iloc[0]["target_id"]
        evidence = get_area_evidence(target, 250, period="august", data_dir=self.directory)
        self.assertEqual(set(evidence["collisions"]["collision_id"]), {"0001", "0002", "0003"})
        fatal = evidence["collisions"].set_index("collision_id").loc["0002"]
        self.assertTrue(fatal["has_harm"])
        self.assertFalse(fatal["has_injury"])

    def test_union_counts_deduplicate_overlapping_areas(self):
        result = self.analysis()
        self.assertEqual(len(result["selected_targets"]), 2)
        self.assertEqual(int(result["selected_targets"]["harm_crash_count"].sum()), 4)
        for row in result["comparison"].itertuples():
            self.assertEqual(row.future_harm_crashes_covered, 1)
            self.assertEqual(row.history_signal_reports_covered, 3)
            self.assertEqual(row.future_harm_coverage_pct, 100)

    def test_changing_all_september_information_cannot_change_any_september_prediction(self):
        before = self.analysis()
        self.reports.loc[3, "latitude"] = north(0)
        self.reports.loc[3, "created_date"] = "2025-09-29T23:59:59"
        self.crashes.loc[3, "latitude"] = north(1200)
        self.crashes.loc[3, "number_of_persons_injured"] = 500
        self.write()
        after = self.analysis()
        assert_frame_equal(before["targets_scored"], after["targets_scored"])
        assert_frame_equal(before["selected_targets"], after["selected_targets"])
        self.assertEqual(before["model"], after["model"])
        self.assertNotEqual(before["metadata"]["input_version"], after["metadata"]["input_version"])
        self.assertNotEqual(before["area_comparison"]["actual_harm_crashes"].sum(), after["area_comparison"]["actual_harm_crashes"].sum())

    def test_training_model_explicitly_rejects_september_rows(self):
        reports, crashes = load_events(self.directory)
        with self.assertRaisesRegex(ValueError, "August data ONLY"):
            AugustOnlyRegression(reports, crashes, build_candidates(reports))

    def test_split_files_are_disjoint_and_forecast_runs_with_only_august_files(self):
        original = {name: (self.directory / name).read_bytes() for name in ("signal_complaints_clean.csv", "collisions_clean.csv")}
        split = split_inputs(self.directory)
        self.assertEqual(split["periods"]["august_train"]["signal_reports"], 3)
        train = self.directory / "august_train"
        self.assertEqual({path.name for path in train.iterdir()}, set(original))
        train_reports, train_crashes = load_events(train)
        training_columns = pd.read_csv(train / "signal_complaints_clean.csv", nrows=0).columns
        self.assertNotIn("closed_date", training_columns)
        self.assertNotIn("status", training_columns)
        actual_reports, actual_crashes = load_events(self.directory / "september_actuals")
        self.assertTrue(train_reports.created_date.lt("2025-09-01").all())
        self.assertTrue(train_crashes.occurred_at.lt("2025-09-01").all())
        self.assertTrue(actual_reports.created_date.ge("2025-09-01").all())
        self.assertTrue(actual_crashes.occurred_at.ge("2025-09-01").all())
        self.assertFalse(set(train_reports.unique_key) & set(actual_reports.unique_key))
        for name, contents in original.items():
            self.assertEqual((self.directory / name).read_bytes(), contents)
        forecast = get_september_forecast(train_dir=train)
        self.assertTrue(all(f"predicted_{outcome}" in forecast for outcome in OUTCOMES))
        manifest = train_and_forecast(train, self.directory / "forecast_only")
        self.assertFalse(manifest["september_observations_used"])
        self.assertTrue((self.directory / "forecast_only" / "september_forecasts.csv").exists())

    def test_date_ranges_aggregate_frozen_daily_forecasts_consistently(self):
        full = self.analysis()["targets_scored"].set_index("target_id")
        first = self.analysis(start_date="2025-09-01", end_date="2025-09-15")["targets_scored"].set_index("target_id")
        last = self.analysis(start_date="2025-09-16", end_date="2025-09-30")["targets_scored"].set_index("target_id")
        for outcome in OUTCOMES:
            np.testing.assert_allclose(full[f"predicted_{outcome}"],
                                       first.loc[full.index, f"predicted_{outcome}"] + last.loc[full.index, f"predicted_{outcome}"])

    def test_public_results_are_independent_copies(self):
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

    def test_bad_ids_and_inconsistent_flags_are_rejected(self):
        self.reports.loc[1, "unique_key"] = "0001"
        self.write()
        with self.assertRaises(ValueError):
            load_events(self.directory)
        self.reports.loc[1, "unique_key"] = "0002"
        self.crashes.loc[1, "has_fatality"] = False
        self.write()
        with self.assertRaises(ValueError):
            load_events(self.directory)

    def test_exact_radius_and_spacing_boundaries(self):
        self.reports = pd.DataFrame([report("1", 0, "2025-08-01"), report("2", 250, "2025-08-01")])
        self.crashes = pd.DataFrame([crash("1", 100, "2025-08-01", injured=1)])
        self.write()
        result = self.analysis(radius_m=100)
        self.assertEqual(result["targets_scored"]["harm_crash_count"].max(), 1)
        self.assertEqual(len(result["selected_targets"]), 2)


class SpatialAndRegressionTests(unittest.TestCase):
    def test_bucket_queries_match_full_scan_at_every_radius(self):
        rng = np.random.default_rng(2026)
        points = np.column_stack([rng.uniform(40.65, 40.75, 1000), rng.uniform(-74.05, -73.95, 1000)])
        index = RadiusIndex(points)
        for origin in points[::29]:
            distances = haversine_m(*origin, points)
            for radius in config.RADII:
                actual, _ = index.query(*origin, radius)
                np.testing.assert_array_equal(actual, np.flatnonzero(distances <= radius + 1e-7))

    def test_ols_recovers_known_coefficients_for_two_targets(self):
        x = np.array([[0, 0], [1, 0], [0, 1], [2, 3], [4, 2]], dtype=float)
        y = np.column_stack([0.25 + 0.5 * x[:, 0] + 2 * x[:, 1], 1 + 3 * x[:, 0] - x[:, 1]])
        coefficients, rank = fit_ols(x, y)
        np.testing.assert_allclose(coefficients, [[0.25, 1], [0.5, 3], [2, -1]], rtol=1e-12)
        self.assertEqual(rank, 3)
        self.assertIsNone(regression_metrics([0, 0], [0, 0])["r2"])

    def test_equal_counts_have_equal_percentiles_and_stable_id_order(self):
        frame = pd.DataFrame(dict(target_id=["b", "a", "c"], signal_report_count=[2, 2, 1], harm_crash_count=[0, 0, 0]))
        scored = score_candidates(frame, "crash_first")
        self.assertEqual(scored.target_id.tolist(), ["a", "b", "c"])
        self.assertEqual(scored.score.nunique(), 1)


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AnalysisEngine(config.DATA_DIR, input_version(config.DATA_DIR))

    def test_input_audit(self):
        quality = get_data_quality()
        self.assertEqual(quality["signal_reports"], 6119)
        self.assertEqual(quality["collisions"], 14066)
        self.assertEqual(quality["harm_crashes"], 6536)
        self.assertEqual(quality["fatal_only_crashes"], 32)
        self.assertEqual(quality["months"]["september"]["harm_crashes"], 3400)
        self.assertEqual(quality["signal_reports_excluded"], 415)
        self.assertEqual(quality["collisions_excluded"], 460)

    def test_fixed_candidates_and_monotonic_neighborhood_counts(self):
        previous = None
        for radius in config.RADII:
            features = self.engine.features("2025-08-01", "2025-08-31", radius)
            if previous is not None:
                assert_frame_equal(features[["target_id", "latitude", "longitude"]], previous[["target_id", "latitude", "longitude"]])
                self.assertTrue(features.signal_report_count.ge(previous.signal_report_count).all())
                self.assertTrue(features.harm_crash_count.ge(previous.harm_crash_count).all())
            previous = features

    def test_actual_area_comparison_matches_original_event_records(self):
        result = get_analysis()
        comparison = result["area_comparison"].set_index("target_id")
        for target in result["selected_targets"].head(5).itertuples():
            august = get_area_evidence(target.target_id, 250, period="august")
            september = get_area_evidence(target.target_id, 250, period="september")
            self.assertEqual(len(august["reports"]), target.signal_report_count)
            self.assertEqual(len(september["reports"]), comparison.loc[target.target_id, "actual_signal_reports"])
            self.assertEqual(len(september["collisions"]), comparison.loc[target.target_id, "actual_collisions"])
            self.assertEqual(int(september["collisions"].has_harm.sum()), comparison.loc[target.target_id, "actual_harm_crashes"])

    def test_all_63_settings_have_valid_spacing_and_unique_september_coverage(self):
        checks = 0
        report_points = self.engine.reports[["latitude", "longitude"]].to_numpy()
        crash_points = self.engine.collisions[["latitude", "longitude"]].to_numpy()
        for radius in config.RADII:
            for budget in config.BUDGETS:
                for row in self.engine.historical_evaluation(radius, budget).itertuples():
                    _, selected = self.engine.scored_and_selected("2025-09-01", "2025-09-30", radius, row.strategy, budget)
                    points = selected[["latitude", "longitude"]].to_numpy()
                    reports = np.zeros(len(report_points), dtype=bool)
                    crashes = np.zeros(len(crash_points), dtype=bool)
                    for index, point in enumerate(points):
                        self.assertTrue((haversine_m(*point, points[index + 1:]) >= 250 - 1e-7).all())
                        reports |= haversine_m(*point, report_points) <= radius + 1e-7
                        crashes |= haversine_m(*point, crash_points) <= radius + 1e-7
                    august_reports, _ = self.engine._time_masks("2025-08-01", "2025-08-31")
                    _, september_crashes = self.engine._time_masks("2025-09-01", "2025-09-30")
                    self.assertEqual(row.history_signal_reports_covered, int((reports & august_reports).sum()))
                    self.assertEqual(row.future_harm_crashes_covered, int((crashes & september_crashes & self.engine.harm).sum()))
                    self.assertEqual(row.future_harm_crashes_total, 3400)
                    self.assertEqual(row.selected_count, len(selected))
                    self.assertLessEqual(len(selected), budget)
                    checks += 1
        self.assertEqual(checks, 63)

    def test_month_forecasts_are_nonnegative_and_harm_cannot_exceed_all_collisions(self):
        result = get_analysis()
        self.assertFalse(result["metadata"]["september_observations_used_for_predictions"])
        self.assertEqual(result["model"]["training_labels_end_date"], "2025-08-31")
        for outcome in OUTCOMES:
            self.assertTrue(result["targets_scored"][f"predicted_{outcome}"].ge(0).all())
        self.assertTrue(result["targets_scored"].predicted_harm_crashes.le(result["targets_scored"].predicted_collisions).all())
        for radius in config.RADII:
            predicted, raw = self.engine.regression.forecast(radius)
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue(np.isfinite(raw).all())

    def test_each_target_metric_is_computed_from_its_own_september_actuals(self):
        metrics = get_model_metrics()
        self.assertEqual(len(metrics), 12)
        comparison = self.engine.area_comparison("2025-09-01", "2025-09-30", 250)
        for outcome in OUTCOMES:
            row = metrics.loc[(metrics.outcome == outcome) & (metrics.estimator == "linear_regression")].iloc[0]
            error = comparison[f"predicted_{outcome}"] - comparison[f"actual_{outcome}"]
            self.assertAlmostEqual(row.mae, np.abs(error).mean())
            self.assertAlmostEqual(row.rmse, np.sqrt(np.mean(error ** 2)))


if __name__ == "__main__":
    unittest.main()
