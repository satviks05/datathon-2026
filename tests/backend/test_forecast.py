"""Tests for the handoff's weekly 500 m Poisson workflow (not legacy OLS)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pyproj import Transformer
from sklearn.exceptions import ConvergenceWarning

from src import forecast_config as cfg
from src.build_forecast import build_forecast, validate_panel
from src.forecast_features import build_features, eligible_areas, load_forecast_events, project_cells, sha256_file
from src.forecast_metrics import compare_models, coverage, evaluate, top_cells
from src.forecast_models import fit_models, predict


def location(cell):
    lon, lat = Transformer.from_crs(cfg.GRID_CRS, "EPSG:4326", always_xy=True).transform(
        (1170 + cell + 0.5) * 500, (9010 + 0.5) * 500)
    return dict(latitude=lat, longitude=lon)


class ForecastFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reports = pd.DataFrame([
            dict(unique_key=f"{i:04}", created_date=time, complaint_type="Traffic Signal Condition", **location(cell))
            for i, cell, time in [
                (1, 0, "2025-08-01T00:00:00"), (2, 2, "2025-08-07T23:59:59"),
                (3, 3, "2025-08-08T00:00:00"), (4, 0, "2025-08-25T00:00:00"),
                (5, 0, "2025-08-31T23:59:59"), (6, 0, "2025-09-01T00:00:00"),
                (7, 1, "2025-09-08T00:00:00"),
            ]])
        self.crashes = pd.DataFrame([
            dict(collision_id=f"{i:04}", crash_date=time[:10], crash_time=time[11:16], **location(cell))
            for i, cell, time in [
                (1, 0, "2025-08-01T00:00"), (2, 1, "2025-08-07T23:59"),
                (3, 1, "2025-08-02T00:00"), (4, 0, "2025-08-08T00:00"),
                (5, 0, "2025-08-14T23:59"), (6, 0, "2025-08-15T00:00"),
                (7, 1, "2025-08-22T00:00"), (8, 0, "2025-08-25T00:00"),
                (9, 0, "2025-08-31T23:59"), (10, 0, "2025-09-01T00:00"),
                (11, 0, "2025-09-07T23:59"), (12, 1, "2025-09-08T00:00"),
                (13, 3, "2025-09-03T12:00"), (14, 2, "2025-09-15T00:00"),
                (15, 0, "2025-09-29T00:00"),
            ]])
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.reports.to_csv(self.root / cfg.INPUT_NAMES[0], index=False)
        self.crashes.to_csv(self.root / cfg.INPUT_NAMES[1], index=False)

    def panel(self):
        return build_features(*load_forecast_events(self.root))

    def test_projected_cell_ids_and_inverse_centers(self):
        # Independent known UTM origin check catches axis swaps and grid offsets.
        point = project_cells(pd.DataFrame([dict(longitude=-75.0, latitude=0.0)]))
        self.assertEqual(point.area_id.iloc[0], "cell_1000_0")
        reports, crashes = load_forecast_events(self.root)
        self.assertEqual(reports.unique_key.iloc[0], "0001")
        areas = eligible_areas(reports, crashes).set_index("area_id")
        self.assertEqual(set(areas.index), {"cell_1170_9010", "cell_1171_9010", "cell_1172_9010"})
        east, north = Transformer.from_crs("EPSG:4326", cfg.GRID_CRS, always_xy=True).transform(
            areas.longitude.to_numpy(), areas.latitude.to_numpy())
        np.testing.assert_allclose(east, [585250, 585750, 586250], atol=1e-7)
        np.testing.assert_allclose(north, [4505250] * 3, atol=1e-7)

    def test_half_open_boundaries_complete_panel_and_zero_rows(self):
        features = self.panel()
        validate_panel(features)
        self.assertEqual(len(features), 21)
        self.assertEqual(features.groupby("area_id").size().tolist(), [7, 7, 7])
        indexed = features.set_index(["area_id", "forecast_start"])
        first = indexed.loc[("cell_1170_9010", "2025-08-08")]
        self.assertEqual((first.past_collisions, first.actual_collisions), (1, 2))
        september = indexed.loc[("cell_1170_9010", "2025-09-01")]
        self.assertEqual((september.past_complaints, september.past_collisions, september.actual_collisions), (2, 2, 2))
        following = indexed.loc[("cell_1170_9010", "2025-09-08")]
        self.assertEqual((following.past_complaints, following.past_collisions), (1, 2))
        empty = indexed.loc[("cell_1172_9010", "2025-09-08")]
        self.assertEqual((empty.past_complaints, empty.past_collisions, empty.actual_collisions), (0, 0, 0))
        self.assertEqual(set(features.forecast_start), set(cfg.TRAIN_STARTS + cfg.EVALUATION_STARTS))
        self.assertEqual(features.forecast_end.max(), "2025-09-29")

    def test_eligibility_and_bands_ignore_all_later_events(self):
        before = self.panel()
        self.crashes.loc[self.crashes.crash_date.ge("2025-08-08"), ["latitude", "longitude"]] = list(location(4).values())
        self.reports.loc[self.reports.created_date.ge("2025-08-08"), ["latitude", "longitude"]] = list(location(4).values())
        self.write()
        after = self.panel()
        frozen = ["area_id", "latitude", "longitude", "activity_band"]
        assert_frame_equal(before[frozen], after[frozen])
        self.assertEqual(before.drop_duplicates("area_id").activity_band.tolist(), ["one", "two_plus", "zero"])

    def test_models_use_training_only_log_scaled_counts(self):
        features = self.panel()
        models = fit_models(features)
        train = features.loc[features.split.eq("train")]
        for name, columns in cfg.MODEL_FEATURES.items():
            model = models[name]
            expected = np.log1p(train[columns].to_numpy())
            np.testing.assert_allclose(model["scaler"].mean_, expected.mean(axis=0))
            np.testing.assert_allclose(model["scaler"].var_, expected.var(axis=0))
            self.assertEqual(model["scaler"].n_samples_seen_, 9)
            self.assertEqual(model["poisson"].alpha, 1.0)
            self.assertEqual(model["poisson"].max_iter, 1000)
            self.assertTrue(model["poisson"].fit_intercept)
        altered = features.copy()
        altered.loc[altered.split.eq("evaluation"), ["past_complaints", "past_collisions", "actual_collisions"]] = 1000000
        other = fit_models(altered)
        for name in models:
            np.testing.assert_array_equal(models[name]["poisson"].coef_, other[name]["poisson"].coef_)
            np.testing.assert_array_equal(models[name]["scaler"].mean_, other[name]["scaler"].mean_)
        prediction = predict(features, models, "test")
        np.testing.assert_array_equal(prediction.persistence_prediction, prediction.past_collisions)
        self.assertTrue(np.isfinite(prediction[[f"{m}_prediction" for m in cfg.MODELS]]).all().all())
        self.assertTrue(prediction[[f"{m}_prediction" for m in cfg.MODELS]].ge(0).all().all())

    def test_new_observation_updates_only_available_later_feature_windows(self):
        before = self.panel()
        models = fit_models(before)
        p_before = predict(before, models, "test")
        extra = dict(collision_id="1000", crash_date="2025-09-03", crash_time="12:00", **location(0))
        self.crashes = pd.concat([self.crashes, pd.DataFrame([extra])], ignore_index=True)
        self.write()
        after = self.panel()
        p_after = predict(after, fit_models(after), "test")
        cols = [f"{m}_prediction" for m in cfg.MODELS]
        first = p_before.forecast_start.eq("2025-09-01")
        assert_frame_equal(p_before.loc[first, cols], p_after.loc[first, cols])
        following = p_before.forecast_start.eq("2025-09-08") & p_before.area_id.eq("cell_1170_9010")
        self.assertEqual(float(p_after.loc[following, "persistence_prediction"].iloc[0] - p_before.loc[following, "persistence_prediction"].iloc[0]), 1)
        self.assertGreater(float(p_after.loc[following, "collision_only_prediction"].iloc[0]), float(p_before.loc[following, "collision_only_prediction"].iloc[0]))

    def test_convergence_failure_is_not_silently_accepted(self):
        def failed_fit(*args, **kwargs):
            warnings.warn("not converged", ConvergenceWarning)
        with patch("src.forecast_models.PoissonRegressor.fit", side_effect=failed_fit):
            with self.assertRaises(ConvergenceWarning):
                fit_models(self.panel())

    def test_reject_invalid_inputs_and_duplicate_ids(self):
        original = self.reports.copy()
        for column, value in [("unique_key", "0002"), ("unique_key", " "),
                              ("latitude", np.nan), ("longitude", 0),
                              ("created_date", "2025-10-01T00:00:00"),
                              ("created_date", "invalid"), ("complaint_type", "Other")]:
            with self.subTest(column=column, value=value):
                self.reports = original.copy()
                self.reports.loc[0, column] = value
                self.write()
                with self.assertRaises((ValueError, TypeError)):
                    load_forecast_events(self.root)

    def test_metrics_denominators_bands_and_prediction_agreement(self):
        features = self.panel()
        predictions = predict(features, fit_models(features), "test")
        _, crashes = load_forecast_events(self.root)
        metrics = evaluate(predictions, crashes)
        self.assertEqual(len(metrics), 24)
        for model in cfg.MODELS:
            weekly = metrics.loc[metrics.model.eq(model) & metrics.scope.eq("week")].set_index("forecast_start")
            self.assertEqual(int(weekly.loc["2025-09-01", "citywide_collisions"]), 3)
            self.assertEqual(int(weekly.loc["2025-09-01", "eligible_collisions"]), 2)
            self.assertAlmostEqual(weekly.loc["2025-09-01", "top_k_coverage_pct"], 200 / 3)
            self.assertTrue(pd.isna(weekly.loc["2025-09-22", "top_k_coverage_pct"]))
            self.assertTrue(weekly.top_k.eq(50).all())  # requested K even with only 3 eligible cells
            overall = metrics.loc[metrics.model.eq(model) & metrics.scope.eq("overall")].iloc[0]
            self.assertEqual(int(overall.citywide_collisions), 5)
            self.assertEqual(int(overall.eligible_collisions), 4)
            self.assertAlmostEqual(overall.top_k_coverage_pct, 80)
            self.assertAlmostEqual(overall.mae, (predictions.actual_collisions - predictions[f"{model}_prediction"]).abs().mean())
            self.assertNotAlmostEqual(overall.top_k_coverage_pct, weekly.top_k_coverage_pct.mean())
            for row in metrics.loc[metrics.model.eq(model) & metrics.scope.eq("activity_band")].itertuples():
                part = predictions.loc[predictions.activity_band.eq(row.activity_band)]
                self.assertEqual(row.area_week_rows, len(part))
                self.assertAlmostEqual(row.mae, (part.actual_collisions - part[f"{model}_prediction"]).abs().mean())
                self.assertTrue(pd.isna(row.top_k))
                self.assertTrue(pd.isna(row.top_k_collisions))
                self.assertTrue(pd.isna(row.citywide_collisions))
        self.assertEqual(len(compare_models(metrics)), 10)

    def test_metric_validation_rejects_inconsistent_actuals_or_missing_rows(self):
        features = self.panel()
        predictions = predict(features, fit_models(features), "test")
        _, crashes = load_forecast_events(self.root)
        wrong = predictions.copy()
        wrong.loc[0, "actual_collisions"] += 1
        with self.assertRaisesRegex(ValueError, "source collision IDs"):
            evaluate(wrong, crashes)
        with self.assertRaisesRegex(ValueError, "same eligible cells"):
            evaluate(predictions.iloc[1:], crashes)
        mixed = predictions.copy()
        mixed.loc[0, "run_id"] = "different-run"
        with self.assertRaisesRegex(ValueError, "one nonempty prediction run"):
            evaluate(mixed, crashes)

    def test_empty_bands_and_zero_mae_denominators_remain_unavailable(self):
        features = self.panel()
        predictions = predict(features, fit_models(features), "test")
        _, crashes = load_forecast_events(self.root)
        predictions["activity_band"] = "one"
        predictions["actual_collisions"] = 0
        for model in cfg.MODELS:
            predictions[f"{model}_prediction"] = 0.0
        metrics = evaluate(predictions, crashes.iloc[:0])
        self.assertEqual(set(metrics.loc[metrics.scope.eq("activity_band"), "activity_band"]), {"one"})
        self.assertTrue(metrics.top_k_coverage_pct.isna().all())
        for comparison in compare_models(metrics):
            self.assertEqual(comparison["mae_reduction"], 0)
            self.assertIsNone(comparison["relative_mae_improvement_pct"])
            self.assertIsNone(comparison["top_k_coverage_difference_pp"])

    def test_build_reproducibility_provenance_and_unchanged_sources(self):
        original = {name: (self.root / name).read_bytes() for name in cfg.INPUT_NAMES}
        first = build_forecast(self.root, self.root / "first")
        second = build_forecast(self.root, self.root / "second")
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["rows"], {"features.csv": 21, "predictions.csv": 12, "metrics.csv": 24})
        for name in ("features.csv", "predictions.csv", "metrics.csv"):
            self.assertEqual((self.root / "first" / name).read_bytes(), (self.root / "second" / name).read_bytes())
            self.assertEqual(first["output_sha256"][name], sha256_file(self.root / "first" / name))
        for name in cfg.INPUT_NAMES:
            self.assertEqual((self.root / name).read_bytes(), original[name])
            self.assertEqual(first["input_sha256"][name], sha256_file(self.root / name))
        predictions = pd.read_csv(self.root / "first/predictions.csv")
        metrics = pd.read_csv(self.root / "first/metrics.csv")
        self.assertEqual(list(predictions.columns), cfg.PREDICTION_COLUMNS)
        self.assertEqual(list(metrics.columns), cfg.METRIC_COLUMNS)
        self.assertEqual(set(predictions.run_id) | set(metrics.run_id), {first["run_id"]})
        self.assertFalse(first["source_audit"]["raw_audit_available"])
        # Publication replaces only complete staged files and leaves no staging directory.
        self.assertEqual({p.name for p in (self.root / "first").iterdir()}, set(cfg.OUTPUT_NAMES))
        json.loads((self.root / "first/run_metadata.json").read_text())

    def test_failed_build_preserves_existing_outputs(self):
        output = self.root / "output"
        build_forecast(self.root, output)
        original = {p.name: p.read_bytes() for p in output.iterdir()}
        with patch("src.build_forecast.fit_models", side_effect=ValueError("failed fit")):
            with self.assertRaisesRegex(ValueError, "failed fit"):
                build_forecast(self.root, output)
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, original)


class ForecastRankingTests(unittest.TestCase):
    def test_top_50_ties_use_lexicographic_ids_after_predictions(self):
        frame = pd.DataFrame({"area_id": [f"cell_{i}_0" for i in range(52)],
                              "combined_prediction": [0.0] * 52})
        shuffled = frame.sample(frac=1, random_state=12)
        expected = sorted(frame.area_id)[:50]
        self.assertEqual(top_cells(shuffled, "combined").area_id.tolist(), expected)
        shuffled.loc[shuffled.area_id.eq("cell_9_0"), "combined_prediction"] = 2.0
        self.assertEqual(top_cells(shuffled, "combined").area_id.iloc[0], "cell_9_0")

    def test_zero_denominator_is_unavailable(self):
        self.assertIsNone(coverage(0, 0))
        self.assertEqual(coverage(0, 10), 0)


if __name__ == "__main__":
    unittest.main()
