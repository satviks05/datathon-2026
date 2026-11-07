"""Regression checks against manually audited original event IDs and real panels."""

import unittest

import numpy as np
import pandas as pd
from pyproj import Transformer

from src import forecast_config as cfg
from src.build_forecast import validate_panel
from src.forecast_features import build_features, load_forecast_events, source_audit
from src.forecast_metrics import evaluate, top_cells
from src.forecast_models import fit_models, predict


class RealWeeklyForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports, cls.crashes = load_forecast_events()
        cls.features = build_features(cls.reports, cls.crashes)
        cls.predictions = predict(cls.features, fit_models(cls.features), "real-test")
        cls.metrics = evaluate(cls.predictions, cls.crashes)

    def test_real_shape_coverage_and_source_exclusions(self):
        validate_panel(self.features)
        self.assertEqual(self.features.area_id.nunique(), 1180)
        self.assertEqual(len(self.features), 8260)
        self.assertEqual(len(self.predictions), 4720)
        self.assertEqual(len(self.metrics), 24)
        audit = source_audit(self.reports, self.crashes, cfg.ROOT / "data/raw")
        self.assertEqual(audit["mapped_cleaned_complaints"], 6119)
        self.assertEqual(audit["mapped_cleaned_collisions"], 14066)
        self.assertEqual(audit["raw_complaints"]["excluded_other_categories"], 28142)
        self.assertEqual(audit["raw_complaints"]["signal_category"]["coordinate_exclusions"], 415)
        self.assertEqual(audit["raw_collisions"]["missing_coordinate_pairs"], 119)
        self.assertEqual(audit["raw_collisions"]["zero_coordinate_pairs"], 341)
        self.assertEqual(self.features.drop_duplicates("area_id").activity_band.value_counts().to_dict(),
                         {"one": 629, "two_plus": 336, "zero": 215})

    def test_two_manually_verified_cells_against_original_event_ids(self):
        # Read source files independently of forecast loader/count helpers.
        reports = pd.read_csv(cfg.DATA_DIR / cfg.INPUT_NAMES[0], dtype={"unique_key": "string"}, float_precision="round_trip")
        crashes = pd.read_csv(cfg.DATA_DIR / cfg.INPUT_NAMES[1], dtype={"collision_id": "string"}, float_precision="round_trip")
        reports["time"] = pd.to_datetime(reports.created_date)
        crashes["time"] = pd.to_datetime(crashes.crash_date + " " + crashes.crash_time)
        transform = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
        for events in (reports, crashes):
            east, north = transform.transform(events.longitude.to_numpy(), events.latitude.to_numpy())
            events["area"] = [f"cell_{int(np.floor(x / 500))}_{int(np.floor(y / 500))}" for x, y in zip(east, north)]
        examples = [
            ("cell_1165_8999", "2025-08-15", "one", {"65845573"}, {"4835362"}, {"4835338", "4836748"}),
            ("cell_1166_8999", "2025-09-08", "zero", {"66069300"}, {"4839808", "4841555"}, {"4841465", "4841917"}),
        ]
        for area, start, band, complaint_ids, past_ids, target_ids in examples:
            row = self.features.loc[self.features.area_id.eq(area) & self.features.forecast_start.eq(start)].iloc[0]
            self.assertEqual(row.activity_band, band)
            for events, id_column, begin, end, expected, count in (
                (reports, "unique_key", row.feature_start, row.feature_end, complaint_ids, row.past_complaints),
                (crashes, "collision_id", row.feature_start, row.feature_end, past_ids, row.past_collisions),
                (crashes, "collision_id", row.forecast_start, row.forecast_end, target_ids, row.actual_collisions),
            ):
                actual = set(events.loc[events.area.eq(area) & events.time.ge(pd.Timestamp(begin))
                                        & events.time.lt(pd.Timestamp(end)), id_column])
                self.assertEqual(actual, expected)
                self.assertEqual(count, len(expected))

    def test_all_real_metrics_independently_recompute_from_predictions(self):
        citywide_counts = [1744, 1673, 1747, 1641]
        eligible_counts = [1090, 1070, 1147, 1036]
        for model in cfg.MODELS:
            selected_actuals = 0
            for start, citywide, eligible in zip(cfg.EVALUATION_STARTS, citywide_counts, eligible_counts):
                rows = self.predictions.loc[self.predictions.forecast_start.eq(start)]
                metric = self.metrics.loc[self.metrics.model.eq(model) & self.metrics.scope.eq("week")
                                          & self.metrics.forecast_start.eq(start)].iloc[0]
                self.assertEqual(metric.citywide_collisions, citywide)
                self.assertEqual(metric.eligible_collisions, eligible)
                self.assertAlmostEqual(metric.mae, np.abs(rows.actual_collisions - rows[f"{model}_prediction"]).mean())
                selected = top_cells(rows, model)
                self.assertEqual(len(selected), 50)
                numerator = int(rows.loc[rows.area_id.isin(selected.area_id), "actual_collisions"].sum())
                self.assertEqual(metric.top_k_collisions, numerator)
                self.assertAlmostEqual(metric.top_k_coverage_pct, 100 * numerator / citywide)
                selected_actuals += numerator
            overall = self.metrics.loc[self.metrics.model.eq(model) & self.metrics.scope.eq("overall")].iloc[0]
            self.assertEqual(overall.citywide_collisions, 6805)
            self.assertEqual(overall.eligible_collisions, 4343)
            self.assertAlmostEqual(overall.top_k_coverage_pct, 100 * selected_actuals / 6805)
            self.assertAlmostEqual(overall.eligible_coverage_pct, 100 * 4343 / 6805)


if __name__ == "__main__":
    unittest.main()
