"""Forecast UI integration and provenance checks, independent of model training."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ui import data
from ui.map_view import source_map_frame, make_source_map


class ExportTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.reset_exports()
        self.data_patch = patch.object(data, "DATA", self.directory)
        self.data_patch.start()

    def tearDown(self):
        self.data_patch.stop()
        self.temp.cleanup()

    def reset_exports(self):
        for filename in data.EXPORTS:
            shutil.copyfile(data.FIXTURES / filename, self.directory / filename)
        meta = json.loads((self.directory / "run_metadata.json").read_text())
        meta.pop("development_fixture")
        for filename in data.SOURCES:
            payload = ("test-only source bytes: " + filename).encode()
            (self.directory / filename).write_bytes(payload)
            meta["input_sha256"][filename] = sha256(payload).hexdigest()
        (self.directory / "run_metadata.json").write_text(json.dumps(meta))

    def update_meta(self, **values):
        path = self.directory / "run_metadata.json"
        meta = json.loads(path.read_text())
        meta.update(values)
        path.write_text(json.dumps(meta))

    def update_table(self, filename, change):
        path = self.directory / filename
        table = pd.read_csv(path)
        change(table).to_csv(path, index=False)


class ExportValidationTests(ExportTestCase):
    def test_complete_bundle_and_zero_event_cells(self):
        bundle = data.load_forecast_bundle()
        self.assertFalse(bundle.stale_reasons)
        self.assertEqual(len(bundle.predictions), 12)
        self.assertEqual(set(bundle.predictions.forecast_start), set(data.WEEKS))
        self.assertEqual(int(bundle.predictions.actual_collisions.eq(0).sum()), 4)
        self.assertFalse(bundle.fixture)

    def test_missing_outputs_never_fall_back_to_old_files(self):
        (self.directory / "predictions.csv").unlink()
        (self.directory / "selected_targets.csv").write_text("legacy,output\n1,2\n")
        with self.assertRaisesRegex(FileNotFoundError, "predictions.csv"):
            data.load_forecast_bundle()

    def test_unsupported_schema_and_production_fixture_rejected(self):
        self.update_meta(schema_version=2)
        with self.assertRaisesRegex(data.ForecastDataError, "schema_version"):
            data.load_forecast_bundle()
        self.reset_exports()
        self.update_meta(development_fixture=True)
        with self.assertRaisesRegex(data.ForecastDataError, "Fixture metadata"):
            data.load_forecast_bundle()

    def test_mixed_run_ids_rejected(self):
        for filename in ("predictions.csv", "metrics.csv"):
            with self.subTest(filename=filename):
                self.reset_exports()
                self.update_table(filename, lambda table: table.assign(run_id="wrong-run"))
                with self.assertRaisesRegex(data.ForecastDataError, "Mixed run"):
                    data.load_forecast_bundle()

    def test_stale_hash_and_missing_input_are_flagged(self):
        (self.directory / data.SOURCES[0]).write_bytes(b"changed input")
        bundle = data.load_forecast_bundle()
        self.assertIn("Input file differs", bundle.stale_reasons[0])
        (self.directory / data.SOURCES[1]).unlink()
        self.assertEqual(len(data.load_forecast_bundle().stale_reasons), 2)

    def test_windows_newlines_are_verified_without_hiding_changed_records(self):
        filename = data.SOURCES[0]
        lf_content = b"id,value\n1,2\n"
        path = self.directory / filename
        meta = json.loads((self.directory / "run_metadata.json").read_text())
        meta["input_sha256"][filename] = sha256(lf_content).hexdigest()
        self.update_meta(input_sha256=meta["input_sha256"])
        path.write_bytes(lf_content.replace(b"\n", b"\r\n"))
        bundle = data.load_forecast_bundle()
        self.assertFalse(bundle.stale_reasons)
        self.assertTrue(any("CRLF" in note for note in bundle.provenance_notes))
        path.write_bytes(b"id,value\r\n1,999\r\n")
        self.assertTrue(data.load_forecast_bundle().stale_reasons)

    def test_malformed_predictions_rejected(self):
        mutations = {
            "missing column": lambda table: table.drop(columns="combined_prediction"),
            "negative prediction": lambda table: table.assign(combined_prediction=-1),
            "nonfinite prediction": lambda table: table.assign(combined_prediction=float("inf")),
            "duplicate area": lambda table: pd.concat([table, table.iloc[:1]]),
            "missing cell-week": lambda table: table.iloc[1:],
            "future feature": lambda table: table.assign(feature_end="2025-10-01"),
        }
        for name, change in mutations.items():
            with self.subTest(name=name):
                self.reset_exports()
                self.update_table("predictions.csv", change)
                with self.assertRaises(data.ForecastDataError):
                    data.load_forecast_bundle()

    def test_malformed_metrics_rejected(self):
        for name, change in {
            "missing model/week": lambda table: table.iloc[1:],
            "negative error": lambda table: table.assign(mae=-1),
            "wrong denominator": lambda table: table.assign(citywide_collisions=0),
            "duplicate metrics": lambda table: pd.concat([table, table.iloc[:1]]),
        }.items():
            with self.subTest(name=name):
                self.reset_exports()
                self.update_table("metrics.csv", change)
                with self.assertRaises(data.ForecastDataError):
                    data.load_forecast_bundle()

    def test_cache_observes_new_run_and_changed_files(self):
        before = data.load_forecast_bundle()
        self.update_meta(run_id="second-run")
        for filename in ("predictions.csv", "metrics.csv"):
            self.update_table(filename, lambda table: table.assign(run_id="second-run"))
        after = data.load_forecast_bundle()
        self.assertNotEqual(before.metadata["run_id"], after.metadata["run_id"])
        self.assertEqual(after.predictions.run_id.unique().tolist(), ["second-run"])

    def test_top50_table_ties_are_deterministic(self):
        bundle = data.load_forecast_bundle()
        week = bundle.predictions.iloc[:1]
        rows = []
        for i in range(60):
            row = week.copy()
            row["area_id"] = f"cell_{i:04d}_9020"
            row["combined_prediction"] = 1.0
            rows.append(row)
        bundle.predictions = pd.concat(rows[::-1], ignore_index=True)
        top = data.week_rows(bundle, data.WEEKS[0], "combined", True)
        self.assertEqual(top.area_id.iloc[0], "cell_0000_9020")
        self.assertEqual(top.area_id.iloc[-1], "cell_0049_9020")

    def test_observed_map_camera_is_constrained_to_nyc(self):
        config = json.loads(make_source_map(pd.DataFrame(), pd.DataFrame()).to_json())
        self.assertEqual(config["initialViewState"]["minZoom"], 9)
        self.assertAlmostEqual(config["initialViewState"]["latitude"], 40.715)
        self.assertAlmostEqual(config["initialViewState"]["longitude"], -73.96)


class DashboardTests(ExportTestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def test_production_shaped_exports_controls_and_downloads(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.slider), 0)
        self.assertEqual(app.metric[0].value, "0.133")
        self.assertTrue(all(not button.proto.disabled for button in app.get("download_button")))
        app.selectbox(key="forecast_week").select("2025-09-22").run()
        app.selectbox(key="prediction_model").select("persistence").run()
        app.checkbox(key="top_only").check().run()
        app.selectbox(key="selected_area").select("cell_1162_9015").run()
        app.radio(key="comparison_scope").set_value("All four weeks").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="selected_area").value, "cell_1162_9015")
        with patch.object(st, "download_button", wraps=st.download_button) as download:
            app.run()
        prediction_call = next(call for call in download.call_args_list
                               if call.args[0] == "Download displayed prediction rows")
        self.assertIn("persistence_2025-09-22_top50", prediction_call.args[2])
        exported = pd.read_csv(BytesIO(prediction_call.args[1]))
        self.assertEqual(set(exported.forecast_start), {"2025-09-22"})
        self.assertEqual(len(exported), 3)
        # Every measure/model/week should render without reusing old ranking controls.
        for model in data.MODELS:
            app.selectbox(key="prediction_model").select(model).run()
            self.assertFalse(app.exception)
        for week in data.WEEKS:
            app.selectbox(key="forecast_week").select(week).run()
            self.assertFalse(app.exception)

    def test_missing_outputs_and_explicit_fixture_mode(self):
        (self.directory / "predictions.csv").unlink()
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any("Model comparison needs" in element.value for element in app.caption))
        self.assertEqual(len(app.metric), 0)
        app.checkbox(key="fixture_mode").check().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("DEVELOPMENT EXAMPLE" in element.value for element in app.warning))
        self.assertTrue(all(button.proto.disabled for button in app.get("download_button")))
        app.checkbox(key="fixture_mode").uncheck().run()
        self.assertEqual(len(app.metric), 0)

    def test_stale_outputs_disable_downloads_and_label_saved_run(self):
        (self.directory / data.SOURCES[0]).write_bytes(b"new inputs")
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any("MODEL INPUT MISMATCH" in element.value for element in app.warning))
        self.assertTrue(all(button.proto.disabled for button in app.get("download_button")))

    def test_mixed_run_stops_rendering(self):
        self.update_meta(run_id="different-run")
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any("Mixed run" in element.value for element in app.caption))
        self.assertEqual(len(app.metric), 0)

    def test_observed_nyc_map_remains_visible_without_model_exports(self):
        (self.directory / "predictions.csv").unlink()
        for filename in data.SOURCES:
            shutil.copyfile(ROOT / "data" / "processed" / filename, self.directory / filename)
        with patch.object(st, "pydeck_chart", wraps=st.pydeck_chart) as chart:
            app = self.app()
        self.assertFalse(app.exception)
        config = json.loads(chart.call_args.args[0].to_json())
        self.assertEqual(len(config["layers"]), 2)
        self.assertGreater(len(config["layers"][0]["data"]), 0)
        self.assertAlmostEqual(config["initialViewState"]["longitude"], -73.96)
        app.selectbox(key="forecast_week").select("2025-09-15").run()
        app.checkbox(key="source_reports").uncheck().run()
        self.assertFalse(app.exception)

    def test_forecast_controls_disable_typing(self):
        app = self.app()
        for key in ("forecast_week", "prediction_model", "selected_area"):
            proto = app.selectbox(key=key).proto
            enum = proto.DESCRIPTOR.fields_by_name["filter_mode"].enum_type
            self.assertEqual(enum.values_by_number[proto.filter_mode].name, "FILTER_MODE_NONE")

    def test_observed_map_is_primary_and_forecast_map_is_removed(self):
        for filename in data.SOURCES:
            shutil.copyfile(ROOT / "data" / "processed" / filename, self.directory / filename)
        app = self.app()
        self.assertFalse(app.exception)
        self.assertEqual(app.tabs[0].label, "Observed map")
        self.assertFalse(any(tab.label == "Forecast map" for tab in app.tabs))
        self.assertFalse(any(box.key == "map_measure" for box in app.selectbox))
        self.assertTrue(any("Circle area represents recorded count" in text.value for text in app.caption))

    def test_source_map_detail_controls_and_descriptive_hover(self):
        (self.directory / "predictions.csv").unlink()
        for filename in data.SOURCES:
            shutil.copyfile(ROOT / "data" / "processed" / filename, self.directory / filename)
        app = self.app()
        app.radio(key="source_view").set_value("Individual events").run()
        self.assertFalse(app.exception)
        reports, crashes = data.source_events_for_week(data.WEEKS[0])
        config = json.loads(make_source_map(reports, crashes, "Individual events").to_json())
        report_tooltip = config["layers"][0]["data"][0]["event_detail"]
        crash_tooltip = config["layers"][1]["data"][0]["event_detail"]
        self.assertIn("Created:", report_tooltip)
        self.assertIn("Type: Traffic Signal Condition", report_tooltip)
        self.assertIn("Status (source snapshot):", report_tooltip)
        self.assertIn("Collision\nOccurred:", crash_tooltip)
        self.assertIn("Injured people:", crash_tooltip)
        app.checkbox(key="fixture_mode").check().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("NYC · observed" in title.value for title in app.subheader))
        self.assertEqual(app.radio(key="source_view").value, "Individual events")


    def test_top50_filter_resets_a_hidden_selection(self):
        bundle = data.load_forecast_bundle()
        frames = []
        for i in range(60):
            frame = bundle.predictions.loc[bundle.predictions.area_id.eq("cell_1160_9020")].copy()
            frame["area_id"] = f"cell_{i:04d}_9020"
            frame["combined_prediction"] = 1.0
            frames.append(frame)
        bundle.predictions = pd.concat(frames, ignore_index=True)
        bundle.metadata["eligible_area_count"] = 60
        with patch.object(data, "load_forecast_bundle", return_value=bundle):
            app = self.app()
            app.selectbox(key="selected_area").select("cell_0059_9020").run()
            app.checkbox(key="top_only").check().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="selected_area").value, "cell_0000_9020")


class ConcentrationMapTests(unittest.TestCase):
    def test_five_events_have_five_times_the_circle_area_of_one(self):
        reports = pd.DataFrame({
            "unique_key": [str(i) for i in range(6)],
            "latitude": [40.75] * 5 + [40.81], "longitude": [-73.99] * 5 + [-73.94],
            "created_date": ["2025-08-25T12:00:00"] * 6,
            "complaint_type": ["Traffic Signal Condition"] * 6,
            "descriptor": ["Controller"] * 6, "status": ["Open"] * 6,
        })
        config = json.loads(make_source_map(reports, pd.DataFrame()).to_json())
        markers = {row["display_count"]: row["marker_pixels"] for row in config["layers"][0]["data"]}
        self.assertAlmostEqual((markers[5] / markers[1]) ** 2, 5)
        self.assertAlmostEqual(markers[1], 6)
        self.assertEqual(config["layers"][0]["radiusMaxPixels"], 60)

    def test_display_bins_conserve_events_and_group_nearby_reports(self):
        reports = pd.DataFrame({
            "unique_key": ["1", "2", "3"],
            "latitude": [40.75, 40.75, 40.81], "longitude": [-73.99, -73.99, -73.94],
            "created_date": ["2025-08-25T12:00:00"] * 3,
            "complaint_type": ["Traffic Signal Condition"] * 3,
            "descriptor": ["Controller"] * 3, "status": ["Closed", "Open", "Open"],
        })
        grouped = source_map_frame(reports, "Signal report")
        self.assertEqual(len(grouped), 2)
        self.assertEqual(grouped.display_count.sum(), 3)
        self.assertEqual(sorted(grouped.display_count), [1, 2])
        self.assertIn("2 signal reports", grouped.loc[grouped.display_count.eq(2), "event_detail"].iloc[0])
        detail = source_map_frame(reports, "Signal report", "Individual events")
        self.assertEqual(len(detail), 3)
        deck = json.loads(make_source_map(reports, pd.DataFrame()).to_json())
        markers = deck["layers"][0]["data"]
        by_count = {marker["display_count"]: marker["marker_pixels"] for marker in markers}
        self.assertGreater(by_count[2], by_count[1])
        self.assertLessEqual(by_count[2], 60)



@unittest.skipUnless(all((data.DATA / filename).exists() for filename in data.EXPORTS),
                     "Canonical real backend exports are not present yet")
class CurrentRealExportTests(unittest.TestCase):
    def test_real_exports_match_contract_and_source_hashes(self):
        bundle = data.load_forecast_bundle()
        self.assertFalse(bundle.fixture)
        self.assertFalse(bundle.stale_reasons)
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
