"""Integration checks for UI states and the backend handoff."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ui import data


class DashboardTests(unittest.TestCase):
    def run_app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def test_real_source_startup_and_fixture_controls(self):
        # Isolate this state from any future backend exports in the checkout.
        def no_export(filename, required):
            return pd.DataFrame(columns=sorted(required)), f"Awaiting {filename}."

        with patch.object(data, "load_export", side_effect=no_export):
            app = self.run_app()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "3,276")
            self.assertEqual(app.metric[1].value, "3,400")
            self.assertEqual(app.metric[2].value, "—")
            self.assertTrue(any("No backend shortlist" in x.value for x in app.info))
            app.checkbox(key="fixture_mode").check().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("DEVELOPMENT FIXTURE" in x.value for x in app.warning))
            self.assertEqual(len(app.get("download_button")), 0)
            app.selectbox(key="selected_target").select("UI-EXAMPLE-C").run()
            app.slider(key="radius_m").set_value(100).run()
            app.selectbox(key="ranking_strategy").select("combined").run()
            app.selectbox(key="shortlist_budget").select(25).run()
            app.radio(key="map_period").set_value("September").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[3].value, "100 m")
            app.checkbox(key="fixture_mode").uncheck().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[2].value, "—")

    def test_real_exports_and_evidence_connection(self):
        def exports(filename, required):
            if filename == "selected_targets.csv":
                table = data.development_fixture(250, 50, "crash_first")
                table["target_id"] = ["REAL-A", "REAL-B", "REAL-C"]
                return table, None
            return pd.DataFrame([
                dict(strategy=strategy, radius_m=250, budget=50, selected_count=3,
                     september_harm_crashes_covered=10, september_harm_coverage_pct=0.294,
                     august_signal_reports_covered=20)
                for strategy in data.STRATEGIES
            ]), None

        with patch.object(data, "load_export", side_effect=exports), patch.object(
            data, "area_evidence", return_value={"reports": pd.DataFrame(), "collisions": pd.DataFrame()}
        ) as evidence:
            app = self.run_app()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[2].value, "3")
            self.assertEqual(len(app.get("download_button")), 2)
            self.assertEqual(evidence.call_args.args[:3], ("REAL-A", 250, "august"))
            app.radio(key="evidence_period").set_value("september").run()
            self.assertEqual(evidence.call_args.args[:3], ("REAL-A", 250, "september"))
            app.slider(key="radius_m").set_value(125).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[2].value, "—")
            self.assertEqual(len(app.get("download_button")), 0)

    def test_configuration_never_falls_back_to_another_radius(self):
        table = data.development_fixture(250, 50, "combined")
        self.assertTrue(data.select_configuration(table, 100, 50, "combined").empty)
        self.assertTrue(data.select_configuration(table, 250, 25, "combined").empty)

    def test_boolean_parsing_includes_fatal_only_crashes(self):
        reports, crashes = data.load_source_events()
        self.assertEqual(int(crashes["has_harm"].sum()), 6536)
        fatal_only = crashes["has_fatality"] & ~crashes["has_injury"]
        self.assertEqual(int(fatal_only.sum()), 32)
        self.assertTrue(crashes.loc[fatal_only, "has_harm"].all())


if __name__ == "__main__":
    unittest.main()
