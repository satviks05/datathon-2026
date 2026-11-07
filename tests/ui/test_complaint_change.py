"""Optional experiment display and provenance, without training inside UI."""

from pathlib import Path
import sys
import unittest

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ui.complaint_change import load_experiment


class ComplaintChangeUITests(unittest.TestCase):
    def test_real_experiment_schema_and_hashes(self):
        predictions, metrics, meta = load_experiment()
        self.assertEqual(len(predictions), 4720)
        self.assertEqual(len(metrics), 32)
        self.assertEqual(meta['train_starts'], ['2025-08-15', '2025-08-22'])
        self.assertTrue(predictions.complaint_change.lt(0).any())
        self.assertTrue(predictions.complaint_change.gt(0).any())

    def test_addon_tab_and_scope_switch(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertTrue(any(tab.label == 'Complaint change' for tab in app.tabs))
        app.radio(key='change_scope').set_value('Selected week').run()
        app.selectbox(key='forecast_week').select('2025-09-22').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('TWO August training windows' in warning.value for warning in app.warning))
        app.checkbox(key='fixture_mode').check().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('add-on is disabled' in info.value for info in app.info))
