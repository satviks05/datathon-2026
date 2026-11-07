"""Small meaningful checks for the independent complaint-change experiment."""

import unittest

import numpy as np

from experiments.complaint_change.build import build_panel, fit_and_predict, signed_count_transform
from src.forecast_features import load_forecast_events


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports, cls.crashes = load_forecast_events()
        cls.panel = build_panel(cls.reports, cls.crashes)

    def test_complete_history_and_fixed_universe(self):
        train = self.panel.loc[self.panel.split.eq('train')]
        self.assertEqual(set(train.forecast_start), {'2025-08-15', '2025-08-22'})
        self.assertTrue(self.panel.previous_feature_start.ge('2025-08-01').all())
        self.assertEqual(len(self.panel), 1180 * 6)
        self.assertTrue(self.panel.previous_feature_end.eq(self.panel.feature_start).all())
        self.assertTrue(self.panel.feature_end.eq(self.panel.forecast_start).all())
        self.assertTrue(self.panel.actual_collisions.eq(0).any())

    def test_two_real_cell_windows_against_original_event_ids(self):
        for row in [self.panel.loc[self.panel.complaint_change.idxmax()],
                    self.panel.loc[self.panel.complaint_change.idxmin()]]:
            source = self.reports.loc[self.reports.area_id.eq(row.area_id)]
            previous = source.loc[source.timestamp.ge(row.previous_feature_start)
                                  & source.timestamp.lt(row.previous_feature_end), 'unique_key'].nunique()
            recent = source.loc[source.timestamp.ge(row.feature_start)
                                & source.timestamp.lt(row.feature_end), 'unique_key'].nunique()
            self.assertEqual(previous, row.previous_complaints)
            self.assertEqual(recent, row.past_complaints)
            self.assertEqual(recent - previous, row.complaint_change)

    def test_signed_transform_preserves_negative_changes(self):
        actual = signed_count_transform(np.array([[-5.0, 0, 5]]))
        np.testing.assert_allclose(actual, [[-np.log(6), 0, np.log(6)]])

    def test_evaluation_labels_never_enter_fit_and_predictions_reproduce(self):
        first, settings = fit_and_predict(self.panel, 'test')
        changed = self.panel.copy()
        changed.loc[changed.split.eq('evaluation'), 'actual_collisions'] = 999
        second, other_settings = fit_and_predict(changed, 'test')
        columns = [name for name in first if name.endswith('_prediction')]
        np.testing.assert_array_equal(first[columns], second[columns])
        self.assertEqual(settings, other_settings)
        self.assertTrue(np.isfinite(first[columns]).all().all())
        self.assertTrue(first[columns].ge(0).all().all())
        train = self.panel.loc[self.panel.split.eq('train')]
        expected_mean = signed_count_transform(train[['past_collisions', 'past_complaints', 'complaint_change']]).mean(axis=0)
        np.testing.assert_allclose(settings['complaint_change_2w']['scaler_mean'], expected_mean)
