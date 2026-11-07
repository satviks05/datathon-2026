"""Ordinary least-squares prediction of next-seven-day harm-crash counts.

Fit only on August-complete labels. September is evaluated, never fitted.
NumPy's least-squares solver fits the intercept and two coefficients directly.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .spatial import DISTANCE_TOLERANCE_M

FEATURE_NAMES = ("signal_reports_per_7_days", "harm_crashes_per_7_days")
PREDICTION_COLUMN = "predicted_harm_crashes_next_7d"


def fit_ols(features, target):
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if features.ndim != 2 or len(features) != len(target) or not len(features):
        raise ValueError("OLS requires nonempty aligned two-dimensional features and targets")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("OLS training values must be finite")
    design = np.column_stack([np.ones(len(features)), features])
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients, int(rank)


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error ** 2))),
                r2=float(1 - np.sum(error ** 2) / total) if total > 0 else None)


class SevenDayRegression:
    def __init__(self, engine):
        self.engine = engine
        self.days = pd.date_range(config.MIN_DATE, config.MAX_DATE, freq="D")
        origin = np.datetime64(config.MIN_DATE, "D")
        self.report_days = (engine.report_times.astype("datetime64[D]") - origin).astype(int)
        self.crash_days = (engine.crash_times.astype("datetime64[D]") - origin).astype(int)

    @lru_cache(maxsize=7)
    def daily_counts(self, radius_m):
        shape = (len(self.engine.candidates), len(self.days))
        report_counts, harm_counts = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
        for index in self.engine.candidates.index:
            indices, distances = self.engine.report_neighbors[index]
            selected = indices[distances <= radius_m + DISTANCE_TOLERANCE_M]
            report_counts[index] = np.bincount(self.report_days[selected], minlength=len(self.days))
            indices, distances = self.engine.crash_neighbors[index]
            selected = indices[(distances <= radius_m + DISTANCE_TOLERANCE_M) & self.engine.harm[indices]]
            harm_counts[index] = np.bincount(self.crash_days[selected], minlength=len(self.days))
        return report_counts, harm_counts

    @lru_cache(maxsize=7)
    def fit(self, radius_m):
        reports, harm = self.daily_counts(radius_m)
        # Prefixes retain record counts, not complaint-crash pair counts.
        report_prefix = np.pad(reports.cumsum(axis=1), ((0, 0), (1, 0)))
        harm_prefix = np.pad(harm.cumsum(axis=1), ((0, 0), (1, 0)))
        august_last_index = (pd.Timestamp(config.MODEL_TRAIN_END) - self.days[0]).days
        features, targets, window_count = [], [], 0
        for lookback in config.TRAIN_LOOKBACK_DAYS:
            for end in range(lookback - 1, august_last_index - config.FORECAST_DAYS + 1):
                begin = end - lookback + 1
                reports_before = report_prefix[:, end + 1] - report_prefix[:, begin]
                harm_before = harm_prefix[:, end + 1] - harm_prefix[:, begin]
                future_harm = harm_prefix[:, end + config.FORECAST_DAYS + 1] - harm_prefix[:, end + 1]
                features.append(np.column_stack([reports_before, harm_before]) * config.FORECAST_DAYS / lookback)
                targets.append(future_harm)
                window_count += 1
        x, y = np.concatenate(features), np.concatenate(targets)
        coefficients, rank = fit_ols(x, y)
        return dict(
            coefficients=coefficients, design_rank=rank, training_rows=len(y),
            training_window_count=window_count, training_target_mean=float(y.mean()),
            training_features_start_date=config.MIN_DATE, training_labels_end_date=config.MODEL_TRAIN_END,
            training_last_feature_end_date=(pd.Timestamp(config.MODEL_TRAIN_END) - pd.Timedelta(days=config.FORECAST_DAYS)).date().isoformat(),
            feature_names=list(FEATURE_NAMES), lookback_days=list(config.TRAIN_LOOKBACK_DAYS),
            forecast_days=config.FORECAST_DAYS,
        )

    def predict(self, features, radius_m):
        model = self.fit(radius_m)
        if features.empty:
            return np.empty(0), np.empty((0, len(FEATURE_NAMES)))
        days = (pd.Timestamp(features.iloc[0]["end_date"]) - pd.Timestamp(features.iloc[0]["start_date"])).days + 1
        x = features[["signal_report_count", "harm_crash_count"]].to_numpy(dtype=float) * config.FORECAST_DAYS / days
        raw = model["coefficients"][0] + x @ model["coefficients"][1:]
        return raw, x

    def description(self, radius_m):
        model = self.fit(radius_m)
        return {**{key: value for key, value in model.items() if key != "coefficients"},
                "intercept": float(model["coefficients"][0]),
                "coefficients": {name: float(value) for name, value in zip(FEATURE_NAMES, model["coefficients"][1:])},
                "estimator": "ordinary least squares with intercept (numpy.linalg.lstsq)",
                "target": "injury OR fatal crashes in the next seven days within the selected radius",
                "postprocessing": "max(raw_prediction, 0); raw predictions also retained",
                "uncertainty": "No calibrated probabilities or confidence intervals; spatial/temporal samples overlap"}

    @lru_cache(maxsize=7)
    def weekly_predictions(self, radius_m):
        rows = []
        reports, harm = self.daily_counts(radius_m)
        train_mean = self.fit(radius_m)["training_target_mean"]
        # Four complete, nonoverlapping September outcome weeks; Sep 29-30 omitted.
        for end_date in ("2025-08-31", "2025-09-07", "2025-09-14", "2025-09-21"):
            end = pd.Timestamp(end_date)
            start = end - pd.Timedelta(days=6)
            end_index = (end - self.days[0]).days
            features = self.engine.features(start.date().isoformat(), end.date().isoformat(), radius_m)
            raw, x = self.predict(features, radius_m)
            actual = harm[:, end_index + 1:end_index + config.FORECAST_DAYS + 1].sum(axis=1)
            frame = self.engine.candidates.copy()
            frame["radius_m"] = radius_m
            frame["history_start_date"] = start.date().isoformat()
            frame["history_end_date"] = end.date().isoformat()
            frame["forecast_start_date"] = (end + pd.Timedelta(days=1)).date().isoformat()
            frame["forecast_end_date"] = (end + pd.Timedelta(days=config.FORECAST_DAYS)).date().isoformat()
            frame["actual_harm_crashes_next_7d"] = actual
            frame["raw_prediction"] = raw
            frame[PREDICTION_COLUMN] = np.maximum(raw, 0)
            frame["historical_rate_baseline"] = x[:, 1]
            frame["training_mean_baseline"] = train_mean
            frame["zero_baseline"] = 0.0
            for index, name in enumerate(FEATURE_NAMES):
                frame[name] = x[:, index]
            rows.append(frame)
        return pd.concat(rows, ignore_index=True)

    @lru_cache(maxsize=7)
    def test_metrics(self, radius_m):
        predictions = self.weekly_predictions(radius_m)
        rows = []
        for label, frame in [("all_complete_september_weeks", predictions)] + list(predictions.groupby("forecast_start_date")):
            actual = frame["actual_harm_crashes_next_7d"].to_numpy()
            for estimator, column in (("linear_regression", PREDICTION_COLUMN),
                                      ("historical_rate", "historical_rate_baseline"),
                                      ("training_mean", "training_mean_baseline"),
                                      ("always_zero", "zero_baseline")):
                rows.append(dict(radius_m=radius_m, estimator=estimator, test_period=label,
                                 test_rows=len(frame), **regression_metrics(actual, frame[column].to_numpy()),
                                 raw_negative_prediction_pct=float(100 * frame["raw_prediction"].lt(0).mean()) if estimator == "linear_regression" else 0.0,
                                 **({f"raw_{key}": value for key, value in regression_metrics(actual, frame["raw_prediction"].to_numpy()).items()} if estimator == "linear_regression" else {})))
        return pd.DataFrame(rows)
