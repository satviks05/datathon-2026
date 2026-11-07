"""Three linear count forecasts for September, with August-only inputs.

The model class accepts NO September records. Later September lag features use
its own previous predictions, never September observations (recursive forecast).
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .spatial import DISTANCE_TOLERANCE_M, RadiusIndex

OUTCOMES = ("signal_reports", "collisions", "harm_crashes")
FEATURE_NAMES = ("previous_7d_signal_reports_per_day", "previous_7d_collisions_per_day",
                 "previous_7d_harm_crashes_per_day", "forecast_day_is_weekend")
PREDICTION_COLUMN = "predicted_harm_crashes"


def fit_ols(features, target):
    features, target = np.asarray(features, dtype=float), np.asarray(target, dtype=float)
    if features.ndim != 2 or target.ndim not in (1, 2) or len(features) != len(target) or not len(features):
        raise ValueError("OLS requires nonempty aligned features and targets")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("OLS training values must be finite")
    coefficients, _, rank, _ = np.linalg.lstsq(np.column_stack([np.ones(len(features)), features]), target, rcond=None)
    return coefficients, int(rank)


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    error = predicted - actual
    total = float(np.sum((actual - actual.mean()) ** 2))
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error ** 2))),
                r2=float(1 - np.sum(error ** 2) / total) if total > 0 else None)


class AugustOnlyRegression:
    def __init__(self, reports, collisions, candidates):
        cutoff = pd.Timestamp(config.FORECAST_START)
        if not reports["created_date"].lt(cutoff).all() or not collisions["occurred_at"].lt(cutoff).all():
            raise ValueError("Training model accepts August data ONLY; September is comparison data")
        if candidates.empty:
            raise ValueError("At least one August signal-report candidate is required")
        self.candidates = candidates.copy()
        self.days = pd.date_range(config.MIN_DATE, config.MODEL_TRAIN_END)
        self.forecast_dates = pd.date_range(config.FORECAST_START, config.FORECAST_END)
        origin = np.datetime64(config.MIN_DATE, "D")
        self.report_days = (reports["created_date"].to_numpy().astype("datetime64[D]") - origin).astype(int)
        self.crash_days = (collisions["occurred_at"].to_numpy().astype("datetime64[D]") - origin).astype(int)
        self.harm = collisions["has_harm"].to_numpy()
        self.report_neighbors = self._neighbors(reports)
        self.crash_neighbors = self._neighbors(collisions)

    def _neighbors(self, events):
        index = RadiusIndex(events[["latitude", "longitude"]].to_numpy())
        return [index.query(row.latitude, row.longitude, max(config.RADII)) for row in self.candidates.itertuples()]

    @lru_cache(maxsize=7)
    def daily_counts(self, radius_m):
        counts = np.zeros((len(self.candidates), len(self.days), len(OUTCOMES)), dtype=np.int64)
        for index in self.candidates.index:
            ids, distances = self.report_neighbors[index]
            keep = ids[distances <= radius_m + DISTANCE_TOLERANCE_M]
            counts[index, :, 0] = np.bincount(self.report_days[keep], minlength=len(self.days))
            ids, distances = self.crash_neighbors[index]
            keep = ids[distances <= radius_m + DISTANCE_TOLERANCE_M]
            counts[index, :, 1] = np.bincount(self.crash_days[keep], minlength=len(self.days))
            keep = keep[self.harm[keep]]
            counts[index, :, 2] = np.bincount(self.crash_days[keep], minlength=len(self.days))
        return counts

    @lru_cache(maxsize=7)
    def fit(self, radius_m):
        counts = self.daily_counts(radius_m)
        features, labels = [], []
        for day in range(config.LAG_DAYS, len(self.days)):
            previous = counts[:, day - config.LAG_DAYS:day, :].mean(axis=1)
            weekend = np.full((len(counts), 1), float(self.days[day].dayofweek >= 5))
            features.append(np.column_stack([previous, weekend]))
            labels.append(counts[:, day, :])
        x, y = np.concatenate(features), np.concatenate(labels)
        coefficients, rank = fit_ols(x, y)
        return dict(coefficients=coefficients, design_rank=rank, training_rows=len(y),
                    training_window_count=len(self.days) - config.LAG_DAYS,
                    training_target_mean=y.mean(axis=0))

    @lru_cache(maxsize=7)
    def forecast(self, radius_m):
        model = self.fit(radius_m)
        august = self.daily_counts(radius_m)
        series = np.zeros((len(august), len(self.days) + len(self.forecast_dates), len(OUTCOMES)))
        series[:, :len(self.days), :] = august
        raw_forecasts = np.empty((len(august), len(self.forecast_dates), len(OUTCOMES)))
        for step, date in enumerate(self.forecast_dates):
            day = len(self.days) + step
            previous = series[:, day - config.LAG_DAYS:day, :].mean(axis=1)
            weekend = np.full((len(series), 1), float(date.dayofweek >= 5))
            x = np.column_stack([previous, weekend])
            raw = model["coefficients"][0] + x @ model["coefficients"][1:]
            prediction = np.maximum(raw, 0)
            prediction[:, 2] = np.minimum(prediction[:, 2], prediction[:, 1])
            series[:, day, :] = prediction
            raw_forecasts[:, step, :] = raw
        return series[:, len(self.days):, :], raw_forecasts

    def aggregate(self, start_date, end_date, radius_m):
        start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
        if start < self.forecast_dates[0] or end > self.forecast_dates[-1] or start > end:
            raise ValueError("Frozen forecast dates must be inside September 2025")
        mask = (self.forecast_dates >= start) & (self.forecast_dates <= end)
        predictions, raw = self.forecast(radius_m)
        totals, raw_totals = predictions[:, mask, :].sum(axis=1), raw[:, mask, :].sum(axis=1)
        august = self.daily_counts(radius_m)
        frame = self.candidates.copy()
        for index, outcome in enumerate(OUTCOMES):
            frame[f"predicted_{outcome}"] = totals[:, index]
            frame[f"raw_predicted_{outcome}"] = raw_totals[:, index]
            frame[f"august_rate_{outcome}"] = august[:, :, index].mean(axis=1) * mask.sum()
            frame[f"last_august_week_rate_{outcome}"] = august[:, -config.LAG_DAYS:, index].mean(axis=1) * mask.sum()
        frame["radius_m"] = radius_m
        frame["forecast_start_date"] = start.date().isoformat()
        frame["forecast_end_date"] = end.date().isoformat()
        frame["training_end_date"] = config.MODEL_TRAIN_END
        return frame

    def daily_forecasts(self, radius_m):
        predictions, raw = self.forecast(radius_m)
        frame = self.candidates.loc[self.candidates.index.repeat(len(self.forecast_dates))].reset_index(drop=True)
        frame["date"] = np.tile(self.forecast_dates.strftime("%Y-%m-%d"), len(self.candidates))
        frame["radius_m"] = radius_m
        for index, outcome in enumerate(OUTCOMES):
            frame[f"predicted_{outcome}"] = predictions[:, :, index].reshape(-1)
            frame[f"raw_predicted_{outcome}"] = raw[:, :, index].reshape(-1)
        return frame

    def description(self, radius_m):
        model = self.fit(radius_m)
        return dict(
            estimator="ordinary least squares with intercept (numpy.linalg.lstsq)",
            targets=list(OUTCOMES), feature_names=list(FEATURE_NAMES),
            models={outcome: dict(intercept=float(model["coefficients"][0, index]),
                                 coefficients={name: float(value) for name, value in zip(FEATURE_NAMES, model["coefficients"][1:, index])})
                    for index, outcome in enumerate(OUTCOMES)},
            training_rows=model["training_rows"], training_window_count=model["training_window_count"],
            design_rank=model["design_rank"], training_start_date=config.MIN_DATE,
            training_labels_end_date=config.MODEL_TRAIN_END, lag_days=config.LAG_DAYS,
            forecast_start_date=config.FORECAST_START, forecast_end_date=config.FORECAST_END,
            forecast_mode="all September dates frozen using August only; recursive predicted lags",
            september_observations_used=False,
            postprocessing="Daily predictions clipped to zero; harm predictions capped at all-collision predictions; raw estimates retained",
            limitations="Sparse counts, overlapping areas, recursive error accumulation; no calibrated probabilities or confidence intervals",
        )
