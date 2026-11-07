"""Reusable analysis engine: spatial memberships once, date/radius queries on demand."""

from functools import lru_cache
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .loaders import load_events
from .ranking import score_candidates, select_spaced
from .regression import SevenDayRegression
from .spatial import DISTANCE_TOLERANCE_M, RadiusIndex, build_candidates

CONTEXT_COLUMNS = ["persons_injured", "persons_killed", "cyclist_injured"]
FORECAST_COLUMNS = ["raw_prediction", "predicted_harm_crashes_next_7d",
                    "signal_reports_per_7_days", "harm_crashes_per_7_days",
                    "forecast_start_date", "forecast_end_date"]


def date_window(start_date, end_date):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    for value in (start, end):
        if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
            raise ValueError("Use timezone-free calendar dates, at daily granularity")
    if start > end or start < pd.Timestamp(config.MIN_DATE) or end > pd.Timestamp(config.MAX_DATE):
        raise ValueError(f"Dates must be ordered and inside {config.MIN_DATE} through {config.MAX_DATE}")
    return start, end + pd.Timedelta(days=1)


def validate_controls(radius_m, budget=None, strategy=None):
    if radius_m not in config.RADII:
        raise ValueError(f"radius_m must be one of {config.RADII}")
    if budget is not None and budget not in config.BUDGETS:
        raise ValueError(f"budget must be one of {config.BUDGETS}")
    if strategy is not None and strategy not in config.STRATEGIES:
        raise ValueError(f"strategy must be one of {config.STRATEGIES}")


def inspection_for(descriptor):
    value = descriptor.casefold()
    if "pedestrian" in value:
        return "Inspect the reported pedestrian-signal equipment."
    if "controller" in value or "cabinet" in value:
        return "Inspect the reported signal-control equipment."
    if "veh" in value and ("signal" in value or "lamp" in value):
        return "Inspect the reported vehicle-signal equipment."
    if not descriptor:
        return "Review the historically reported signal area; no signal reports in the selected dates."
    return f"Inspect the reported component; reported descriptor: {descriptor}."


class AnalysisEngine:
    def __init__(self, data_dir, version):
        self.data_dir = data_dir
        self.version = version
        self.reports, self.collisions = load_events(data_dir)
        self.candidates = build_candidates(self.reports)
        self.target_lookup = dict(zip(self.candidates["target_id"], self.candidates.index))
        self.report_times = self.reports["created_date"].to_numpy()
        self.crash_times = self.collisions["occurred_at"].to_numpy()
        self.harm = self.collisions["has_harm"].to_numpy()
        self.descriptors = self.reports["descriptor"].fillna("").astype(str).to_numpy()
        self.injury_context = self.collisions[[
            "number_of_persons_injured", "number_of_persons_killed", "number_of_cyclist_injured",
        ]].to_numpy(dtype=np.int64)
        self.report_neighbors = self._neighborhoods(self.reports)
        self.crash_neighbors = self._neighborhoods(self.collisions)
        spacing_index = RadiusIndex(self.candidates[["latitude", "longitude"]].to_numpy())
        self.blocked_neighbors = []
        for row in self.candidates.itertuples():
            indices, distances = spacing_index.query(row.latitude, row.longitude, config.SPACING_M)
            self.blocked_neighbors.append(indices[distances < config.SPACING_M - DISTANCE_TOLERANCE_M])
        self.regression = SevenDayRegression(self)

    def _neighborhoods(self, events):
        index = RadiusIndex(events[["latitude", "longitude"]].to_numpy())
        return [index.query(row.latitude, row.longitude, max(config.RADII))
                for row in self.candidates.itertuples()]

    def _time_masks(self, start_date, end_date):
        start, stop = date_window(start_date, end_date)
        start, stop = start.to_datetime64(), stop.to_datetime64()
        return ((self.report_times >= start) & (self.report_times < stop),
                (self.crash_times >= start) & (self.crash_times < stop))

    @staticmethod
    def _member_indices(neighbors, index, radius_m, time_mask):
        indices, distances = neighbors[index]
        keep = (distances <= radius_m + DISTANCE_TOLERANCE_M) & time_mask[indices]
        return indices[keep]

    @lru_cache(maxsize=16)
    def features(self, start_date, end_date, radius_m):
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        features = self.candidates.copy()
        report_counts, crash_counts, descriptors, context = [], [], [], []
        for index in self.candidates.index:
            reports = self._member_indices(self.report_neighbors, index, radius_m, report_mask)
            crashes = self._member_indices(self.crash_neighbors, index, radius_m, crash_mask)
            report_counts.append(len(reports))
            crash_counts.append(int(self.harm[crashes].sum()))
            counts = Counter(value for value in self.descriptors[reports] if value)
            top = min(counts, key=lambda value: (-counts[value], value)) if counts else ""
            descriptors.append(top)
            context.append(self.injury_context[crashes].sum(axis=0))
        features["signal_report_count"] = np.asarray(report_counts, dtype=np.int64)
        features["harm_crash_count"] = np.asarray(crash_counts, dtype=np.int64)
        features["top_descriptor"] = descriptors
        features["suggested_inspection"] = [inspection_for(value) for value in descriptors]
        for i, column in enumerate(CONTEXT_COLUMNS):
            features[column] = np.asarray(context, dtype=np.int64).reshape(-1, 3)[:, i]
        features["radius_m"] = radius_m
        features["start_date"] = start_date
        features["end_date"] = end_date
        return features

    def scored_and_selected(self, start_date, end_date, radius_m, strategy, budget):
        features = self.features(start_date, end_date, radius_m).copy()
        available = pd.Timestamp(end_date) >= pd.Timestamp(config.MODEL_TRAIN_END)
        if available and not features.empty:
            raw, x = self.regression.predict(features, radius_m)
            features["raw_prediction"] = raw
            features["predicted_harm_crashes_next_7d"] = np.maximum(raw, 0)
            features["signal_reports_per_7_days"] = x[:, 0]
            features["harm_crashes_per_7_days"] = x[:, 1]
        else:
            for column in FORECAST_COLUMNS[:4]:
                features[column] = np.nan
        features["forecast_start_date"] = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat()
        features["forecast_end_date"] = (pd.Timestamp(end_date) + pd.Timedelta(days=config.FORECAST_DAYS)).date().isoformat()
        scored = score_candidates(features, strategy)
        # Empty date windows should not manufacture a review shortlist.
        if (strategy == "linear_regression" and not available) or features[["signal_report_count", "harm_crash_count"]].to_numpy().sum() == 0:
            selected = scored.iloc[:0].copy()
            selected["selected_rank"] = pd.Series(dtype="int64")
            selected["budget"] = pd.Series(dtype="int64")
        else:
            selected = select_spaced(scored, self.blocked_neighbors, budget)
        return (scored[config.SCORED_COLUMNS + CONTEXT_COLUMNS + FORECAST_COLUMNS],
                selected[config.SELECTED_COLUMNS + CONTEXT_COLUMNS + FORECAST_COLUMNS])

    def coverage(self, selected, radius_m, start_date, end_date):
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        report_union = np.zeros(len(self.reports), dtype=bool)
        crash_union = np.zeros(len(self.collisions), dtype=bool)
        for target_id in selected["target_id"]:
            index = self.target_lookup[target_id]
            report_union[self._member_indices(self.report_neighbors, index, radius_m, report_mask)] = True
            crash_union[self._member_indices(self.crash_neighbors, index, radius_m, crash_mask)] = True
        return int(report_union.sum()), int((crash_union & self.harm).sum())

    @lru_cache(maxsize=21)
    def historical_evaluation(self, radius_m, budget):
        rows = []
        for end_date in ("2025-08-31", "2025-09-07", "2025-09-14", "2025-09-21"):
            end = pd.Timestamp(end_date)
            start_date = (end - pd.Timedelta(days=6)).date().isoformat()
            future_start = (end + pd.Timedelta(days=1)).date().isoformat()
            future_end = (end + pd.Timedelta(days=config.FORECAST_DAYS)).date().isoformat()
            _, future_mask = self._time_masks(future_start, future_end)
            denominator = int((future_mask & self.harm).sum())
            for strategy in config.STRATEGIES:
                _, selected = self.scored_and_selected(start_date, end_date, radius_m, strategy, budget)
                history_reports, _ = self.coverage(selected, radius_m, start_date, end_date)
                _, future_harm = self.coverage(selected, radius_m, future_start, future_end)
                rows.append(dict(
                    strategy=strategy, radius_m=radius_m, budget=budget, selected_count=len(selected),
                    future_harm_crashes_covered=future_harm,
                    future_harm_coverage_pct=100 * future_harm / denominator if denominator else 0.0,
                    history_signal_reports_covered=history_reports, future_harm_crashes_total=denominator,
                    start_date=start_date, end_date=end_date,
                    forecast_start_date=future_start, forecast_end_date=future_end,
                ))
        return pd.DataFrame(rows)

    @lru_cache(maxsize=16)
    def analysis(self, start_date, end_date, radius_m, strategy, budget):
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        future_start = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date().isoformat()
        future_end = (pd.Timestamp(end_date) + pd.Timedelta(days=config.FORECAST_DAYS)).date().isoformat()
        model_available = pd.Timestamp(end_date) >= pd.Timestamp(config.MODEL_TRAIN_END)
        outcomes_available = model_available and future_end <= config.MAX_DATE
        denominator = None
        if outcomes_available:
            _, future_mask = self._time_masks(future_start, future_end)
            denominator = int((future_mask & self.harm).sum())
        comparisons, current_scored, current_selected = [], None, None
        for option in config.STRATEGIES:
            scored, selected = self.scored_and_selected(start_date, end_date, radius_m, option, budget)
            reports, _ = self.coverage(selected, radius_m, start_date, end_date)
            crashes = self.coverage(selected, radius_m, future_start, future_end)[1] if outcomes_available else None
            comparisons.append(dict(
                strategy=option, radius_m=radius_m, budget=budget, selected_count=len(selected),
                history_signal_reports_covered=reports, future_harm_crashes_covered=crashes,
                future_harm_crashes_total=denominator,
                future_harm_coverage_pct=(100 * crashes / denominator if denominator else 0.0) if outcomes_available else None,
                start_date=start_date, end_date=end_date,
                forecast_start_date=future_start, forecast_end_date=future_end,
            ))
            if option == strategy:
                current_scored, current_selected = scored, selected
        return {
            "targets_scored": current_scored.reset_index(drop=True),
            "selected_targets": current_selected.reset_index(drop=True),
            "comparison": pd.DataFrame(comparisons),
            "reports": self.reports.loc[report_mask].copy(),
            "collisions": self.collisions.loc[crash_mask].copy(),
            "evaluation": self.historical_evaluation(radius_m, budget).copy(),
            "model": self.regression.description(radius_m) if model_available else None,
            "metadata": dict(
                start_date=start_date, end_date=end_date, radius_m=radius_m,
                strategy=strategy, budget=budget, input_version=self.version,
                method_version=config.METHOD_VERSION, date_granularity="day",
                timezone="America/New_York (source local-clock assumption)",
                signal_reports_total=int(report_mask.sum()), harm_crashes_total=int((crash_mask & self.harm).sum()),
                candidate_count=len(self.candidates), candidate_period="August 2025 (fixed)",
                analysis_kind="seven-day count forecast with retrospective evidence",
                forecast_start_date=future_start, forecast_end_date=future_end,
                model_available=model_available, outcomes_available=outcomes_available,
                model_note=("Fitted on August-complete labels; coefficients frozen throughout September." if model_available else
                            "Model unavailable before August 31: avoid using future training labels. Observed event layers remain available."),
                history_days=(pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1,
                training_lookback_days=list(config.TRAIN_LOOKBACK_DAYS),
                history_note="Counts normalized to seven-day equivalents; model trained on 7/14/21-day lookbacks. Other history lengths are extrapolations.",
                comparison_kind="unique harm crashes in the next seven days, when fully observed",
                evaluation_note="Separate fixed weekly September backtest; four complete nonoverlapping outcome weeks. September was previously explored.",
            ),
        }

    def evidence(self, target_id, radius_m, start_date, end_date):
        if target_id not in self.target_lookup:
            raise ValueError(f"Unknown target_id: {target_id}")
        index = self.target_lookup[target_id]
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        result = {}
        for name, table, neighbors, mask in (
            ("reports", self.reports, self.report_neighbors, report_mask),
            ("collisions", self.collisions, self.crash_neighbors, crash_mask),
        ):
            indices, distances = neighbors[index]
            keep = (distances <= radius_m + DISTANCE_TOLERANCE_M) & mask[indices]
            records = table.iloc[indices[keep]].copy()
            records["distance_m"] = distances[keep]
            result[name] = records.reset_index(drop=True)
        return result
