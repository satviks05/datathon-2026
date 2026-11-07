"""Join frozen August-only September forecasts to separately observed events."""

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .loaders import load_events
from .ranking import score_candidates, select_spaced
from .regression import AugustOnlyRegression, OUTCOMES, regression_metrics
from .spatial import DISTANCE_TOLERANCE_M, RadiusIndex, build_candidates

CONTEXT_COLUMNS = ["persons_injured", "persons_killed", "cyclist_injured"]
FORECAST_COLUMNS = ([f"predicted_{name}" for name in OUTCOMES]
                    + [f"raw_predicted_{name}" for name in OUTCOMES]
                    + [f"august_rate_{name}" for name in OUTCOMES]
                    + [f"last_august_week_rate_{name}" for name in OUTCOMES]
                    + ["forecast_start_date", "forecast_end_date", "training_end_date"])


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
    return f"Inspect the reported component; reported descriptor: {descriptor}." if descriptor else "Review the historically reported signal area."


class AnalysisEngine:
    def __init__(self, data_dir, version):
        self.data_dir, self.version = data_dir, version
        self.reports, self.collisions = load_events(data_dir)
        self.candidates = build_candidates(self.reports)
        self.target_lookup = dict(zip(self.candidates["target_id"], self.candidates.index))
        self.report_times = self.reports["created_date"].to_numpy()
        self.crash_times = self.collisions["occurred_at"].to_numpy()
        self.harm = self.collisions["has_harm"].to_numpy()
        self.descriptors = self.reports["descriptor"].fillna("").astype(str).to_numpy()
        self.injury_context = self.collisions[["number_of_persons_injured", "number_of_persons_killed", "number_of_cyclist_injured"]].to_numpy(dtype=np.int64)
        self.report_neighbors = self._neighborhoods(self.reports)
        self.crash_neighbors = self._neighborhoods(self.collisions)
        index = RadiusIndex(self.candidates[["latitude", "longitude"]].to_numpy())
        self.blocked_neighbors = []
        for row in self.candidates.itertuples():
            ids, distances = index.query(row.latitude, row.longitude, config.SPACING_M)
            self.blocked_neighbors.append(ids[distances < config.SPACING_M - DISTANCE_TOLERANCE_M])
        # A hard boundary: the model receives ONLY August rows. Its constructor
        # rejects September rows, and never receives the evaluation engine.
        august_reports = self.reports.loc[self.reports.created_date.lt(config.FORECAST_START)].copy().reset_index(drop=True)
        august_crashes = self.collisions.loc[self.collisions.occurred_at.lt(config.FORECAST_START)].copy().reset_index(drop=True)
        self.regression = AugustOnlyRegression(august_reports, august_crashes, self.candidates)
        self.has_september_actuals = bool(self.reports.created_date.ge(config.FORECAST_START).any()
                                         or self.collisions.occurred_at.ge(config.FORECAST_START).any())

    def _neighborhoods(self, events):
        index = RadiusIndex(events[["latitude", "longitude"]].to_numpy())
        return [index.query(row.latitude, row.longitude, max(config.RADII)) for row in self.candidates.itertuples()]

    def _time_masks(self, start_date, end_date):
        start, stop = date_window(start_date, end_date)
        start, stop = start.to_datetime64(), stop.to_datetime64()
        return ((self.report_times >= start) & (self.report_times < stop),
                (self.crash_times >= start) & (self.crash_times < stop))

    @staticmethod
    def _member_indices(neighbors, index, radius_m, time_mask):
        ids, distances = neighbors[index]
        return ids[(distances <= radius_m + DISTANCE_TOLERANCE_M) & time_mask[ids]]

    @lru_cache(maxsize=16)
    def features(self, start_date, end_date, radius_m):
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        frame = self.candidates.copy()
        reports_count, crashes_count, harm_count, descriptors, context = [], [], [], [], []
        for index in self.candidates.index:
            reports = self._member_indices(self.report_neighbors, index, radius_m, report_mask)
            crashes = self._member_indices(self.crash_neighbors, index, radius_m, crash_mask)
            reports_count.append(len(reports))
            crashes_count.append(len(crashes))
            harm_count.append(int(self.harm[crashes].sum()))
            counts = Counter(value for value in self.descriptors[reports] if value)
            descriptors.append(min(counts, key=lambda value: (-counts[value], value)) if counts else "")
            context.append(self.injury_context[crashes].sum(axis=0))
        frame["signal_report_count"] = np.asarray(reports_count, dtype=np.int64)
        frame["collision_count"] = np.asarray(crashes_count, dtype=np.int64)
        frame["harm_crash_count"] = np.asarray(harm_count, dtype=np.int64)
        frame["top_descriptor"] = descriptors
        frame["suggested_inspection"] = [inspection_for(value) for value in descriptors]
        for i, column in enumerate(CONTEXT_COLUMNS):
            frame[column] = np.asarray(context, dtype=np.int64).reshape(-1, 3)[:, i]
        frame["radius_m"], frame["start_date"], frame["end_date"] = radius_m, start_date, end_date
        return frame

    def scored_and_selected(self, start_date, end_date, radius_m, strategy, budget):
        features = self.features(config.MIN_DATE, config.MODEL_TRAIN_END, radius_m).copy()
        available = start_date >= config.FORECAST_START
        if available:
            forecasts = self.regression.aggregate(start_date, end_date, radius_m)
            for column in FORECAST_COLUMNS:
                features[column] = forecasts[column].to_numpy()
        else:
            for column in FORECAST_COLUMNS:
                features[column] = np.nan
        features["start_date"], features["end_date"] = start_date, end_date
        scored = score_candidates(features, strategy)
        if strategy == "linear_regression" and not available:
            selected = scored.iloc[:0].copy()
            selected["selected_rank"] = pd.Series(dtype="int64")
            selected["budget"] = pd.Series(dtype="int64")
        else:
            selected = select_spaced(scored, self.blocked_neighbors, budget)
        columns = config.SCORED_COLUMNS + ["collision_count"] + CONTEXT_COLUMNS + FORECAST_COLUMNS
        return scored[columns], selected[columns + ["selected_rank", "budget"]]

    def _union_masks(self, selected, radius_m, start_date, end_date):
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        report_union, crash_union = np.zeros(len(self.reports), dtype=bool), np.zeros(len(self.collisions), dtype=bool)
        for target_id in selected["target_id"]:
            index = self.target_lookup[target_id]
            report_union[self._member_indices(self.report_neighbors, index, radius_m, report_mask)] = True
            crash_union[self._member_indices(self.crash_neighbors, index, radius_m, crash_mask)] = True
        return report_union, crash_union

    def coverage(self, selected, radius_m, start_date, end_date):
        reports, crashes = self._union_masks(selected, radius_m, start_date, end_date)
        return int(reports.sum()), int((crashes & self.harm).sum())

    def comparison(self, start_date, end_date, radius_m, budget):
        _, crash_mask = self._time_masks(start_date, end_date)
        available = start_date >= config.FORECAST_START and self.has_september_actuals
        denominator = int((crash_mask & self.harm).sum()) if available else None
        rows = []
        for strategy in config.STRATEGIES:
            _, selected = self.scored_and_selected(start_date, end_date, radius_m, strategy, budget)
            history_reports, _ = self.coverage(selected, radius_m, config.MIN_DATE, config.MODEL_TRAIN_END)
            reports, crashes = self._union_masks(selected, radius_m, start_date, end_date)
            harm = int((crashes & self.harm).sum()) if available else None
            rows.append(dict(strategy=strategy, radius_m=radius_m, budget=budget, selected_count=len(selected),
                             history_signal_reports_covered=history_reports,
                             future_signal_reports_covered=int(reports.sum()) if available else None,
                             future_collisions_covered=int(crashes.sum()) if available else None,
                             future_harm_crashes_covered=harm, future_harm_crashes_total=denominator,
                             future_harm_coverage_pct=(100 * harm / denominator if denominator else 0.0) if available else None,
                             start_date=start_date, end_date=end_date, forecast_start_date=start_date, forecast_end_date=end_date))
        return pd.DataFrame(rows)

    @lru_cache(maxsize=21)
    def historical_evaluation(self, radius_m, budget):
        result = self.comparison(config.FORECAST_START, config.FORECAST_END, radius_m, budget)
        # Compatibility fields describe the full September period ONLY.
        result["september_harm_crashes_covered"] = result["future_harm_crashes_covered"]
        result["september_harm_coverage_pct"] = result["future_harm_coverage_pct"]
        result["september_harm_crashes_total"] = result["future_harm_crashes_total"]
        result["august_signal_reports_covered"] = result["history_signal_reports_covered"]
        return result

    def area_comparison(self, start_date, end_date, radius_m):
        predictions = self.regression.aggregate(start_date, end_date, radius_m)
        actual = self.features(start_date, end_date, radius_m)
        columns = ("signal_report_count", "collision_count", "harm_crash_count")
        for outcome, column in zip(OUTCOMES, columns):
            predictions[f"actual_{outcome}"] = actual[column].to_numpy() if self.has_september_actuals else np.nan
            predictions[f"residual_{outcome}"] = predictions[f"actual_{outcome}"] - predictions[f"predicted_{outcome}"]
        return predictions

    @lru_cache(maxsize=16)
    def model_metrics(self, start_date, end_date, radius_m):
        if not self.has_september_actuals:
            return pd.DataFrame()
        frame = self.area_comparison(start_date, end_date, radius_m)
        rows = []
        for outcome in OUTCOMES:
            for estimator in ("linear_regression", "august_rate", "last_august_week_rate", "always_zero"):
                column = f"predicted_{outcome}" if estimator == "linear_regression" else f"{estimator}_{outcome}"
                predicted = np.zeros(len(frame)) if estimator == "always_zero" else frame[column].to_numpy()
                rows.append(dict(radius_m=radius_m, outcome=outcome, estimator=estimator,
                                 start_date=start_date, end_date=end_date, test_rows=len(frame),
                                 **regression_metrics(frame[f"actual_{outcome}"].to_numpy(), predicted)))
        return pd.DataFrame(rows)

    @lru_cache(maxsize=16)
    def analysis(self, start_date, end_date, radius_m, strategy, budget):
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        scored, selected = self.scored_and_selected(start_date, end_date, radius_m, strategy, budget)
        available = start_date >= config.FORECAST_START
        return dict(
            targets_scored=scored.reset_index(drop=True), selected_targets=selected.reset_index(drop=True),
            reports=self.reports.loc[report_mask].copy(), collisions=self.collisions.loc[crash_mask].copy(),
            area_comparison=self.area_comparison(start_date, end_date, radius_m) if available else pd.DataFrame(),
            comparison=self.comparison(start_date, end_date, radius_m, budget),
            evaluation=self.historical_evaluation(radius_m, budget).copy(),
            model=self.regression.description(radius_m),
            metadata=dict(
                start_date=start_date, end_date=end_date, radius_m=radius_m, strategy=strategy, budget=budget,
                input_version=self.version, method_version=config.METHOD_VERSION,
                date_granularity="day", timezone="America/New_York local-clock assumption",
                signal_reports_total=int(report_mask.sum()), collisions_total=int(crash_mask.sum()),
                harm_crashes_total=int((crash_mask & self.harm).sum()), candidate_count=len(self.candidates),
                candidate_period="August 2025 (fixed)", model_available=available,
                outcomes_available=available and self.has_september_actuals,
                training_start_date=config.MIN_DATE, training_end_date=config.MODEL_TRAIN_END,
                forecast_start_date=start_date if available else None, forecast_end_date=end_date if available else None,
                september_observations_used_for_predictions=False,
                model_note="All September forecasts frozen from August only. Select September dates for predictions; August dates display observations only.",
                analysis_kind="August-only forecast versus separately observed September outcomes",
                comparison_kind="Unique observed events within selected forecast-priority circles",
                evaluation_note="Full September comparison; previously explored, so exploratory evaluation. No September input or refitting.",
            ),
        )

    def evidence(self, target_id, radius_m, start_date, end_date):
        if target_id not in self.target_lookup:
            raise ValueError(f"Unknown target_id: {target_id}")
        index = self.target_lookup[target_id]
        report_mask, crash_mask = self._time_masks(start_date, end_date)
        result = {}
        for name, table, neighbors, mask in (("reports", self.reports, self.report_neighbors, report_mask),
                                             ("collisions", self.collisions, self.crash_neighbors, crash_mask)):
            ids, distances = neighbors[index]
            keep = (distances <= radius_m + DISTANCE_TOLERANCE_M) & mask[ids]
            records = table.iloc[ids[keep]].copy()
            records["distance_m"] = distances[keep]
            result[name] = records.reset_index(drop=True)
        return result
