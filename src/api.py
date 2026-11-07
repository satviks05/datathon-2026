"""Public backend interface. This module never imports Streamlit."""

from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config
from .engine import AnalysisEngine, date_window, validate_controls
from .loaders import input_version, load_events
from .regression import AugustOnlyRegression
from .spatial import build_candidates


@lru_cache(maxsize=2)
def _load_engine(data_dir, version):
    return AnalysisEngine(Path(data_dir), version)


def _engine(data_dir=None):
    directory = Path(data_dir or config.DATA_DIR).resolve()
    return _load_engine(str(directory), input_version(directory))


@lru_cache(maxsize=2)
def _load_august_model(train_dir, version):
    reports, crashes = load_events(train_dir)
    return AugustOnlyRegression(reports, crashes, build_candidates(reports))


def get_september_forecast(start_date=config.DEFAULT_START_DATE, end_date=config.DEFAULT_END_DATE,
                          radius_m=config.DEFAULT_RADIUS_M, *, train_dir=None):
    """Forecast-only interface: reads August files, requires NO September file."""
    validate_controls(radius_m)
    start, stop = date_window(start_date, end_date)
    directory = Path(train_dir or config.DATA_DIR / "august_train").resolve()
    model = _load_august_model(str(directory), input_version(directory))
    return model.aggregate(start.date().isoformat(), (stop - pd.Timedelta(days=1)).date().isoformat(), radius_m).copy(deep=True)


def get_config(*, data_dir=None):
    return dict(
        radii_m=list(config.RADII), budgets=list(config.BUDGETS), strategies=list(config.STRATEGIES),
        default_radius_m=config.DEFAULT_RADIUS_M, default_budget=config.DEFAULT_BUDGET,
        default_strategy=config.DEFAULT_STRATEGY, min_date=config.FORECAST_START, max_date=config.FORECAST_END,
        source_min_date=config.MIN_DATE, source_max_date=config.MAX_DATE,
        default_start_date=config.DEFAULT_START_DATE, default_end_date=config.DEFAULT_END_DATE,
        date_step_days=1, radius_step_m=25, grid_m=config.GRID_M, spacing_m=config.SPACING_M,
        method_version=config.METHOD_VERSION,
        forecast_days=config.FORECAST_DAYS, model_training_end_date=config.MODEL_TRAIN_END,
        lag_days=config.LAG_DAYS, forecast_mode="August-only frozen September forecast",
        input_version=input_version(Path(data_dir or config.DATA_DIR)),
    )


def get_analysis(start_date=config.DEFAULT_START_DATE, end_date=config.DEFAULT_END_DATE,
                 radius_m=config.DEFAULT_RADIUS_M, strategy=config.DEFAULT_STRATEGY,
                 budget=config.DEFAULT_BUDGET, *, data_dir=None):
    """Inclusive daily range. Return one consistent bundle for all explorer panels.

    Cached internal results are copied so one UI session cannot alter another's.
    """
    validate_controls(radius_m, budget, strategy)
    start, stop = date_window(start_date, end_date)
    start, end = start.date().isoformat(), (stop - pd.Timedelta(days=1)).date().isoformat()
    return deepcopy(_engine(data_dir).analysis(start, end, int(radius_m), strategy, int(budget)))


def get_area_evidence(target_id, radius_m, period="august", *, start_date=None, end_date=None, data_dir=None):
    """Original nearby records, including all crash outcomes and distance_m.

    Explicit start_date/end_date override period; provide both for the explorer.
    The original three-argument UI contract remains supported.
    """
    validate_controls(radius_m)
    if (start_date is None) != (end_date is None):
        raise ValueError("Provide both start_date and end_date")
    if start_date is None:
        if period not in {"august", "september"}:
            raise ValueError("period must be august or september")
        start_date, end_date = (("2025-08-01", "2025-08-31") if period == "august"
                                else ("2025-09-01", "2025-09-30"))
    date_window(start_date, end_date)
    return _engine(data_dir).evidence(str(target_id), radius_m, start_date, end_date)


def get_historical_evaluation(radius_m=config.DEFAULT_RADIUS_M, budget=config.DEFAULT_BUDGET, *, data_dir=None):
    """Full September forecast comparison; all model inputs are August only."""
    validate_controls(radius_m, budget)
    return _engine(data_dir).historical_evaluation(int(radius_m), int(budget)).copy(deep=True)


def get_model_metrics(radius_m=config.DEFAULT_RADIUS_M, *, start_date=config.DEFAULT_START_DATE,
                      end_date=config.DEFAULT_END_DATE, data_dir=None):
    """Matching selected-September MAE/RMSE/R2 for all three forecast targets."""
    validate_controls(radius_m)
    date_window(start_date, end_date)
    if pd.Timestamp(start_date) < pd.Timestamp(config.FORECAST_START):
        raise ValueError("Model metrics compare September forecasts only")
    return _engine(data_dir).model_metrics(str(pd.Timestamp(start_date).date()), str(pd.Timestamp(end_date).date()), radius_m).copy(deep=True)


def get_data_quality(*, data_dir=None):
    engine = _engine(data_dir)
    reports, crashes = engine.reports, engine.collisions
    months = {}
    for label, start, end in (("august", "2025-08-01", "2025-08-31"),
                              ("september", "2025-09-01", "2025-09-30")):
        report_mask, crash_mask = engine._time_masks(start, end)
        months[label] = dict(signal_reports=int(report_mask.sum()), collisions=int(crash_mask.sum()),
                             harm_crashes=int((crash_mask & engine.harm).sum()))
    result = dict(
        signal_reports=len(reports), collisions=len(crashes), candidate_count=len(engine.candidates),
        harm_crashes=int(crashes["has_harm"].sum()), injury_crashes=int(crashes["has_injury"].sum()),
        fatal_crashes=int(crashes["has_fatality"].sum()),
        fatal_only_crashes=int((crashes["has_fatality"] & ~crashes["has_injury"]).sum()),
        persons_injured=int(crashes["number_of_persons_injured"].sum()),
        persons_killed=int(crashes["number_of_persons_killed"].sum()),
        reports_missing_borough=int(reports["borough"].isna().sum()),
        collisions_missing_borough=int(crashes["borough"].isna().sum()),
        months=months, input_version=engine.version,
        assumptions=["Source timestamps are NYC local clock; no offsets supplied.",
                     "Grid grouping is approximate; radius/spacing use haversine meters.",
                     "Fixed August candidates make this retrospective exploration, not an as-of replay.",
                     "Closure status does not establish that a physical defect was resolved.",
                     "Reported components and nearby crashes do not establish fault or causation."],
    )
    raw = engine.data_dir.parent / "raw"
    report_source, crash_source = raw / "nyc311_dot_aug_sep_2025.csv", raw / "nyc_crashes_aug_sep_2025.csv"
    if report_source.exists():
        original = pd.read_csv(report_source, usecols=["unique_key", "complaint_type"], dtype={"unique_key": "string"})
        signals = original.loc[original["complaint_type"].str.strip().eq("Traffic Signal Condition")]
        result.update(signal_reports_source=len(signals),
                      signal_reports_excluded=int((~signals["unique_key"].isin(reports["unique_key"])).sum()))
    if crash_source.exists():
        original = pd.read_csv(crash_source, usecols=["collision_id"], dtype={"collision_id": "string"})
        result.update(collisions_source=len(original),
                      collisions_excluded=int((~original["collision_id"].isin(crashes["collision_id"])).sum()))
    return result
