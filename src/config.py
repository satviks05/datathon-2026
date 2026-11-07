"""Shared values consumed by the backend and Streamlit controls."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RADII = (100, 125, 150, 175, 200, 225, 250)
BUDGETS = (25, 50, 100)
STRATEGIES = ("linear_regression", "crash_first", "complaint_first")
DEFAULT_RADIUS_M = 250
DEFAULT_BUDGET = 50
DEFAULT_STRATEGY = "linear_regression"
GRID_M = 50
SPACING_M = 250
EARTH_RADIUS_M = 6_371_008.8
GRID_ORIGIN_LAT = 40.7
GRID_ORIGIN_LON = -74.0
MIN_DATE = "2025-08-01"
MAX_DATE = "2025-09-30"
DEFAULT_START_DATE = "2025-08-25"
DEFAULT_END_DATE = "2025-08-31"
MODEL_TRAIN_END = "2025-08-31"
FORECAST_DAYS = 7
TRAIN_LOOKBACK_DAYS = (7, 14, 21)
METHOD_VERSION = "signal-area-ols-7day-v1"

SCORED_COLUMNS = [
    "target_id", "latitude", "longitude", "radius_m", "strategy",
    "signal_report_count", "harm_crash_count", "score", "top_descriptor",
    "suggested_inspection", "start_date", "end_date",
]
SELECTED_COLUMNS = SCORED_COLUMNS + ["selected_rank", "budget"]
EVALUATION_COLUMNS = [
    "strategy", "radius_m", "budget", "selected_count",
    "future_harm_crashes_covered", "future_harm_coverage_pct",
    "future_harm_crashes_total", "history_signal_reports_covered",
    "start_date", "end_date", "forecast_start_date", "forecast_end_date",
]
