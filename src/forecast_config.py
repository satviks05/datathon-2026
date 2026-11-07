"""Version-one weekly forecasting contract, independent of the legacy backend."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
SCHEMA_VERSION = 1
GRID_CRS = "EPSG:32618"
GRID_SIZE_M = 500
UNIVERSE_START = "2025-08-01"
UNIVERSE_END = "2025-08-08"
TRAIN_STARTS = ("2025-08-08", "2025-08-15", "2025-08-22")
EVALUATION_STARTS = ("2025-09-01", "2025-09-08", "2025-09-15", "2025-09-22")
MODELS = ("persistence", "collision_only", "combined")
MODEL_FEATURES = {
    "collision_only": ["past_collisions"],
    "combined": ["past_collisions", "past_complaints"],
}
BANDS = ("zero", "one", "two_plus")
TOP_K = 50
INPUT_NAMES = ("signal_complaints_clean.csv", "collisions_clean.csv")
FEATURE_COLUMNS = [
    "area_id", "latitude", "longitude", "activity_band", "split",
    "feature_start", "feature_end", "forecast_start", "forecast_end",
    "past_complaints", "past_collisions", "actual_collisions",
]
PREDICTION_COLUMNS = [
    "run_id", *[c for c in FEATURE_COLUMNS if c != "split"],
    *[f"{model}_prediction" for model in MODELS],
]
METRIC_COLUMNS = [
    "run_id", "model", "scope", "forecast_start", "activity_band", "area_week_rows",
    "mae", "top_k", "top_k_collisions", "citywide_collisions", "eligible_collisions",
    "top_k_coverage_pct", "eligible_coverage_pct",
]
OUTPUT_NAMES = ("features.csv", "predictions.csv", "metrics.csv", "run_metadata.json")
