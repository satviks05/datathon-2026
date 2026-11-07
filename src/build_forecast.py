"""Build the agreed weekly grid forecasts: python -m src.build_forecast."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd
import pyproj
import scipy
import sklearn

from . import forecast_config as cfg
from .forecast_features import build_features, load_forecast_events, sha256_file, source_audit, window
from .forecast_metrics import compare_models, evaluate
from .forecast_models import fit_models, model_settings, predict


LIMITATIONS = [
    "Exploratory historical evaluation: September has already been explored, not an untouched final test.",
    "Only three training and four evaluation windows; neighboring cells are not independent experiments.",
    "Eligible universe is frozen from August 1–7 complaints OR collisions; later outside-cell collisions are excluded from model rows but included in coverage denominators.",
    "Traffic Signal Condition complaints only; reports are not verified signal defects.",
    "Mapped event counts are not exposure-adjusted crash risk, causal effects, or injuries prevented.",
    "Coordinate screening is a rough box, not an NYC boundary polygon; unusable coordinates are excluded.",
    "Models and scalers are fit once on August training rows; later September features use observations available before each forecast date.",
    "September 29–30 are outside complete seven-day evaluation target windows.",
    "Half-open timestamps are interpreted as NYC local-clock time; no timezone offsets are encoded.",
]


def validate_panel(features):
    if list(features.columns) != cfg.FEATURE_COLUMNS or features.empty:
        raise ValueError("Empty or noncanonical feature panel")
    if features.duplicated(["area_id", "forecast_start"]).any():
        raise ValueError("Duplicate feature area/week")
    area_count = features.area_id.nunique()
    if len(features) != area_count * 7:
        raise ValueError("Expected every eligible cell in all seven windows")
    for split, starts in (("train", cfg.TRAIN_STARTS), ("evaluation", cfg.EVALUATION_STARTS)):
        frame = features.loc[features.split.eq(split)]
        if set(frame.forecast_start) != set(starts) or len(frame) != area_count * len(starts):
            raise ValueError("Incorrect forecast windows or split")
        for start, group in frame.groupby("forecast_start"):
            if group.area_id.nunique() != area_count:
                raise ValueError("Incomplete eligible universe in a window")
            for name, value in window(start).items():
                if not group[name].eq(value).all():
                    raise ValueError("Feature/target bounds disagree with the handoff")
    values = features[["past_complaints", "past_collisions", "actual_collisions"]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or not (values % 1 == 0).all():
        raise ValueError("Features and actuals must be finite nonnegative integer counts")
    if not features.activity_band.isin(cfg.BANDS).all():
        raise ValueError("Unknown activity band")
    if features.groupby("area_id").activity_band.nunique().gt(1).any():
        raise ValueError("Activity bands must be frozen")
    if not np.isfinite(features[["latitude", "longitude"]].to_numpy()).all():
        raise ValueError("Invalid cell center")


def build_forecast(data_dir=cfg.DATA_DIR, output_dir=cfg.DATA_DIR, raw_dir=None):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if raw_dir is None:
        raw_dir = data_dir.parent / "raw"
    inputs = {name: sha256_file(data_dir / name) for name in cfg.INPUT_NAMES}
    reports, crashes = load_forecast_events(data_dir)
    features = build_features(reports, crashes)
    validate_panel(features)
    models = fit_models(features)
    audit = source_audit(reports, crashes, raw_dir)
    runtime = dict(python=platform.python_version(), architecture=platform.machine(),
                   pandas=pd.__version__, numpy=np.__version__, scikit_learn=sklearn.__version__,
                   scipy=scipy.__version__, pyproj=pyproj.__version__, proj=pyproj.proj_version_str)
    implementation = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("forecast_*.py"))}
    implementation["build_forecast.py"] = sha256_file(__file__)
    provenance = dict(schema_version=cfg.SCHEMA_VERSION, input_sha256=inputs,
                      implementation_sha256=implementation, runtime=runtime, source_audit=audit,
                      model_settings=model_settings(models), grid_crs=cfg.GRID_CRS, grid_size_m=cfg.GRID_SIZE_M,
                      training_windows=[window(s) for s in cfg.TRAIN_STARTS],
                      evaluation_windows=[window(s) for s in cfg.EVALUATION_STARTS])
    run_id = "weekly-" + hashlib.sha256(json.dumps(provenance, sort_keys=True, allow_nan=False).encode()).hexdigest()
    predictions = predict(features, models, run_id)
    metrics = evaluate(predictions, crashes)
    metadata = dict(
        **provenance, run_id=run_id, created_at_utc=datetime.now(timezone.utc).isoformat(),
        universe_start=cfg.UNIVERSE_START, universe_end_exclusive=cfg.UNIVERSE_END,
        eligible_area_count=int(features.area_id.nunique()),
        activity_band_area_counts=features.drop_duplicates("area_id").activity_band.value_counts().to_dict(),
        evaluation_kind="exploratory historical evaluation", target="all mapped collisions in next seven days",
        top_k=cfg.TOP_K, rank_tie_break="prediction descending, area_id lexicographically ascending",
        comparisons=compare_models(metrics), limitations=LIMITATIONS,
        rows={"features.csv": len(features), "predictions.csv": len(predictions), "metrics.csv": len(metrics)},
        publication_contract="CSVs are staged and replaced, metadata last. Consumers must validate matching run IDs and output_sha256 to reject interrupted/mixed publication.",
    )
    # All validation and fitting must finish before any existing export is replaced.
    if inputs != {name: sha256_file(data_dir / name) for name in cfg.INPUT_NAMES}:
        raise ValueError("Cleaned inputs changed during the build; no outputs published")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forecast-stage-", dir=output_dir) as stage:
        stage = Path(stage)
        for name, table in (("features.csv", features), ("predictions.csv", predictions), ("metrics.csv", metrics)):
            table.to_csv(stage / name, index=False, encoding="utf-8", lineterminator="\n")
        metadata["output_sha256"] = {name: sha256_file(stage / name) for name in metadata["rows"]}
        (stage / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for name in cfg.OUTPUT_NAMES:
            os.replace(stage / name, output_dir / name)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=cfg.DATA_DIR)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Optional raw sources for exclusion audit; never cleaned or rewritten")
    args = parser.parse_args()
    metadata = build_forecast(args.data_dir, args.output_dir, args.raw_dir)
    print(json.dumps({"run_id": metadata["run_id"], "output_dir": str(args.output_dir.resolve()),
                      "eligible_area_count": metadata["eligible_area_count"], "rows": metadata["rows"],
                      "comparisons": metadata["comparisons"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
