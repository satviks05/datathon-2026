"""Forecast export validation and presentation filters; no training or metric computation."""

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODELS = {
    "combined": "Combined",
    "collision_only": "Collision history",
    "persistence": "Previous week",
}
PREDICTION_COLUMNS = {model: f"{model}_prediction" for model in MODELS}
WEEKS = ("2025-09-01", "2025-09-08", "2025-09-15", "2025-09-22")
BANDS = {"zero", "one", "two_plus"}
SOURCES = ("signal_complaints_clean.csv", "collisions_clean.csv")
EXPORTS = ("predictions.csv", "metrics.csv", "run_metadata.json")
PREDICTION_FIELDS = {
    "run_id", "area_id", "latitude", "longitude", "activity_band", "feature_start",
    "feature_end", "forecast_start", "forecast_end", "past_complaints",
    "past_collisions", "actual_collisions", *PREDICTION_COLUMNS.values(),
}
METRIC_FIELDS = {
    "run_id", "model", "scope", "forecast_start", "activity_band", "area_week_rows",
    "mae", "top_k", "top_k_collisions", "citywide_collisions", "eligible_collisions",
    "top_k_coverage_pct", "eligible_coverage_pct",
}
COVERAGE_FIELDS = (
    "top_k", "top_k_collisions", "citywide_collisions", "eligible_collisions",
    "top_k_coverage_pct", "eligible_coverage_pct",
)


class ForecastDataError(ValueError):
    """A canonical export violates the agreed UI contract."""


@dataclass
class ForecastBundle:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    metadata: dict
    stale_reasons: tuple = ()
    fixture: bool = False
    provenance_notes: tuple = ()


def signature(path):
    path = Path(path)
    if not path.exists():
        return str(path), None, None
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False)
def source_hash(path, file_signature, normalize_newlines=False):
    # The signature is part of the cache key; don't prefix it with an underscore.
    with Path(path).open("rb") as source:
        content = source.read()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    return sha256(content).hexdigest()


def _require(table, columns, name):
    missing = columns - set(table.columns)
    if missing:
        raise ForecastDataError(f"{name} is missing: {', '.join(sorted(missing))}")
    if table.empty:
        raise ForecastDataError(f"{name} has no rows")


def _numbers(table, columns, name, *, integer=False, nullable=False):
    for column in columns:
        try:
            values = pd.to_numeric(table[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ForecastDataError(f"{name}: {column} must be numeric") from exc
        nonblank = values.dropna() if nullable else values
        if (not np.isfinite(nonblank.to_numpy(dtype=float)).all()
                or (not nullable and values.isna().any()) or nonblank.lt(0).any()):
            raise ForecastDataError(f"{name}: {column} must be finite and nonnegative")
        if integer and nonblank.mod(1).ne(0).any():
            raise ForecastDataError(f"{name}: {column} must contain whole counts")
        table[column] = values


def _validate_metadata(meta, fixture):
    if not isinstance(meta, dict):
        raise ForecastDataError("run_metadata.json must contain an object")
    if type(meta.get("schema_version")) is not int or meta["schema_version"] != 1:
        raise ForecastDataError("Unsupported schema_version; expected integer 1")
    if not isinstance(meta.get("run_id"), str) or not meta["run_id"].strip():
        raise ForecastDataError("Metadata needs a nonempty run_id")
    if bool(meta.get("development_fixture", False)) != fixture:
        raise ForecastDataError("Fixture metadata is not valid as a production run")
    if meta.get("grid_size_m") != 500 or meta.get("grid_crs") != "EPSG:32618":
        raise ForecastDataError("Expected the agreed 500 m EPSG:32618 grid")
    if meta.get("universe_start") != "2025-08-01" or meta.get("universe_end_exclusive") != "2025-08-08":
        raise ForecastDataError("Expected eligibility frozen using August 1–7 only")
    count = meta.get("eligible_area_count")
    if type(count) is not int or count < 1:
        raise ForecastDataError("eligible_area_count must be a positive integer")
    required = {"created_at_utc", "input_sha256", "training_windows", "evaluation_windows",
                "model_settings", "source_audit", "limitations"}
    if required - set(meta):
        raise ForecastDataError("Metadata missing: " + ", ".join(sorted(required - set(meta))))
    if not isinstance(meta["input_sha256"], dict):
        raise ForecastDataError("input_sha256 must map cleaned input filenames to SHA-256 hashes")


def _validate_predictions(predictions, meta):
    _require(predictions, PREDICTION_FIELDS, "predictions.csv")
    for column in ("run_id", "area_id", "activity_band"):
        if predictions[column].isna().any() or predictions[column].astype(str).str.strip().eq("").any():
            raise ForecastDataError(f"predictions.csv has missing {column}")
    if predictions["run_id"].ne(meta["run_id"]).any():
        raise ForecastDataError("Mixed run IDs in predictions.csv / metadata")
    if not predictions["activity_band"].isin(BANDS).all():
        raise ForecastDataError("Unknown activity band")
    if not predictions["area_id"].str.fullmatch(r"cell_-?\d+_-?\d+").all():
        raise ForecastDataError("area_id must follow cell_{cell_x}_{cell_y}")
    if predictions.duplicated(["area_id", "forecast_start"]).any():
        raise ForecastDataError("Duplicate area/week prediction rows")
    for column in ("feature_start", "feature_end", "forecast_start", "forecast_end"):
        values = predictions[column].astype("string")
        if values.isna().any() or not values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ForecastDataError(f"{column} must use YYYY-MM-DD")
        try:
            predictions[column] = pd.to_datetime(values, format="%Y-%m-%d", errors="raise").dt.strftime("%Y-%m-%d")
        except ValueError as exc:
            raise ForecastDataError(f"Invalid dates in {column}") from exc
    if set(predictions["forecast_start"]) != set(WEEKS):
        raise ForecastDataError("Expected all four September forecast weeks")
    start = predictions["forecast_start"]
    expected = {
        "feature_start": start.map(lambda value: (date.fromisoformat(value) - timedelta(days=7)).isoformat()),
        "feature_end": start,
        "forecast_end": start.map(lambda value: (date.fromisoformat(value) + timedelta(days=7)).isoformat()),
    }
    if any(not predictions[column].eq(values).all() for column, values in expected.items()):
        raise ForecastDataError("Feature/forecast windows must be adjacent seven-day half-open windows")
    _numbers(predictions, ["past_complaints", "past_collisions", "actual_collisions"],
             "predictions.csv", integer=True)
    _numbers(predictions, PREDICTION_COLUMNS.values(), "predictions.csv")
    for column, lower, upper in (("latitude", 40.4, 41.0), ("longitude", -74.4, -73.6)):
        try:
            predictions[column] = pd.to_numeric(predictions[column], errors="raise")
        except ValueError as exc:
            raise ForecastDataError(f"Invalid {column}") from exc
        if not predictions[column].between(lower, upper).all():
            raise ForecastDataError(f"Invalid NYC cell-center {column}")
    # Preserve zeros: require the exact same universe in all four windows.
    universes = [set(group["area_id"]) for _, group in predictions.groupby("forecast_start")]
    if any(universe != universes[0] for universe in universes) or len(universes[0]) != meta["eligible_area_count"]:
        raise ForecastDataError("Every week must include the same eligible cells, including zero-event cells")
    for column in ("latitude", "longitude", "activity_band"):
        if predictions.groupby("area_id")[column].nunique().gt(1).any():
            raise ForecastDataError(f"Cell {column} changed across forecast weeks")


def _validate_metrics(metrics, predictions, meta):
    _require(metrics, METRIC_FIELDS, "metrics.csv")
    if metrics["run_id"].isna().any() or metrics["run_id"].ne(meta["run_id"]).any():
        raise ForecastDataError("Mixed run IDs in metrics.csv / metadata")
    if not metrics["model"].isin(MODELS).all() or not metrics["scope"].isin({"week", "overall", "activity_band"}).all():
        raise ForecastDataError("Unknown metric model or scope")
    metrics[["forecast_start", "activity_band"]] = metrics[["forecast_start", "activity_band"]].fillna("")
    if metrics.duplicated(["model", "scope", "forecast_start", "activity_band"]).any():
        raise ForecastDataError("Duplicate metric rows")
    _numbers(metrics, ["mae"], "metrics.csv")
    _numbers(metrics, ["area_week_rows"], "metrics.csv", integer=True)
    if metrics["area_week_rows"].le(0).any():
        raise ForecastDataError("Metrics need positive area_week_rows")
    _numbers(metrics, COVERAGE_FIELDS[:4], "metrics.csv", integer=True, nullable=True)
    _numbers(metrics, COVERAGE_FIELDS[4:], "metrics.csv", nullable=True)
    weekly = metrics.loc[metrics["scope"].eq("week")]
    expected = {(model, week) for model in MODELS for week in WEEKS}
    if set(zip(weekly["model"], weekly["forecast_start"])) != expected or weekly["activity_band"].ne("").any():
        raise ForecastDataError("Metrics need one weekly row for every model/week")
    overall = metrics.loc[metrics["scope"].eq("overall")]
    if len(overall) != 3 or set(overall["model"]) != set(MODELS) or overall[["forecast_start", "activity_band"]].ne("").any().any():
        raise ForecastDataError("Metrics need one pooled overall row per model")
    bands = metrics.loc[metrics["scope"].eq("activity_band")]
    expected_bands = {(model, band) for model in MODELS for band in predictions["activity_band"].unique()}
    if set(zip(bands["model"], bands["activity_band"])) != expected_bands or bands["forecast_start"].ne("").any():
        raise ForecastDataError("Metrics need pooled activity-band rows for every populated band/model")
    if bands[list(COVERAGE_FIELDS)].notna().any().any():
        raise ForecastDataError("Activity-band coverage fields must be blank, not zero")
    coverage = metrics.loc[~metrics["scope"].eq("activity_band")]
    if coverage[list(COVERAGE_FIELDS[:4])].isna().any().any() or coverage["top_k"].ne(50).any():
        raise ForecastDataError("Weekly/overall coverage needs count denominators and top_k=50")
    if ((coverage["top_k_collisions"] > coverage["eligible_collisions"]).any()
            or (coverage["eligible_collisions"] > coverage["citywide_collisions"]).any()):
        raise ForecastDataError("Coverage must satisfy top-50 <= eligible <= citywide collision counts")
    for column in COVERAGE_FIELDS[4:]:
        if not coverage.loc[coverage["citywide_collisions"].gt(0), column].between(0, 100).all():
            raise ForecastDataError(f"{column} must be 0–100 for positive denominators")
        if coverage.loc[coverage["citywide_collisions"].eq(0), column].notna().any():
            raise ForecastDataError("Zero-denominator coverage must be blank")
    # Validate row counts, without recomputing backend MAE or coverage.
    n = meta["eligible_area_count"]
    if weekly["area_week_rows"].ne(n).any() or overall["area_week_rows"].ne(4 * n).any():
        raise ForecastDataError("Metric row counts do not match the prediction universe")
    for band, frame in bands.groupby("activity_band"):
        expected_n = int(predictions["activity_band"].eq(band).sum())
        if frame["area_week_rows"].ne(expected_n).any():
            raise ForecastDataError("Activity-band row counts do not match predictions")


def _expected_hash(meta, filename):
    matches = []
    for key, value in meta["input_sha256"].items():
        if str(key).replace("\\", "/").split("/")[-1] == filename:
            if isinstance(value, dict):
                value = value.get("sha256")
            matches.append(value)
    if len(matches) != 1 or not isinstance(matches[0], str) or len(matches[0]) != 64:
        raise ForecastDataError(f"Metadata needs one SHA-256 hash for {filename}")
    digest = matches[0].lower()
    if any(c not in "0123456789abcdef" for c in digest):
        raise ForecastDataError(f"Invalid SHA-256 for {filename}")
    return digest


@st.cache_data(show_spinner=False)
def _read_bundle(directory, source_directory, file_signatures, run_id, fixture):
    directory, source_directory = Path(directory), Path(source_directory)
    try:
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(directory / "predictions.csv", dtype={"run_id": "string", "area_id": "string"})
        metrics = pd.read_csv(directory / "metrics.csv", dtype={"run_id": "string"})
    except (OSError, ValueError) as exc:
        raise ForecastDataError(f"Could not read forecast outputs: {exc}") from exc
    _validate_metadata(metadata, fixture)
    if metadata["run_id"] != run_id:
        raise ForecastDataError("Outputs changed during loading; retry after the build finishes")
    _validate_predictions(predictions, metadata)
    _validate_metrics(metrics, predictions, metadata)
    stale, notes = [], []
    for filename in SOURCES:
        expected = _expected_hash(metadata, filename)
        if fixture:
            continue
        path = source_directory / filename
        if not path.exists():
            stale.append(f"Cannot verify missing cleaned input: {filename}")
        elif source_hash(str(path), signature(path)) != expected:
            if source_hash(str(path), signature(path), normalize_newlines=True) == expected:
                notes.append(f"{filename}: verified after normalizing Windows CRLF line endings to LF")
            else:
                stale.append(f"Input file differs from the model run: {filename}")
    paths = [directory / name for name in EXPORTS] + [source_directory / name for name in SOURCES]
    if tuple(signature(path) for path in paths) != file_signatures:
        raise ForecastDataError("Files changed during loading; retry after the build finishes")
    return ForecastBundle(predictions, metrics, metadata, tuple(stale), fixture, tuple(notes))


def load_forecast_bundle(*, fixture=False):
    directory = FIXTURES if fixture else DATA
    missing = [name for name in EXPORTS if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError("Awaiting canonical forecast outputs: " + ", ".join(missing))
    try:
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForecastDataError(f"Cannot read run_metadata.json: {exc}") from exc
    _validate_metadata(metadata, fixture)
    paths = [directory / name for name in EXPORTS] + [DATA / name for name in SOURCES]
    return _read_bundle(str(directory), str(DATA), tuple(signature(path) for path in paths), metadata["run_id"], fixture)


def week_rows(bundle, week, model, top_only=False):
    rows = bundle.predictions.loc[bundle.predictions["forecast_start"].eq(week)].copy()
    rows = rows.sort_values([PREDICTION_COLUMNS[model], "area_id"], ascending=[False, True], kind="stable")
    return rows.head(50).copy() if top_only else rows


def week_label(start):
    start = date.fromisoformat(str(start)[:10])
    end = start + timedelta(days=6)
    return f"{start.strftime('%b')} {start.day}–{end.day}, {start.year}"


@st.cache_data(show_spinner=False)
def _read_source_events(directory, signatures):
    directory = Path(directory)
    reports = pd.read_csv(directory / SOURCES[0], dtype={"unique_key": "string"})
    crashes = pd.read_csv(directory / SOURCES[1], dtype={"collision_id": "string"})
    reports["event_at"] = pd.to_datetime(reports["created_date"], errors="raise")
    crashes["event_at"] = pd.to_datetime(
        crashes["crash_date"].astype(str).str[:10] + " " + crashes["crash_time"].astype(str), errors="raise")
    for frame in (reports, crashes):
        for column in ("latitude", "longitude"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return reports, crashes


def source_events_for_week(forecast_week):
    """Read existing cleaned observations only; no forecasts or evaluation metrics."""
    signatures = tuple(signature(DATA / name) for name in SOURCES)
    reports, crashes = _read_source_events(str(DATA), signatures)
    end = pd.Timestamp(forecast_week)
    start = pd.Timestamp(date.fromisoformat(forecast_week) - timedelta(days=7))
    return tuple(frame.loc[frame["event_at"].ge(start) & frame["event_at"].lt(end)].copy()
                 for frame in (reports, crashes))
