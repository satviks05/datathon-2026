"""Validated event IDs, projected cells, and complete half-open weekly panels."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from . import forecast_config as cfg


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def window(forecast_start):
    start = pd.Timestamp(forecast_start)
    return {
        "feature_start": (start - pd.Timedelta(days=7)).date().isoformat(),
        "feature_end": start.date().isoformat(),
        "forecast_start": start.date().isoformat(),
        "forecast_end": (start + pd.Timedelta(days=7)).date().isoformat(),
    }


def in_window(events, start, end):
    return events.loc[events["timestamp"].ge(pd.Timestamp(start))
                      & events["timestamp"].lt(pd.Timestamp(end))]


def project_cells(events):
    """Keep original IDs/timestamps; longitude is always the first coordinate."""
    events = events.copy()
    transform = Transformer.from_crs("EPSG:4326", cfg.GRID_CRS, always_xy=True)
    east, north = transform.transform(events["longitude"].tolist(),
                                      events["latitude"].tolist(), errcheck=True)
    cells = np.floor(np.column_stack([east, north]) / cfg.GRID_SIZE_M).astype("int64")
    events["cell_x"], events["cell_y"] = cells[:, 0], cells[:, 1]
    events["area_id"] = [f"cell_{x}_{y}" for x, y in cells]
    return events


def _validate(events, identifier):
    ids = events[identifier]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"Missing, blank, or duplicate {identifier}")
    times = events["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(times.dtype) or times.dt.tz is not None:
        raise ValueError("Expected NYC local-clock timestamps without timezone offsets")
    if times.isna().any() or not times.between("2025-08-01", "2025-10-01", inclusive="left").all():
        raise ValueError(f"Invalid or out-of-period timestamps for {identifier}")
    for name, low, high in (("latitude", 40.4, 41.0), ("longitude", -74.4, -73.6)):
        events[name] = pd.to_numeric(events[name], errors="raise")
        if not np.isfinite(events[name]).all() or not events[name].between(low, high).all():
            raise ValueError(f"Invalid mapped NYC {name}")
    return project_cells(events.sort_values(["timestamp", identifier], kind="stable").reset_index(drop=True))


def load_forecast_events(data_dir=cfg.DATA_DIR):
    """Only read relevant columns: closure/status/harm flags never enter features."""
    data_dir = Path(data_dir)
    reports = pd.read_csv(
        data_dir / cfg.INPUT_NAMES[0], dtype={"unique_key": "string"}, float_precision="round_trip",
        usecols=["unique_key", "created_date", "complaint_type", "latitude", "longitude"],
    )
    crashes = pd.read_csv(
        data_dir / cfg.INPUT_NAMES[1], dtype={"collision_id": "string"}, float_precision="round_trip",
        usecols=["collision_id", "crash_date", "crash_time", "latitude", "longitude"],
    )
    if not reports["complaint_type"].eq("Traffic Signal Condition").all():
        raise ValueError("Expected Traffic Signal Condition complaints only")
    reports["timestamp"] = pd.to_datetime(reports["created_date"], format="ISO8601", errors="raise")
    crashes["timestamp"] = pd.to_datetime(
        crashes["crash_date"].astype(str).str[:10] + " " + crashes["crash_time"].astype(str),
        format="%Y-%m-%d %H:%M", errors="raise",
    )
    return _validate(reports, "unique_key"), _validate(crashes, "collision_id")


def count_by_cell(events, identifier, start, end):
    return in_window(events, start, end).groupby("area_id")[identifier].nunique()


def eligible_areas(reports, crashes):
    early = pd.concat([
        in_window(events, cfg.UNIVERSE_START, cfg.UNIVERSE_END)[["area_id", "cell_x", "cell_y"]]
        for events in (reports, crashes)
    ]).drop_duplicates("area_id").sort_values("area_id").set_index("area_id")
    if early.empty:
        raise ValueError("No eligible areas in August 1–7; cannot fit the agreed workflow")
    transform = Transformer.from_crs(cfg.GRID_CRS, "EPSG:4326", always_xy=True)
    longitude, latitude = transform.transform(
        ((early["cell_x"] + 0.5) * cfg.GRID_SIZE_M).tolist(),
        ((early["cell_y"] + 0.5) * cfg.GRID_SIZE_M).tolist(), errcheck=True,
    )
    early["latitude"], early["longitude"] = latitude, longitude
    counts = count_by_cell(crashes, "collision_id", cfg.UNIVERSE_START, cfg.UNIVERSE_END)
    counts = counts.reindex(early.index, fill_value=0)
    early["activity_band"] = np.select([counts.eq(0), counts.eq(1)], ["zero", "one"], default="two_plus")
    return early[["latitude", "longitude", "activity_band"]].reset_index()


def build_features(reports, crashes):
    areas = eligible_areas(reports, crashes)
    rows = []
    for split, starts in (("train", cfg.TRAIN_STARTS), ("evaluation", cfg.EVALUATION_STARTS)):
        for start in starts:
            bounds = window(start)
            frame = areas.copy().set_index("area_id")
            frame["split"] = split
            for name, value in bounds.items():
                frame[name] = value
            for name, events, identifier, begin, end in (
                ("past_complaints", reports, "unique_key", bounds["feature_start"], bounds["feature_end"]),
                ("past_collisions", crashes, "collision_id", bounds["feature_start"], bounds["feature_end"]),
                ("actual_collisions", crashes, "collision_id", bounds["forecast_start"], bounds["forecast_end"]),
            ):
                frame[name] = count_by_cell(events, identifier, begin, end).reindex(frame.index, fill_value=0).astype("int64")
            rows.append(frame.reset_index())
    return pd.concat(rows, ignore_index=True)[cfg.FEATURE_COLUMNS]


def _coordinate_audit(frame):
    lat = pd.to_numeric(frame["latitude"], errors="coerce")
    lon = pd.to_numeric(frame["longitude"], errors="coerce")
    missing = lat.isna() | lon.isna()
    zero = ~missing & lat.eq(0) & lon.eq(0)
    mapped = lat.between(40.4, 41.0) & lon.between(-74.4, -73.6)
    return {
        "rows": len(frame), "mapped_rows": int(mapped.sum()),
        "missing_coordinate_pairs": int(missing.sum()), "zero_coordinate_pairs": int(zero.sum()),
        "other_outside_coordinate_bounds": int((~mapped & ~missing & ~zero).sum()),
        "coordinate_exclusions": int((~mapped).sum()),
    }


def source_audit(reports, crashes, raw_dir=None):
    audit = {
        "complaint_scope": "Traffic Signal Condition only",
        "mapped_cleaned_complaints": len(reports), "mapped_cleaned_collisions": len(crashes),
        "distinct_complaint_ids": int(reports.unique_key.nunique()),
        "distinct_collision_ids": int(crashes.collision_id.nunique()),
        "coordinate_screen": {"latitude": [40.4, 41.0], "longitude": [-74.4, -73.6]},
        "coordinate_screen_is_city_boundary": False,
        "timestamp_interpretation": "NYC local-clock time; half-open windows",
        "raw_audit_available": False,
    }
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        report_path = raw_dir / "nyc311_dot_aug_sep_2025.csv"
        crash_path = raw_dir / "nyc_crashes_aug_sep_2025.csv"
        if report_path.exists() and crash_path.exists():
            raw_reports = pd.read_csv(report_path, usecols=["complaint_type", "latitude", "longitude"])
            raw_crashes = pd.read_csv(crash_path, usecols=["latitude", "longitude"])
            signal = raw_reports.complaint_type.str.strip().eq("Traffic Signal Condition")
            audit.update(
                raw_audit_available=True,
                raw_complaints={"all_categories_rows": len(raw_reports),
                                "excluded_other_categories": int((~signal).sum()),
                                "signal_category": _coordinate_audit(raw_reports.loc[signal])},
                raw_collisions=_coordinate_audit(raw_crashes),
                raw_input_sha256={p.name: sha256_file(p) for p in (report_path, crash_path)},
            )
    return audit
