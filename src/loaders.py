"""Validate the existing snapshots without rewriting or re-cleaning them."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COUNT_COLUMNS = (
    "number_of_persons_injured", "number_of_persons_killed",
    "number_of_cyclist_injured",
)


def input_version(data_dir):
    """Content-based signature: never reuse results for different input bytes."""
    digest = hashlib.sha256()
    settings = {name: getattr(config, name) for name in (
        "METHOD_VERSION", "RADII", "BUDGETS", "STRATEGIES", "GRID_M",
        "SPACING_M", "EARTH_RADIUS_M", "GRID_ORIGIN_LAT", "GRID_ORIGIN_LON",
        "MIN_DATE", "MAX_DATE",
        "MODEL_TRAIN_END", "FORECAST_DAYS", "LAG_DAYS", "FORECAST_START", "FORECAST_END",
    )}
    settings["runtime"] = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    digest.update(json.dumps(settings, sort_keys=True).encode())
    for name in ("signal_complaints_clean.csv", "collisions_clean.csv"):
        digest.update(name.encode())
        digest.update((Path(data_dir) / name).read_bytes())
    # Method changes invalidate artifacts as well as configuration changes.
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_events(table, id_column, timestamp_column):
    ids = table[id_column]
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"Missing, blank, or duplicate {id_column}")
    for column, bounds in (("latitude", (40.4, 41.0)), ("longitude", (-74.4, -73.6))):
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all() or not table[column].between(*bounds).all():
            raise ValueError(f"Invalid NYC coordinates in {column}")
    times = table[timestamp_column]
    if times.isna().any() or not times.between(
        pd.Timestamp(config.MIN_DATE), pd.Timestamp(config.MAX_DATE) + pd.Timedelta(days=1),
        inclusive="left",
    ).all():
        raise ValueError(f"Invalid or out-of-period {timestamp_column}")
    return table.sort_values([timestamp_column, id_column], kind="stable").reset_index(drop=True)


def load_events(data_dir):
    data_dir = Path(data_dir)
    reports = pd.read_csv(data_dir / "signal_complaints_clean.csv", dtype={"unique_key": "string"})
    crashes = pd.read_csv(data_dir / "collisions_clean.csv", dtype={"collision_id": "string"})
    if not reports["complaint_type"].eq("Traffic Signal Condition").all():
        raise ValueError("Only Traffic Signal Condition reports are supported")
    reports["created_date"] = pd.to_datetime(reports["created_date"], errors="raise")
    reports["closed_date"] = (pd.to_datetime(reports["closed_date"], errors="raise")
                              if "closed_date" in reports else pd.NaT)
    crashes["occurred_at"] = pd.to_datetime(
        crashes["crash_date"].astype(str).str[:10] + " " + crashes["crash_time"].astype(str),
        format="%Y-%m-%d %H:%M", errors="raise",
    )
    for column in COUNT_COLUMNS:
        values = pd.to_numeric(crashes[column], errors="raise")
        if not np.isfinite(values).all() or values.lt(0).any() or values.mod(1).ne(0).any():
            raise ValueError(f"Invalid nonnegative integer counts in {column}")
        crashes[column] = values.astype("int64")
    for column, count in (("has_injury", COUNT_COLUMNS[0]), ("has_fatality", COUNT_COLUMNS[1])):
        values = crashes[column].astype(str).str.lower().map({"true": True, "false": False})
        if values.isna().any() or not values.eq(crashes[count].gt(0)).all():
            raise ValueError(f"Invalid or inconsistent {column}")
        crashes[column] = values.astype(bool)
    crashes["has_harm"] = crashes["has_injury"] | crashes["has_fatality"]
    return (
        _validate_events(reports, "unique_key", "created_date"),
        _validate_events(crashes, "collision_id", "occurred_at"),
    )
