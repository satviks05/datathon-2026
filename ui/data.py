"""Read authoritative files and call the backend without duplicating its analysis."""

from importlib import import_module
from pathlib import Path
from math import isfinite

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
STRATEGIES = {
    "crash_first": "Crash-first",
    "complaint_first": "Complaint-first",
    "combined": "Combined",
}
RADII = (100, 125, 150, 175, 200, 225, 250)
BUDGETS = (25, 50, 100)
TARGET_COLUMNS = {
    "target_id", "latitude", "longitude", "radius_m", "strategy",
    "signal_report_count", "harm_crash_count", "score", "top_descriptor",
    "suggested_inspection", "selected_rank", "budget",
}
EVALUATION_COLUMNS = {
    "strategy", "radius_m", "budget", "selected_count",
    "september_harm_crashes_covered", "september_harm_coverage_pct",
    "august_signal_reports_covered",
}


def signature(path):
    path = Path(path)
    if not path.exists():
        return (str(path), None, None)
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def input_version():
    """Invalidate UI caches when inputs, exports, or backend configuration change."""
    paths = [DATA / name for name in (
        "signal_complaints_clean.csv", "collisions_clean.csv",
        "targets_scored.csv", "selected_targets.csv", "evaluation.csv",
    )]
    config = ROOT / "src"
    if config.exists():
        paths.extend(sorted(config.rglob("*.py")))
    return tuple(signature(path) for path in paths)


@st.cache_data(show_spinner=False)
def read_csv(path, file_signature):
    # file_signature is intentionally a cache argument.
    return pd.read_csv(path, dtype={"unique_key": "string", "collision_id": "string",
                                    "target_id": "string"})


def load_source_events():
    reports = read_csv(str(DATA / "signal_complaints_clean.csv"),
                       signature(DATA / "signal_complaints_clean.csv")).copy()
    crashes = read_csv(str(DATA / "collisions_clean.csv"),
                       signature(DATA / "collisions_clean.csv")).copy()
    reports["created_date"] = pd.to_datetime(reports["created_date"], errors="raise")
    crashes["occurred_at"] = pd.to_datetime(
        crashes["crash_date"].astype(str).str[:10] + " " + crashes["crash_time"].astype(str),
        errors="raise",
    )
    # Explicitly parse CSV booleans; bool("False") would be incorrect.
    for name in ("has_injury", "has_fatality"):
        values = crashes[name].astype(str).str.lower().map({"true": True, "false": False})
        if values.isna().any():
            raise ValueError(f"Invalid boolean value in {name}")
        crashes[name] = values
    crashes["has_harm"] = crashes["has_injury"] | crashes["has_fatality"]
    return reports, crashes


def load_export(filename, required):
    path = DATA / filename
    if not path.exists():
        return pd.DataFrame(columns=sorted(required)), f"Awaiting {filename}."
    table = read_csv(str(path), signature(path)).copy()
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {', '.join(sorted(missing))}")
    for column in required & {"latitude", "longitude", "radius_m", "budget", "score",
                              "selected_rank", "signal_report_count", "harm_crash_count",
                              "selected_count", "september_harm_crashes_covered",
                              "september_harm_coverage_pct", "august_signal_reports_covered"}:
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not table[column].map(isfinite).all():
            raise ValueError(f"{filename} has missing or nonfinite numeric values in {column}")
        if column not in {"latitude", "longitude"} and table[column].lt(0).any():
            raise ValueError(f"{filename} has negative values in {column}")
    if "september_harm_coverage_pct" in table and not table["september_harm_coverage_pct"].between(0, 100).all():
        raise ValueError(f"{filename} coverage percentages must be between 0 and 100")
    if not table["strategy"].isin(STRATEGIES).all():
        raise ValueError(f"{filename} contains an unknown strategy")
    if not table["radius_m"].isin(RADII).all() or not table["budget"].isin(BUDGETS).all():
        raise ValueError(f"{filename} contains unsupported radius/budget values")
    keys = ["strategy", "radius_m", "budget"]
    if "target_id" in required:
        if table["target_id"].isna().any():
            raise ValueError(f"{filename} contains missing target IDs")
        keys += ["target_id"]
        if not table["latitude"].between(-90, 90).all() or not table["longitude"].between(-180, 180).all():
            raise ValueError(f"{filename} has invalid coordinates")
    if table.duplicated(keys).any():
        raise ValueError(f"{filename} contains duplicate configuration rows")
    return table, None


def select_configuration(table, radius_m, budget, strategy=None):
    mask = table["radius_m"].eq(radius_m) & table["budget"].eq(budget)
    if strategy is not None:
        mask &= table["strategy"].eq(strategy)
    return table.loc[mask].copy()


def development_fixture(radius_m, budget, strategy):
    """Static layout examples, not candidates, computed scores, or findings."""
    examples = [
        ("UI-EXAMPLE-A", 40.7506, -73.9935, 0.90, 12, 4, "Controller"),
        ("UI-EXAMPLE-B", 40.7359, -73.9911, 0.80, 8, 3, "Pedestrian Signal"),
        ("UI-EXAMPLE-C", 40.7061, -73.9969, 0.70, 6, 2, "Veh Signal Lamp"),
    ]
    return pd.DataFrame([
        dict(target_id=tid, latitude=lat, longitude=lon, score=score,
             signal_report_count=reports, harm_crash_count=crashes,
             top_descriptor=descriptor, selected_rank=rank, radius_m=radius_m,
             strategy=strategy, budget=budget,
             suggested_inspection="Example wording only — no field recommendation.")
        for rank, (tid, lat, lon, score, reports, crashes, descriptor)
        in enumerate(examples, start=1)
    ])


@st.cache_data(show_spinner=False)
def area_evidence(target_id, radius_m, period, version):
    try:
        module = import_module("src.api")
    except ModuleNotFoundError as exc:
        if exc.name in {"src", "src.api"}:
            return None
        raise
    helper = getattr(module, "get_area_evidence", None)
    if helper is None:
        raise ValueError("src.api must expose get_area_evidence(target_id, radius_m, period)")
    result = helper(target_id, radius_m, period=period)
    if not isinstance(result, dict) or not all(
        isinstance(result.get(key), pd.DataFrame) for key in ("reports", "collisions")
    ):
        raise ValueError("Evidence helper must return reports and collisions DataFrames")
    # Display the requested period only; do not silently accept mixed-period evidence.
    for name, id_column, date_column in (
        ("reports", "unique_key", "created_date"),
        ("collisions", "collision_id", "crash_date"),
    ):
        table = result[name]
        if not table.empty:
            if not {id_column, date_column}.issubset(table.columns):
                raise ValueError(f"Evidence {name} must preserve {id_column} and {date_column}")
            dates = pd.to_datetime(table[date_column], errors="raise")
            month = 8 if period == "august" else 9
            if not ((dates.dt.year == 2025) & (dates.dt.month == month)).all():
                raise ValueError(f"Evidence {name} does not match the requested {period} period")
    return result
