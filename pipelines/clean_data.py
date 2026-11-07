"""Clean the two source CSVs for the traffic signal inspection project."""

import argparse
from pathlib import Path

import pandas as pd


START = pd.Timestamp("2025-08-01")
END = pd.Timestamp("2025-10-01")
COUNT_COLUMNS = [
    "number_of_persons_injured",
    "number_of_persons_killed",
    "number_of_cyclist_injured",
]


def clean_events(data, id_column, event_date):
    data = data.drop_duplicates().copy()
    if data[id_column].isna().any() or data[id_column].duplicated().any():
        raise ValueError(f"Missing or conflicting IDs in {id_column}")

    in_period = data[event_date].ge(START) & data[event_date].lt(END)
    if not in_period.all():
        raise ValueError(f"Invalid or out-of-period values in {event_date}")

    # Use a bounding box to exclude invalid points, including (0, 0).
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    geocoded = data["latitude"].between(40.4, 41.0) & data["longitude"].between(-74.4, -73.6)
    data = data.loc[geocoded].copy()
    data["borough"] = data["borough"].str.strip().str.upper().replace(
        {"": pd.NA, "UNSPECIFIED": pd.NA}
    )
    return data.sort_values([event_date, id_column])


def clean_complaints(source):
    reports = pd.read_csv(source, dtype={"unique_key": "string"}, float_precision="round_trip")
    reports = reports.loc[
        reports["complaint_type"].str.strip().eq("Traffic Signal Condition")
    ].copy()
    for column in ["agency", "complaint_type", "descriptor", "status"]:
        reports[column] = reports[column].str.strip()
    for column in ["created_date", "closed_date"]:
        reports[column] = pd.to_datetime(reports[column], errors="raise")

    if reports["closed_date"].lt(reports["created_date"]).any():
        raise ValueError("A 311 ticket closed before it was created")

    cleaned = clean_events(reports, "unique_key", "created_date")
    print(f"Signal complaints: {len(cleaned):,} of {len(reports):,} signal records")
    return cleaned


def clean_collisions(source):
    crashes = pd.read_csv(source, dtype={"collision_id": "string"}, float_precision="round_trip")
    crashes["crash_date"] = pd.to_datetime(crashes["crash_date"], errors="raise").dt.normalize()
    crashes["crash_time"] = pd.to_datetime(
        crashes["crash_time"], format="%H:%M", errors="raise"
    ).dt.strftime("%H:%M")

    for column in COUNT_COLUMNS:
        counts = pd.to_numeric(crashes[column], errors="raise")
        if counts.isna().any() or counts.lt(0).any() or counts.mod(1).ne(0).any():
            raise ValueError(f"Expected nonnegative integers in {column}")
        crashes[column] = counts.astype("int64")

    cleaned = clean_events(crashes, "collision_id", "crash_date")
    cleaned["has_injury"] = cleaned["number_of_persons_injured"].gt(0)
    cleaned["has_fatality"] = cleaned["number_of_persons_killed"].gt(0)
    print(f"Collisions: {len(cleaned):,} of {len(crashes):,} records")
    print(f"  With injury: {cleaned['has_injury'].sum():,}")
    print(f"  With injury or fatality: {(cleaned['has_injury'] | cleaned['has_fatality']).sum():,}")
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Folder containing the two source CSVs")
    parser.add_argument("output_dir", type=Path, help="Folder for the cleaned CSVs")
    args = parser.parse_args()

    reports = clean_complaints(args.source_dir / "nyc311_dot_aug_sep_2025.csv")
    crashes = clean_collisions(args.source_dir / "nyc_crashes_aug_sep_2025.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports.to_csv(
        args.output_dir / "signal_complaints_clean.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S"
    )
    crashes.to_csv(args.output_dir / "collisions_clean.csv", index=False, date_format="%Y-%m-%d")


if __name__ == "__main__":
    main()
