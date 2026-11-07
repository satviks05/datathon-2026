# Shared data contracts — version 2

The existing team cleaner supersedes the starter's proposed cleaning schemas. Preserve the actual field names below; do not rename columns independently in the map and analysis. Both cleaned CSVs are committed snapshots and can be regenerated with `pipelines/clean_data.py`.

Paths are relative to repository root. Treat timestamps as America/New_York local time. Dates use `YYYY-MM-DD`. Coordinates are WGS84 decimal degrees.

## Existing cleaning outputs

### `data/processed/signal_complaints_clean.csv`

One Traffic Signal Condition report per `unique_key`; 6,119 records in the initial snapshot.

Columns: `unique_key` (string ID), `created_date` (local ISO datetime), `closed_date` (nullable local ISO datetime), `agency`, `complaint_type`, `descriptor`, `status`, `borough` (nullable), `latitude`, `longitude`.

The source includes 34,676 DOT complaints; 6,534 are signal reports. The cleaner drops 415 signal reports with unusable locations. Non-signal reports are outside project scope, not coordinate exclusions.

### `data/processed/collisions_clean.csv`

One collision per `collision_id`; 14,066 records in the initial snapshot.

Columns: `collision_id` (string ID), `crash_date` (date), `crash_time` (`HH:MM` local time), `borough` (nullable), `latitude`, `longitude`, `number_of_persons_injured`, `number_of_persons_killed`, `number_of_cyclist_injured` (nonnegative integer counts), `has_injury`, `has_fatality` (CSV `True`/`False` booleans).

Combine `crash_date` and `crash_time` for temporal comparisons. Parse boolean strings explicitly; the string `False` must not be treated as truthy.

## Ownership and quality reporting

Person 1 owns the shared cleaner and complaint-specific extensions. Person 2 owns separate crash validation and data-quality tooling in `pipelines/crashes/`. Do not split or edit the shared cleaner concurrently without coordinating.

Planned quality reports, not yet implemented: `data/processed/complaints_quality.json` and `crashes_quality.json`. Record input rows, category filtering, duplicates, coordinate exclusions, output rows, and the exact coordinate rule separately. Preserve raw inputs unchanged. Do not impute borough centroids for missing locations.

## Person 3: planned analysis outputs

Inputs: the two cleaned CSVs above. Preserve source identifiers and field names in GeoJSON properties.

- `data/processed/matches.csv`: `complaint_id` (from `unique_key`), `crash_id` (from `collision_id`), `distance_m`, `complaint_precedes_crash`. One pair per IDs, including all pairs within the largest evaluated radius (initially 100 m). Use distances to derive 25/50/100 m subsets without duplicating pairs per radius. Preceding means complaint creation strictly before combined crash datetime.
- `data/processed/complaints.geojson` and `crashes.geojson`: FeatureCollections mirroring their cleaned CSV properties; geometry is a Point in **[longitude, latitude]** order. Use JSON booleans for crash flags and null for missing borough/closure values. Person 3 owns these files.
- `data/processed/summary.json`: planned `schema_version: 2`, `dot_complaints_source_total`, `signal_complaints_total`, `signal_complaints_mappable`, `signal_complaints_coordinate_excluded`, `crashes_total`, `crashes_mappable`, `crashes_coordinate_excluded`, and `radii_m: [25, 50, 100]`. Explain duplicate handling separately when applicable.
- `analysis/findings.md`: evidence, radius sensitivity, limitations, and priority locations with a documented prioritization method.

Use geodesic distances or a suitable metric projection, not Euclidean distance in latitude/longitude degrees. Distinguish matched pairs from distinct crashes. Dates near the two-month window's edges have less observed history/follow-up. Geographic overlap alone does not establish a causal connection to signals.

## Person 4: planned map and integration

Consume the GeoJSON, matches, and summary. Keep app code, build configuration, and presentation in `web/`. Own any conversion/copy into app-specific assets; do not edit analysis outputs in place. Document dependencies, setup, and launch commands in `web/README.md`.

The map can also consume the cleaned CSV snapshots directly during development, using the same field names. Label the complaint layer **Traffic Signal Complaints**. Show category filtering and coordinate exclusions separately. Coordinate additional summary fields with Person 3.
