# Shared data contracts — version 1

Agree on changes before coding. Paths are relative to repository root. Both source datasets contain local NYC timestamps with no explicit timezone offset. Interpret them as America/New_York; preserve local ISO timestamps in outputs. Date-only fields use `YYYY-MM-DD`.

## Person 1: 311 cleaning

Input: `data/raw/nyc311_dot_aug_sep_2025.csv`

Output: `data/processed/complaints.csv`, one usable geographic record per distinct `complaint_id`.

Columns: `complaint_id` (string from `unique_key`), `created_at` (ISO datetime from `created_date`), `date` (creation date), `borough` (uppercase or empty for unknown), `latitude` (number), `longitude` (number), `complaint_type` (string), `descriptor` (string), `status` (string).

Quality output: `data/processed/complaints_quality.json`.

## Person 2: crash cleaning

Input: `data/raw/nyc_crashes_aug_sep_2025.csv`

Output: `data/processed/crashes.csv`, one usable geographic record per distinct `crash_id`.

Columns: `crash_id` (string from `collision_id`), `occurred_at` (ISO datetime combining `crash_date` and `crash_time`), `date` (crash date), `borough` (uppercase or empty for unknown), `latitude` (number), `longitude` (number), `persons_injured` (integer), `persons_killed` (integer), `cyclists_injured` (integer).

Quality output: `data/processed/crashes_quality.json`.

## Common cleaning rules

- Validate IDs and duplicates; investigate conflicting duplicates rather than silently keeping one.
- Reject missing/unparseable, nonfinite, zero, or out-of-range coordinate pairs from geographic outputs. Use the initial screening box documented in the root README, or coordinate a documented refinement.
- Do not require a known borough when coordinates are usable; blank crash boroughs occur in the source.
- Validate dates and injury values; document invalid values separately. Do not silently convert missing injuries to zero.
- Quality JSON fields: `input_rows`, `output_rows`, `exclusions_by_reason` (mutually exclusive primary reasons), `warnings`, and `coordinate_rule`. Explain duplicate handling and date validation in `warnings` or additional fields.
- Retain raw data unchanged; no imputed borough centroids or made-up locations.

## Person 3: analysis

Inputs: the two cleaned CSVs and their quality reports.

Outputs:

- `data/processed/matches.csv`: `complaint_id`, `crash_id`, `distance_m`, `complaint_precedes_crash`. One pair per IDs, including all pairs within the largest evaluated radius (initially 100 m). Use the distance to derive the 25/50/100 m subsets; do not duplicate pairs for different radii. A preceding complaint means `created_at < occurred_at`.
- `data/processed/complaints.geojson` and `crashes.geojson`: FeatureCollections mirroring cleaned CSV properties; geometry is a Point using **[longitude, latitude]** in WGS84. These names are owned by Person 3, not the cleaners.
- `data/processed/summary.json`: documented, versioned statistics including `schema_version: 1`, `complaints_total`, `crashes_total`, `complaints_mappable`, `crashes_mappable`, `complaints_excluded`, `crashes_excluded`, and `radii_m: [25, 50, 100]`. Totals refer to raw input rows; explain any deduplication in additional fields.
- `analysis/findings.md`: evidence, radius sensitivity, limitations, and priority locations with a documented prioritization method.

Compute metric distances using a suitable projection or geodesic calculation, not Euclidean distance in latitude/longitude degrees. Counts of matches and counts of distinct crashes answer different questions. Two months of records limit before/after comparisons, particularly near the start and end of the window.

## Person 4: map and integration

Consumes the GeoJSON, `matches.csv`, and `summary.json`. Keep application code, build configuration, and presentation materials in `web/`. Own any conversion/copy into app-specific assets; never edit analysis output files in place. Document the app's setup and launch commands in `web/README.md`.

Framework choice is left to the team. Coordinate any additional summary fields with Person 3. Map popups and labels must distinguish raw events, matched pairs, distinct crashes, and exclusions.
