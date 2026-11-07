# Weekly collision forecast backend

This implements `PROJECT_HANDOFF.md`, schema version **1**. Run from the repository root on 64-bit Python 3.12:

```sh
python -m pip install -r requirements-model.txt
python -m src.build_forecast
python -m unittest discover -s tests/backend -v
```

The two committed cleaned inputs and the existing cleaner remain unchanged. The build reads `data/processed/signal_complaints_clean.csv` and `collisions_clean.csv`. Only the four canonical forecast outputs are written. Scratch builds can use `--output-dir tmp/forecast-check`.

## Export contract for the UI owner

UTF-8 CSV, no index, ISO dates, exclusive end bounds. `area_id` is a string. Every eligible cell appears in all seven feature windows and all four prediction windows, including zero counts. Predictions are unrounded, nonnegative expected **all-collision counts**, not injury counts or probabilities.

`features.csv`:

```text
area_id,latitude,longitude,activity_band,split,feature_start,feature_end,forecast_start,forecast_end,past_complaints,past_collisions,actual_collisions
```

`predictions.csv` (evaluation rows only):

```text
run_id,area_id,latitude,longitude,activity_band,feature_start,feature_end,forecast_start,forecast_end,past_complaints,past_collisions,actual_collisions,persistence_prediction,collision_only_prediction,combined_prediction
```

`metrics.csv`:

```text
run_id,model,scope,forecast_start,activity_band,area_week_rows,mae,top_k,top_k_collisions,citywide_collisions,eligible_collisions,top_k_coverage_pct,eligible_coverage_pct
```

Models: `persistence`, `collision_only`, `combined`. Scopes: `week`, `overall`, `activity_band`. Week rows have `forecast_start`; band rows have `activity_band`. Overall rows have neither. Band coverage fields are blank, not zero. `top_k=50` is the requested count even for smaller universes. Blank coverage also means zero denominator. Pooled coverage divides summed weekly numerators by summed citywide denominators.

`run_metadata.json` provides schema version, run ID, UTC creation time, cleaned-input SHA-256s, windows, model settings, source audit, universe size, and limitations. UI should validate hashes and matching run IDs before presenting current results. Features have no run ID column by the agreed schema; the metadata records their SHA-256 along with the other CSVs.

### Tiny synthetic interface sample — NOT findings

The following is only a parsing/layout example, not a production output or evaluation result. Never display it as research data or put it in production downloads:

```csv
run_id,area_id,latitude,longitude,activity_band,feature_start,feature_end,forecast_start,forecast_end,past_complaints,past_collisions,actual_collisions,persistence_prediction,collision_only_prediction,combined_prediction
SYNTHETIC-UI-EXAMPLE,cell_1170_9010,40.699,-73.991,zero,2025-08-25,2025-09-01,2025-09-01,2025-09-08,0,0,0,0.0,0.4,0.5
```

## Integration requests (outside backend ownership)

- UI owner: replace radius/old ranking controls with forecast week, model, and count measure; read `predictions.csv`, `metrics.csv`, and metadata. Do not use `selected_targets.csv` / `evaluation.csv`, recalculate competing metrics, or silently substitute old results.
- Shared-file owner: add narrow `.gitignore` exceptions for only `features.csv`, `predictions.csv`, `metrics.csv`, and `run_metadata.json` under `data/processed/` if the team elects to track demo outputs. Current policy ignores them; this backend does not edit `.gitignore` or force-add artifacts.
- Methodology owner: update README/setup/presentation to the new command and the actual exploratory results. No exposure, causality, statistical significance, or guaranteed complaint benefit claim.

## Fixed design

Longitude-first pyproj transforms with `always_xy=True`: WGS84 to EPSG:32618. Cells use `floor(easting/500)` and `floor(northing/500)`; IDs are `cell_{cell_x}_{cell_y}`. Map coordinates are inverse-transformed projected cell centers. Eligible cells are the union of complaint and collision cells from `[2025-08-01, 2025-08-08)` only. Activity bands use collision counts from that interval only: zero, one, two_plus.

Training forecast starts: August 8, 15, 22. Evaluation forecast starts: September 1, 8, 15, 22. Each has a preceding seven-day feature interval and subsequent seven-day target interval. The trained pipelines apply log1p, StandardScaler, and PoissonRegressor(alpha=1.0, max_iter=1000), with default intercept. Fit only the three training windows, then freeze the scaler and fitted model. Later September observations may enter later feature windows; this is weekly updated historical forecasting.

All three models use the identical evaluation rows. Rank per model/week by expected count descending, then string area ID ascending, with no spatial spacing rule. Coverage denominators include all mapped citywide target-week collisions, including cells outside the frozen eligible universe. September 29–30 are not evaluated.
