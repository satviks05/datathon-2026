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

## Integration status and owner requests

- The forecast UI is now on `main` and merged into this branch. It reads the canonical predictions, metrics, and metadata without invoking the old ranking API. The real bundle passes its schema/hash checks with no stale or fixture warning. Real week/model/measure changes, top-50 filtering, pooled comparison, and enabled downloads were checked with Streamlit AppTest. This backend does not modify the UI owner's implementation.
- Shared-file owner: add narrow `.gitignore` exceptions for only `features.csv`, `predictions.csv`, `metrics.csv`, and `run_metadata.json` under `data/processed/` for discoverability. The four stable, backend-owned demo exports are explicitly committed; `.gitignore` is unchanged and no unrelated generated artifacts are added. Future changes to these tracked files are already visible to Git.
- Methodology owner: update README/setup/presentation to the new command and the actual exploratory results. No exposure, causality, statistical significance, or guaranteed complaint benefit claim.

## Fixed design

Longitude-first pyproj transforms with `always_xy=True`: WGS84 to EPSG:32618. Cells use `floor(easting/500)` and `floor(northing/500)`; IDs are `cell_{cell_x}_{cell_y}`. Map coordinates are inverse-transformed projected cell centers. Eligible cells are the union of complaint and collision cells from `[2025-08-01, 2025-08-08)` only. Activity bands use collision counts from that interval only: zero, one, two_plus.

Training forecast starts: August 8, 15, 22. Evaluation forecast starts: September 1, 8, 15, 22. Each has a preceding seven-day feature interval and subsequent seven-day target interval. The trained pipelines apply log1p, StandardScaler, and PoissonRegressor(alpha=1.0, max_iter=1000), with default intercept. Fit only the three training windows, then freeze the scaler and fitted model. Later September observations may enter later feature windows; this is weekly updated historical forecasting.

All three models use the identical evaluation rows. Rank per model/week by expected count descending, then string area ID ascending, with no spatial spacing rule. Coverage denominators include all mapped citywide target-week collisions, including cells outside the frozen eligible universe. September 29–30 are not evaluated.

## Verified real-data results

The frozen union contains **1,180 cells**: 215 early-history zero-collision cells, 629 one-collision cells, and 336 two-plus cells. Outputs contain 8,260 feature rows (3,540 training), 4,720 prediction rows, and 24 metric rows. Both Poisson fits converged with the specified settings, in 3 and 5 iterations respectively.

| Model | Pooled MAE | Top-50 collisions | Top-50 citywide coverage |
| --- | ---: | ---: | ---: |
| persistence | 0.937076 | 375 | 5.510654% |
| collision_only | 0.845225 | 375 | 5.510654% |
| combined | 0.841684 | 357 | 5.246143% |

Combined lowers MAE by **0.003541 (0.41894%)** versus collision-only and **0.095392 (10.17978%)** versus persistence. Its top-50 coverage is **0.26451 percentage points worse** than both references. This is a small MAE gain with a worse concentration ranking; complaints do not improve every outcome. No tuning or September model selection was performed.

| Forecast start | persistence MAE | collision_only MAE | combined MAE | Citywide collisions | Eligible collisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025-09-01 | 0.885593 | 0.852143 | 0.849489 | 1,744 | 1,090 |
| 2025-09-08 | 0.952542 | 0.844531 | 0.842308 | 1,673 | 1,070 |
| 2025-09-15 | 0.973729 | 0.874180 | 0.868837 | 1,747 | 1,147 |
| 2025-09-22 | 0.936441 | 0.810046 | 0.806102 | 1,641 | 1,036 |

The eligible universe contains **4,343 of 6,805** mapped evaluation collisions (**63.82072%**). The other 2,462 remain in citywide coverage denominators. Weekly coverage is not averaged to obtain pooled coverage. Activity-band MAE is exported for every model: combined is 0.796122 (zero), 0.802232 (one), and 0.944693 (two_plus), versus collision-only 0.798882, 0.806058, and 0.948201.

Source audit: 34,676 raw 311 rows, of which 6,534 are signal reports; 28,142 other-category reports are outside scope. Coordinate filtering excludes 415 signal reports, leaving 6,119. Of 14,526 raw collisions, 119 lack coordinate pairs and 341 have zero pairs, leaving 14,066. Category exclusions and unusable coordinates are separate limitations.

## Manual event-ID audit

The following were checked directly against source CSV IDs, independently recomputing projected cell membership and half-open timestamp masks. `tests/backend/test_forecast_real.py` preserves these checks:

| Cell / forecast start | Prior complaint IDs | Prior collision IDs | Target collision IDs | Early band |
| --- | --- | --- | --- | --- |
| `cell_1165_8999` / 2025-08-15 | 65845573 | 4835362 | 4835338, 4836748 | one |
| `cell_1166_8999` / 2025-09-08 | 66069300 | 4839808, 4841555 | 4841465, 4841917 | zero |

The second cell legitimately acquires collisions later despite its fixed zero early-history band. Synthetic tests additionally cover exact midnight/end boundaries, zero-event rows, complaints-only and collisions-only eligibility, exclusion of later-only cells, training-only scalers, weekly feature updates, convergence errors, deterministic top-50 ties, citywide denominators, pooled weighting, absent bands, zero denominators, unchanged inputs, mixed/invalid metric rows, repeatability, and preservation of existing outputs on fit failure.

Exports are staged only after fitting/validation succeed, with metadata replaced last. A filesystem cannot atomically replace four ordinary files as a unit: consumers must validate both run IDs and the metadata's CSV SHA-256s to reject an interrupted or mixed publication. No model binaries are required by the UI.

Validation after merging the forecast UI from `main`: **37 backend tests** (17 new weekly-forecast tests, 20 legacy regression tests) and **18 forecast UI tests** passed, including the real-output startup/hash test (not skipped). A separately created Python 3.12 virtual environment installed only `requirements-model.txt` and reproduced all three canonical CSVs byte for byte; metadata matched except its UTC creation timestamp. No source CSVs, cleaner, root requirements, or shared documentation were edited; UI changes come solely from merging their owner's reviewed `main` commits.

A headless Chromium check loaded the real forecast dashboard with no page exceptions, switched to pooled comparison, and downloaded 1,180 September 1 prediction rows with the current run ID. Cell markers rendered, but CARTO background tiles were blocked by the managed network policy (`basemaps.cartocdn.com`, proxy HTTP 403). Full basemap verification requires applying the pending CARTO domain settings. No fixture or stale-data state was used for this check.
