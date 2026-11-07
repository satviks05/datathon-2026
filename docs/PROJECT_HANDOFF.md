# NYC complaint–collision forecasting: team handoff

**Status:** Agreed direction and proposed implementation contract for the three-hour prototype. Models and prediction exports are not implemented yet. Assign names below and confirm this contract before parallel coding.

**Repository:** https://github.com/satviks05/datathon-2026

## 1. What we are building

**Question:** Do past traffic-signal complaint counts improve predictions of subsequent collision counts beyond past collision counts alone?

Build a retrospective Streamlit map and model comparison using August–September 2025 NYC data. Compare two simple trained models and one persistence benchmark. The result can be that complaints help, do not help, or make predictions worse. Do not promise a winning model before evaluation.

**User:** A transportation analyst choosing locations for further safety investigation.

**Practical output:** A map and downloadable table of predicted weekly collision concentrations, accompanied by an honest comparison against crash history alone.

**Primary target:** ALL mapped police-reported collisions in an area in the next seven days. Injury/fatality prediction is outside the first build. Harm flags may remain available as source context.

**Complaint scope:** Traffic Signal Condition only. The cleaned complaint file does not contain all DOT complaint categories.

**Interpretation:** Predict event counts, not per-person or per-trip crash risk. Complaints are not verified defects. Proximity and prediction do not establish causation or show that a particular signal repair would prevent crashes.

## 2. Current repository: reuse versus replace

Already merged into `main`:

- Original inputs: `data/raw/nyc311_dot_aug_sep_2025.csv` and `nyc_crashes_aug_sep_2025.csv`.
- Existing cleaner: `pipelines/clean_data.py`.
- Verified snapshots: `data/processed/signal_complaints_clean.csv` (6,119 rows) and `collisions_clean.csv` (14,066 rows).
- Streamlit entry point: `app.py`; presentation modules: `ui/data.py`, `ui/map_view.py`.
- Theme: `.streamlit/config.toml`; UI dependencies: `ui/requirements.txt`.
- Existing UI tests: `tests/ui/test_dashboard.py`.

Reuse cleaned inputs, the app entry point, styling, map foundations, and relevant source-loading code. No second dashboard and no new cleaning pass.

**This handoff supersedes the earlier radius/inspection-ranking design for new work.** Replace its radius slider, three ranking strategies, 250 m spacing, August-only candidate grid, `selected_targets.csv` / `evaluation.csv` interface, and `get_area_evidence()` dependency. Those contracts do not describe the new model. Existing starter READMEs and `docs/DATA_CONTRACTS.md` / `docs/ui.md` need updating by their assigned owners.

## 3. People, file ownership, and branches

| Role | Assign name | Sole editing ownership | Branch |
| --- | --- | --- | --- |
| Backend coder | __________ | `src/**`, `tests/backend/**`, `data/processed/features.csv`, `predictions.csv`, `metrics.csv`, `run_metadata.json`, `requirements-model.txt`, `docs/model_contract.md` | `feature/forecast-backend` |
| UI coder / merge coordinator | __________ | `app.py`, `ui/**`, `.streamlit/**`, `assets/**`, `tests/ui/**`, `docs/ui.md` | `feature/forecast-ui` |
| Methodology and presentation | __________ | `README.md`, `docs/methodology.md`, `docs/pitch.md`, `slides/**`, updates to this handoff and superseded general documentation | `docs/forecast-pitch` |

Raw inputs, existing cleaned snapshots, and the cleaner are read-only during this build. Unlisted shared files need an agreed owner before editing. Do not put UI scoring logic in a second implementation. Request a change from the relevant owner instead of editing their files concurrently.

The UI coordinator merges PRs one at a time. The methodology member can contribute a documentation PR; the two code branches are the integration dependencies.

## 4. Exact spatial and time definitions

### Fixed geography

1. Project WGS84 coordinates to **EPSG:32618**, whose units are meters. Use `pyproj` with `always_xy=True` and longitude first.
2. Assign `cell_x = floor(easting / 500)` and `cell_y = floor(northing / 500)`. This fixes both the 500 m resolution and grid origin.
3. Define `area_id = "cell_{cell_x}_{cell_y}"`; keep it a string.
4. Display each cell at its center, transformed back to WGS84. Export `latitude` and `longitude` separately. GeoJSON, if needed, uses `[longitude, latitude]`.
5. Freeze the eligible area universe to cells containing at least one mapped complaint OR collision during **August 1–7, 2025**. Do not use September or later August outcomes to choose cells.
6. Generate a row for EVERY eligible cell in EVERY forecast window, including zero-complaint and zero-collision rows. Do not create only rows where an event occurred.

This defines a limited early-history study universe, not every NYC road or intersection. Later collisions outside it must be disclosed. Do not silently expand it using evaluation outcomes. The early universe is a simplicity choice with a coverage tradeoff, not a claim that other cells are safe.

### Time preparation

- Read `unique_key` and `collision_id` as strings.
- Parse `created_date` as complaint submission time.
- Build `occurred_at` from the date in `crash_date` PLUS `crash_time`.
- Treat timestamps as NYC local-clock time; these supplied months have no daylight-saving transition. No timezone offset is encoded in the CSVs.
- Use half-open windows: `start <= timestamp < end`. Sort actual timestamps when needed.
- Never use closure time, status, future counts, or future labels as predictors.

### Fixed seven-day windows

`forecast_start` names the first predicted day; `forecast_end` is exclusive. The seven preceding days supply both predictors.

| Split | Feature window, inclusive dates | Target window, inclusive dates |
| --- | --- | --- |
| train | Aug 1–7 | Aug 8–14 |
| train | Aug 8–14 | Aug 15–21 |
| train | Aug 15–21 | Aug 22–28 |
| evaluation | Aug 25–31 | Sep 1–7 |
| evaluation | Sep 1–7 | Sep 8–14 |
| evaluation | Sep 8–14 | Sep 15–21 |
| evaluation | Sep 15–21 | Sep 22–28 |

The fitted model and preprocessing are frozen before September evaluation. Later September forecasts legitimately use earlier September events that would have been available at that forecast date. **This is a weekly updated historical forecast, not a single forecast for all September made in August.** September 29–30 are outside complete seven-day target windows; do not extrapolate their counts into a full week.

Historical data may appear as an outcome in an earlier training row and a predictor in a later row. That is legitimate temporal reuse; leakage means using information that was unavailable at the forecast date.

## 5. Features and models

One row per eligible area per forecast window:

```text
past_complaints = distinct complaint IDs in the preceding seven days
past_collisions = distinct collision IDs in the preceding seven days
actual_collisions = distinct collision IDs in the next seven days
```

Zero-fill only genuinely empty area/time counts. Do not replace invalid timestamps or missing locations with made-up events or coordinates.

Use these model IDs everywhere:

| ID | Method | Predictors |
| --- | --- | --- |
| `persistence` | Predict next week equals last week | `past_collisions` |
| `collision_only` | Regularized Poisson regression | `log1p(past_collisions)` |
| `combined` | Same regression and settings | `log1p(past_collisions)`, `log1p(past_complaints)` |

For both trained models use scikit-learn `PoissonRegressor(alpha=1.0, max_iter=1000)` and a `StandardScaler` in a `Pipeline`. Log-transform counts before scaling; fit the scaler and model using training rows only. Leave intercept behavior at the default. Do not add area IDs as numeric predictors.

These are fixed starting settings, not an optimized choice. Report convergence problems rather than silently accepting them. No hyperparameter search or choosing a winning configuration based on September. Keep predictions as nonnegative floating-point expected counts; do not round before evaluation.

Training computation should be small. Spend time validating the feature table, not adding algorithms. Only three training windows and four evaluation windows are available, so conclusions are preliminary.

## 6. Evaluation definitions

Every model is evaluated on the SAME eligible area/week rows. Use chronological separation, not a random row split.

### Primary: MAE

```text
MAE = mean(abs(actual_collisions - prediction))
```

Report each evaluation week and pooled evaluation rows. Compare combined against BOTH `collision_only` and `persistence`.

```text
relative improvement (%) = 100 * (reference_MAE - combined_MAE) / reference_MAE
```

If reference MAE is zero, report the absolute difference and leave relative improvement undefined. Negative improvement means the combined model is worse. Also show absolute MAE so small relative gains are not overstated.

### Secondary: top-50 coverage

For each model/week, order cells by predicted count descending, break ties lexicographically by `area_id`, and take `min(50, eligible_area_count)` cells. No spacing filter is used in this grid model.

- Numerator: actual mapped collisions inside those cells during the target week.
- Primary coverage denominator: ALL mapped citywide collisions in that target week, including outside eligible cells.
- Also disclose `eligible_collisions / citywide_collisions`: the maximum coverage available within the study universe.
- Report coverage as 0–100 percent, and combined-minus-reference differences in percentage points.
- Cells do not overlap, so each collision contributes once. Preserve event IDs to check this.

For pooled coverage, sum weekly numerators and denominators; do not simply average weekly percentages. Top-50 sets can change between weeks.

### Busy-area check

Use early-history **Aug 1–7 collision counts only** to assign a fixed activity band per eligible cell: `zero` (0), `one` (1), or `two_plus` (2+). Report MAE per band and model. Empty bands remain absent or explicitly unavailable; never invent a statistic.

This checks whether gains are concentrated in historically busy areas. It does not control for actual traffic exposure or establish a causal complaint effect. An intercept-only or population-density comparison is not required for the first build.

September has already been explored. Call this **exploratory historical evaluation**, not an untouched final test. Do not claim statistical significance from four weekly comparisons or treat neighboring cells as independent experiments.

## 7. Canonical backend-to-UI files

All files live under `data/processed/`. The backend owns production generation; the UI reads outputs and never trains models or calculates competing metrics. UTF-8 CSV, no index column, ISO `YYYY-MM-DD` dates, string IDs, and finite numeric predictions/counts.

### `features.csv` — backend audit artifact

```text
area_id,latitude,longitude,activity_band,split,
feature_start,feature_end,forecast_start,forecast_end,
past_complaints,past_collisions,actual_collisions
```

Bounds ending in `_end` are exclusive. Uniqueness key: `(area_id, forecast_start)`. Split values: `train`, `evaluation`.

### `predictions.csv` — UI map/table input

Evaluation rows only, with one row per area/week:

```text
run_id,area_id,latitude,longitude,activity_band,
feature_start,feature_end,forecast_start,forecast_end,
past_complaints,past_collisions,actual_collisions,
persistence_prediction,collision_only_prediction,combined_prediction
```

Uniqueness key: `(area_id, forecast_start)`. Include zero-event rows. All three predictions must be populated. Counts are nonnegative integers; predictions are nonnegative floats. The UI must not fabricate missing predictions.

### `metrics.csv` — authoritative evaluation input

One row per `(model, scope, forecast_start, activity_band)`:

```text
run_id,model,scope,forecast_start,activity_band,area_week_rows,
mae,top_k,top_k_collisions,citywide_collisions,eligible_collisions,
top_k_coverage_pct,eligible_coverage_pct
```

- `model`: `persistence`, `collision_only`, `combined`.
- `scope`: `week`, `overall`, `activity_band`.
- For `week`, populate `forecast_start` and leave `activity_band` empty.
- For `overall`, leave `forecast_start` / `activity_band` empty; pooled MAE and pooled coverage use all evaluation rows/windows.
- For `activity_band`, leave `forecast_start` empty and populate the band; report pooled MAE and leave coverage fields blank (coverage is not being compared within bands).
- Blank means not applicable, never zero. Coverage is blank if its denominator is zero.
- `top_k` is 50 as the requested count, even if fewer eligible areas exist; record eligible area count in metadata.

### `run_metadata.json` — provenance and settings

At minimum:

```text
run_id, schema_version (=1), created_at_utc, input_sha256,
grid_crs (=EPSG:32618), grid_size_m (=500), eligible_area_count,
universe_start (=2025-08-01), universe_end_exclusive (=2025-08-08),
training_windows, evaluation_windows, model_settings, source_audit, limitations
```

Use the same `run_id` in every export. UI rejects mixed run IDs and unsupported schema versions. Generate files from one completed run, then publish them together; do not mix old metrics with new predictions. `source_audit` distinguishes category filtering from coordinate exclusions and records current input counts. Store input SHA-256 hashes for the two cleaned files.

The UI caches by file signatures AND run ID. If inputs no longer match the metadata hashes, show a stale-output warning and do not present results as current. The backend can keep fitted objects local; the UI does not need model pickle files.

### Commands to implement

Backend owner supplies one command:

```bash
python -m src.build_forecast
```

It reads existing cleaned snapshots, builds features, fits models, evaluates, and writes all outputs. Running it twice with unchanged inputs/settings must reproduce numerical outputs (run timestamp/ID can differ).

UI command remains:

```bash
python -m streamlit run app.py
```

Do not silently fall back to the old ranking exports. If model outputs are absent, show an explicit pending state. Development fixtures must be labeled and excluded from results/downloads.

**Artifact policy:** Current `.gitignore` ignores these generated outputs. During development, each coder can run the generation command. For the final three-hour demo, propose explicitly tracking the four small outputs above once stable so teammates can launch without training. The backend owner requests that narrow `.gitignore` change from the team; never force-add unrelated generated data, environments, or model binaries.

## 8. UI minimum scope

Reuse the current light theme and map-plus-table layout. Replace old inspection-ranking controls rather than adding a second workflow.

Controls:

- Forecast week: the four evaluation windows.
- Prediction model: combined, collision-only, persistence.
- Map measure: past signal complaints, past collisions, predicted collisions, observed target-week collisions.
- Optional: show top 50 only. Keep top-50 evaluation fixed; variable shortlist sizes are outside the first build.

Map:

- One marker per eligible 500 m cell, located at the cell center.
- Size and/or color shows the selected count, with a visible legend and units. Marker size is a display encoding, not a geographic evidence radius.
- Clearly distinguish preceding-week observations, future-week predictions, and retrospective target-week outcomes.
- Use the same visual scale when comparing models for the same count measure. Do not automatically imply that a brighter map means better model performance.
- Area selection can use the existing dropdown. Map-click selection is optional.

Area panel:

- Area ID, feature dates, forecast dates, past complaint and collision counts.
- All three predictions alongside observed target-week collision count.
- Expected-count wording; no calibrated crash-probability or risk label.

Comparison:

- Model MAE and top-50 coverage, by week and overall, from `metrics.csv`.
- Activity-band error table and citywide-versus-eligible coverage context.
- CSV download of displayed prediction rows and authoritative metrics.
- Explicit missing-data, training-period, study-universe, and exposure limitations.

## 9. Three-hour parallel schedule

| Elapsed time | Backend coder | UI coder | Methodology/presentation |
| --- | --- | --- | --- |
| 0–15 min | Confirm this contract; publish a tiny labeled sample export | Confirm paths/columns; map old UI elements to replacements | Confirm question, claims, and outcome definitions |
| 15–60 min | Build and manually verify features | Build updated controls/map/panels against sample files | Draft methods, limitations, and slide outline |
| 60–100 min | Fit models; generate real predictions and metrics | Connect real outputs as they arrive | Check actual results; start findings slides |
| 100–135 min | Check leakage, counts, and comparisons | Complete map, charts, downloads, and missing states | Select examples and explain gains or losses honestly |
| 135–160 min | Reproduce outputs from a clean environment | Coordinate merges; launch integrated app | Verify README and rehearse demo |
| 160–180 min | Fix critical bugs only | Freeze working demo and prepare submission | Finish pitch and submit with buffer |

Protect a complete data → model → evaluation → map path. No new algorithms, grid-resolution searches, radius controls, extra outcomes, or styling overhaul during this sprint.

**Fallback trigger:** If a verified feature table is not ready at minute 60, jointly switch to one snapshot: Aug 1–14 predictors → Aug 15–28 training target; Sep 1–14 predictors → Sep 15–28 evaluation target, keeping the early-history area universe. Label two-week counts, revise exports/metadata together, and acknowledge the weaker single-period evaluation. Do not let backend and UI silently implement different fallbacks.

## 10. Git and local setup

Use **Python 3.12, 64-bit**. The UI was tested on it. Python 3.15 caused the team's pandas installation failure. Do not install a compiler toolchain to work around that issue.

From the repo root, Windows PowerShell:

```powershell
git fetch origin
git switch main
git pull origin main
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ui/requirements.txt
# After backend requirements are committed:
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m src.build_forecast
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Backend dependencies should include pandas, numpy, scikit-learn, and pyproj with a tested compatible version set. Existing UI requirements already include pandas, Streamlit, and pydeck. Prefer compatible ranges or tested pins across both files. The backend module and dependency file above are deliverables, not present-day commands until implemented.

Create your branch from updated `main`:

```bash
git switch -c feature/forecast-backend
# UI owner uses feature/forecast-ui instead.
```

Before merging a PR:

```bash
git fetch origin
git merge origin/main
# Run your checks and resolve conflicts on your branch.
git add <specific-owned-paths>
git commit -m "Describe the completed change"
git push -u origin <your-branch>
```

Review and merge one PR at a time. Other owners then merge updated `origin/main` into their branches and check compatibility. Imports can break without a text conflict, so rerun the app after code merges. Commit local work or stash before switching branches. Do not overwrite uncommitted teammate work or force-push shared branches.

All AI sessions should receive this handoff plus: **“Read any file you need. Edit only my assigned files. Keep the agreed export schema. Request changes from the other owner when needed. Do not present fixtures as findings.”** GitHub commits/PRs are the coordination mechanism between sessions.

## 11. Acceptance checks

Backend:

- [ ] Manually verify at least two cells/windows against original event IDs.
- [ ] Check boundary timestamps, zero-event rows, fixed cell IDs, and projected meter units.
- [ ] Keep study-universe selection and fitted preprocessing independent of September outcomes.
- [ ] Every predictor timestamp precedes `forecast_start`; labels fall before exclusive `forecast_end`.
- [ ] All models use identical evaluation rows; predictions are finite and nonnegative.
- [ ] Top-50 ties are deterministic; citywide and eligible denominators are correct.
- [ ] Aggregate metrics agree with predictions; activity bands use early history only.
- [ ] Exports share run ID and input hashes; regeneration works from committed inputs.

UI:

- [ ] Existing old ranking widgets/fixtures no longer masquerade as this model.
- [ ] Week/model/measure changes update map, selected area, table, and labels consistently.
- [ ] Missing/malformed/mixed-run/stale outputs produce an understandable state.
- [ ] Predictions and retrospective actuals are clearly distinguished.
- [ ] Downloads contain the displayed settings and real output rows.
- [ ] Update old UI tests to the new contract; verify startup and control changes.

Story and submission:

- [ ] Report comparison against both persistence and collision-only models, including worse results.
- [ ] State signal-only scope, excluded coordinates, fixed eligible universe, and short historical window.
- [ ] No claim of per-traveler risk, verified signal causation, staffing shortage, or injuries prevented.
- [ ] Slides and dashboard use matching exported values and periods.
- [ ] README has actual dependency/build/launch commands; a teammate can reproduce the demo.
- [ ] Public GitHub submission and demo are ready before the event deadline.

## 12. Suggested presentation structure

1. Question: do complaints add predictive information beyond crash history?
2. Data and exclusions: signal-only 311 reports plus mapped collisions.
3. Method: fixed grid, historical weekly features, two trained models plus persistence.
4. Results: later-period MAE and top-50 coverage, with honest tradeoffs.
5. Map example: predicted count versus later observed count in a real area.
6. Use and limits: prioritize further investigation; no causal or exposure-adjusted risk claim.

**Success means a reproducible comparison and clear working demo, not a predetermined winning result.**
