# August-only training -> frozen September forecasts

The goal is to use August 2025 ONLY to predict September signal-complaint and
collision counts, then compare those forecasts with September actuals on the map.
Three ordinary least-squares count models predict signal reports, ALL collisions,
and injury-OR-fatal collisions. This replaces the earlier rolling seven-day design.
Earlier weekly metrics and shortlist counts do not describe this version.

## Ownership and setup

Backend branch: `codex/backend`. Backend owns `src/**`, `tests/backend/**`, root
`requirements.txt`, `.gitignore`, this contract and production analysis files.
The original cleaner in `pipelines/` and raw/source CSVs are unchanged.
The UI owner owns `app.py`, `ui/**`, `.streamlit/**`, `tests/ui/**`, `docs/ui.md`.
The older `docs/DATA_CONTRACTS.md` describes the starter project.

Python 3.10-3.13 (3.13 tested), from repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests/backend -v
python -m src.build
```

The existing local `.venv` is ready. Backend calculations use pandas and NumPy;
OLS uses `numpy.linalg.lstsq` with an intercept. No database or HTTP service.
After UI integration, launch `python -m streamlit run app.py`.

## Physical data separation

`python -m src.build` creates:

| Directory | Signal reports | Collisions | Purpose |
| --- | ---: | ---: | --- |
| `data/processed/august_train/` | 3,276 | 6,777 | Training and forecasting ONLY |
| `data/processed/september_actuals/` | 2,843 | 7,289 | Observed comparison ONLY |

Each directory contains `signal_complaints_clean.csv` and `collisions_clean.csv`.
Dates and original IDs are disjoint. Training complaints omit `closed_date` and
`status`, which could reflect later actions. Source snapshots retain all fields.
No September records or eventual closure/status values enter model fitting.
The loader supplies an empty closure column when reading the reduced train schema.

The training command works with ONLY the August directory:

```powershell
python -m src.train --train-dir data/processed/august_train --output-dir data/processed
```

It requires no September file, writes `september_forecasts.csv` and
`forecast_manifest.json`, and does not calculate actuals or accuracy. Optional
`--daily` writes the larger `september_daily_forecasts.csv` audit locally.

## Forecast design

For each fixed August candidate area and radius, construct daily August counts.
For each August 8-31 target day, features contain the PREVIOUS seven days' mean
report count, mean all-collision count, mean harm-crash count, and whether the
target day is a weekend. Fit three OLS targets using August labels only.

```text
expected_daily_count = intercept
    + b_reports * previous_7d_signal_reports_per_day
    + b_collisions * previous_7d_collisions_per_day
    + b_harm * previous_7d_harm_crashes_per_day
    + b_weekend * forecast_day_is_weekend
```

On August 31, freeze predictions for EVERY September day. September 1 uses
August 25-31 history. September 2 uses six August days plus the MODEL'S September 1
prediction. Continue recursively through September 30, using predictions in later
lags, NEVER September actuals. Calendar dates are known in advance; no outside
analytical dataset is added. Coefficients remain frozen throughout September.

Daily estimates are clipped to zero, with harm estimates capped at all-collision
estimates; raw linear estimates are retained. These are expected area-level counts,
not probabilities or predictions of exact event locations/times. Recursive errors
can accumulate over a month. There are only 24 overlapping August target days per
area, and nearby areas overlap. Do not describe rows as independent samples or
claim calibrated intervals, causality, or generalization to unseen areas/cities.

## Controls and map behavior

- Start/end sliders select September FORECAST/COMPARISON dates, inclusive, daily.
  Default September 1-30. They no longer select a rolling training/history window.
- Radius: 100, 125, 150, 175, 200, 225, 250 m; default 250.
- Strategies: `linear_regression` (default), `crash_first`, `complaint_first`.
- Shortlist (`budget`): 25, 50 (default), 100 review areas, not a crew estimate.
- Update after release. Sum frozen daily predictions for the selected dates,
  rerank areas, apply fixed 250 m spacing, and retrieve matching actual records.
- Candidate locations come ONLY from August signal reports and stay fixed.
  September-only report locations remain visible as actuals but are not new targets.
- August observations can be browsed with explicit August API dates; no September
  prediction is shown for an August period. UI forecast bounds come from config.
- Show predicted counts, observed counts, and their signed differences by area.
  Actual September data must NEVER determine forecast scores or selections.

## Public interface

```python
from src.api import (
    get_config, get_analysis, get_september_forecast,
    get_area_evidence, get_model_metrics, get_historical_evaluation,
)

settings = get_config()
result = get_analysis("2025-09-01", "2025-09-15", radius_m=175,
                      strategy="linear_regression", budget=50)

# Forecast-only: reads the August directory, requires NO September files.
forecast = get_september_forecast("2025-09-01", "2025-09-15", radius_m=175)

if not result["selected_targets"].empty:
    target = result["selected_targets"].iloc[0]["target_id"]
    august_evidence = get_area_evidence(target, 175, period="august")
    september_actuals = get_area_evidence(target, 175,
        start_date="2025-09-01", end_date="2025-09-15")
```

`get_analysis` returns:

| Key | Contents |
| --- | --- |
| `targets_scored` | August training counts + frozen predictions for selected dates; no September actuals |
| `selected_targets` | Spaced shortlist from those scores, with rank and budget |
| `area_comparison` | Predictions, actual counts and residuals for every candidate |
| `reports`, `collisions` | Original observed records in selected dates, for actual map layers |
| `comparison` | Strategy coverage for exactly the selected dates/radius/budget |
| `evaluation` | Separately labeled full-September comparison for same radius/budget |
| `model` | Three fitted equations, feature units, train/forecast cutoffs and limitations |
| `metadata` | Controls, provenance, totals, candidate scope and availability flags |

Target tables retain `target_id`, `latitude`, `longitude`, `radius_m`, `strategy`,
`score`, `top_descriptor`, `suggested_inspection`, `start_date`, `end_date`.
`signal_report_count`, `collision_count`, `harm_crash_count` and injury/fatality
context fields are fixed AUGUST TRAINING counts, not September actuals.
Additional fields: `predicted_signal_reports`, `predicted_collisions`,
`predicted_harm_crashes`, corresponding `raw_predicted_*`, `august_rate_*`,
`last_august_week_rate_*`, `forecast_start_date`, `forecast_end_date`, `training_end_date`.
Selected targets add `selected_rank` and `budget`.

For regression, `score = predicted_harm_crashes` for the selected September range.
Baselines rank August harm-crash/report counts using tied average percentiles.
Scores are not comparable probability scales across strategies.

`area_comparison` adds `actual_signal_reports`, `actual_collisions`,
`actual_harm_crashes`, and `residual_* = actual - predicted`. Positive residuals
mean underprediction. Do not sum area counts as a citywide total: circles overlap.

`get_area_evidence(target_id, radius_m, period="august", *, start_date=None, end_date=None)`
returns reports/collisions DataFrames with original IDs, source fields and distances.
Explicit start/end override the month. Collision records include all outcomes,
parsed `occurred_at`, `has_injury`, `has_fatality` and `has_harm`.

## Evaluation and exports

`get_model_metrics(radius_m=250, start_date="2025-09-01", end_date="2025-09-30")`
returns MAE/RMSE/R2 for each of the THREE outcomes versus August daily-rate,
last-August-week-rate and always-zero baselines. Metrics compare aggregate counts
per candidate for exactly the selected interval. Zero-variance target R2 is missing.
Show baselines honestly if regression loses. September has already been explored;
this remains exploratory historical evaluation, not an untouched final test.

`get_historical_evaluation(radius_m=250, budget=50)` compares all three strategies
for the entire September month. `comparison` fields include:

```text
strategy,radius_m,budget,selected_count,history_signal_reports_covered,
future_signal_reports_covered,future_collisions_covered,
future_harm_crashes_covered,future_harm_crashes_total,future_harm_coverage_pct,
start_date,end_date,forecast_start_date,forecast_end_date
```

Each actual source event counts once inside the union of selected circles.
Percentages are 0-100. The full-September harm denominator is all 3,400 geolocated
harm crashes, including those outside candidate areas. Full-month exports also
retain `september_harm_crashes_covered`, `september_harm_coverage_pct`,
`september_harm_crashes_total`, `august_signal_reports_covered` compatibility fields.

`python -m src.build` writes the split datasets, monthly `september_forecasts.csv`,
`forecast_manifest.json`, `targets_scored.csv`, `selected_targets.csv`,
`evaluation.csv` (63 configurations), `model_metrics.csv` (84 rows), and
`analysis_manifest.json`. `model_predictions.csv` is a generated local audit joining
per-area full-month predictions to actuals. All old seven-day exports are replaced.
Manifests record input/runtime/source versions, split counts, model coefficients,
periods and `september_observations_used_for_predictions=False`.

## UI owner integration changes

1. Add regression as default; remove the earlier `combined` strategy.
2. Use September date bounds/defaults from `get_config`; source August bounds are
   available separately. Label training and forecast periods explicitly.
3. Call `get_analysis` for committed control changes. Use `area_comparison` for
   prediction/actual/difference area layers and `reports`/`collisions` for actual points.
4. Show all three forecasts; keep harm-crash forecast as the review-priority score.
5. Pass selected dates to September evidence and metrics; show August evidence
   separately. Update the existing month-only evidence cache signature.
6. Reset a selected target that disappears, handle empty/unavailable results, and
   never substitute another configuration's CSV rows.
7. Old `predicted_harm_crashes_next_7d` fields and weekly metrics are obsolete.
   Use `predicted_harm_crashes` and the selected forecast dates instead.

## Spatial design and cache/version rules

50 m grouping uses approximate local equirectangular coordinates at 40.7 N,
-74.0 longitude, mean Earth radius 6,371,008.8 m. Radius and spacing use haversine
meters. Representatives are actual August coordinates, most frequent per cell,
ties by ascending latitude/longitude. Rank ties use ascending string target ID.
Radius boundaries are inclusive; suppress targets less than 250 m apart, with
0.0000001 m numerical tolerance. Counts deduplicate original IDs.

Cache spatial memberships, August-only fitted models and frozen daily forecasts;
date controls aggregate them, rather than hardcoding every final view. UI cache
keys include dates, radius, strategy, budget and `get_config()["input_version"]`;
evidence adds target ID. Hashes cover inputs, code, config and runtime versions.
Restart Streamlit after imported backend code changes. Rebuild derived files when
source inputs change. Public outputs are copies, safe for per-session UI transforms.

Reports do not verify defects; proximity does not establish signal-caused crashes.
Forecasts do not establish traffic-exposure-adjusted risk, inspection effectiveness,
crew shortages, live emergency urgency, or injuries prevented. Larger circles
mechanically include more events. Make weak predictive results visible.
