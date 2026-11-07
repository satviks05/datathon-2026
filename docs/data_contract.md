# Seven-day linear regression backend: UI handoff

The current goal is ordinary linear regression predicting the number of injury
OR fatal crashes near a reported-signal area during the NEXT SEVEN DAYS. This
supersedes the original combined-percentile scoring proposal. The application
uses Streamlit, Python modules, and the existing cleaned CSV snapshots.

## Ownership and existing path mapping

| Owner/scope | Paths |
| --- | --- |
| Backend | `src/**`, `tests/backend/**`, `requirements.txt`, `.gitignore`, this document, production analysis CSVs and manifest |
| Existing cleaner, reused unchanged | `pipelines/clean_data.py`, `pipelines/requirements.txt` |
| Existing cleaned inputs, reused unchanged | `data/processed/signal_complaints_clean.csv`, `data/processed/collisions_clean.csv` |
| UI owner, do not edit | `app.py`, `ui/**`, `.streamlit/**`, `tests/ui/**`, `docs/ui.md` on `feature/streamlit-ui` |
| Original challenge inputs, unchanged | `data/raw/**` |

Backend branch: `codex/backend`. The starter `docs/DATA_CONTRACTS.md` describes
the earlier four-person project. This file is the current backend interface;
the UI owner must update their adapter for the regression contract below.

## Install, validate, reproduce

Python 3.10-3.13 (3.13 tested), from repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests/backend -v
python -m src.build
```

Backend computations require pandas and NumPy only. Ordinary least squares is
fit with `numpy.linalg.lstsq`, including an intercept; no HTTP service or database.
Root requirements also include the UI owner's Streamlit/pydeck dependencies.
After UI integration, launch with `python -m streamlit run app.py`.

## UI controls and interpretation

- Daily inclusive start/end dates, August 1 through September 30, 2025.
- Default history: August 25-31, 2025. Forecast: September 1-7.
- Selected dates are the HISTORY/EVIDENCE window. Prediction is always the
  seven calendar days immediately AFTER its end. Show both periods explicitly.
- Radius: 100, 125, 150, 175, 200, 225, 250 m; default 250.
- Strategy: `linear_regression` (default), `crash_first`, `complaint_first`.
  The latter two are transparent ranking baselines. `combined` is no longer used.
- Shortlist size (`budget`): 25, 50 (default), 100, a scenario count rather than
  an estimate of actual staff capacity. Fixed 250 m minimum target spacing.
- Update after slider release; use one backend bundle for all displayed results.
- Model coefficients use August-complete labels and remain frozen in September.
  For an end date before August 31, return observed layers and counts, but no
  regression shortlist: training would otherwise use future information.
- Candidate IDs and coordinates are fixed from August-reported locations.
  This is not an as-of replay before the candidate catalog was known, and it does
  not cover September-only reported locations or every NYC signal.
- If a forecast extends beyond September 30, predictions remain available but
  actual outcomes/coverage are missing (`None`/NaN), never zero-filled.

## Public interface

```python
from src.api import (
    get_analysis, get_area_evidence, get_config,
    get_historical_evaluation, get_model_metrics, get_data_quality,
)

settings = get_config()
result = get_analysis(
    start_date="2025-08-25", end_date="2025-08-31",
    radius_m=250, strategy="linear_regression", budget=50,
)
shortlist = result["selected_targets"]
if not shortlist.empty:
    evidence = get_area_evidence(
        shortlist.iloc[0]["target_id"], 250,
        start_date="2025-08-25", end_date="2025-08-31",
    )
```

`get_analysis` returns:

| Key | Value |
| --- | --- |
| `selected_targets` | Ranked/spaced shortlist for the exact selected controls |
| `targets_scored` | All fixed candidates with selected-history counts and predictions |
| `reports` | All source signal reports in the history dates, for map layers |
| `collisions` | All source collisions in the history dates, with `occurred_at` and `has_harm` |
| `comparison` | Three strategies' unique NEXT-seven-day coverage when fully observed; missing otherwise |
| `evaluation` | Separate fixed weekly September backtest, always using seven-day histories |
| `model` | Fitted coefficients, intercept, training cutoffs, units, and limitations; `None` before August 31 |
| `metadata` | Controls, history/forecast dates, input version, candidate scope, availability flags and explanatory notes |

Target columns:

```text
target_id,latitude,longitude,radius_m,strategy,signal_report_count,
harm_crash_count,score,top_descriptor,suggested_inspection,start_date,end_date,
persons_injured,persons_killed,cyclist_injured,raw_prediction,
predicted_harm_crashes_next_7d,signal_reports_per_7_days,
harm_crashes_per_7_days,forecast_start_date,forecast_end_date
```

The selected table adds `selected_rank` and `budget`.
Source counts/context are for selected HISTORY dates. `raw_prediction` is the
OLS estimate; `predicted_harm_crashes_next_7d` clips negative estimates to zero.
For `linear_regression`, `score` is that nonnegative expected-count estimate,
NOT a probability or percentile. Baseline scores remain average-rank percentiles.
Predictions are also supplied alongside baseline rows for reference.

`comparison` and `evaluation` columns:

```text
strategy,radius_m,budget,selected_count,history_signal_reports_covered,
future_harm_crashes_covered,future_harm_crashes_total,future_harm_coverage_pct,
start_date,end_date,forecast_start_date,forecast_end_date
```

Coverage percentages are 0-100. Each source event counts ONCE inside the union
of selected circles, even when circles overlap. Denominators include ALL
geolocated harm crashes in the forecast dates, including outside candidate areas.
Only interpret zero coverage when the corresponding denominator/outcomes exist.

## Changes required in the current UI branch

1. Add daily two-handle history dates and show the next-seven-day forecast dates.
2. Add `linear_regression` as default strategy; remove `combined`.
3. Call `get_analysis` for dynamic controls. Never use default-history export rows
   for another date range. Source layers, shortlist, counts, evidence, and comparison
   must consume matching dates/radius.
4. Pass both dates to evidence retrieval and include them in its cache key. The
   current month-only evidence adapter must change.
5. Replace old `september_harm_*` export expectations with the generic future
   fields above. The benchmark evaluates weekly forecasts, not full-month coverage.
6. Handle `metadata.model_available=False`, empty shortlists, and
   `metadata.outcomes_available=False`; reset dropped selections to the first target.
7. Label predicted counts clearly. Do not format them as a percentage or assert
   that they are calibrated probabilities. Show model metrics separately from
   shortlist coverage and keep fixed backtests separate from custom-date results.

## Evidence and additional helpers

The original evidence call remains supported:

```python
get_area_evidence(target_id, radius_m, period="august")
```

Period is `august` or `september`; explicit start/end keywords override period.
Return: `{"reports": DataFrame, "collisions": DataFrame}`. Original IDs and source
columns are preserved; add `distance_m`, parsed `occurred_at`, and `has_harm`.
Collisions include ALL outcomes. Filter `has_harm` for the harm-only map layer.
Default evidence is the full August month, so date-aware UI MUST pass its dates.

`get_config()` provides shared control values, default dates, forecast horizon,
training cutoff, method version, and content-based input version.

`get_historical_evaluation(radius_m=250, budget=50)` returns twelve rows:
three strategies across four complete September weeks. Histories are Aug 25-31,
Sep 1-7, Sep 8-14, Sep 15-21; outcomes are Sep 1-7, 8-14, 15-21, 22-28.
Sep 29-30 cannot form another complete nonoverlapping seven-day outcome week.

`get_model_metrics(radius_m=250)` returns MAE, RMSE and R2 by outcome week and
pooled across weeks, comparing OLS, previous-period harm rate, August training
mean and always-zero predictions. The zero baseline exposes count sparsity.
Report both error metrics: a sparse-data model can improve RMSE without beating
zero on MAE. R2 can be negative; zero-variance target R2 is missing, not fabricated.

`get_data_quality()` returns audited counts, month splits, exclusions when raw
inputs are present, missing borough counts and assumptions. Missing borough rows
remain usable because their coordinates are present.

## Linear regression design and time separation

Features, each normalized to a seven-day equivalent:

```text
signal_reports_per_7_days = history_report_count * 7 / history_days
harm_crashes_per_7_days = history_harm_crash_count * 7 / history_days
raw_prediction = intercept + b_reports * signal_reports_per_7_days
                           + b_crashes * harm_crashes_per_7_days
```

Training examples use fixed August candidate areas and 7/14/21-day histories.
For each supported history length and each possible August history endpoint,
include a row only when ALL seven future label days end by August 31. The latest
training history ends August 24, and its label covers August 25-31. Training
data never use September observations. Fit one OLS model per radius; freeze
coefficients during every September backtest. Model features/horizon were fixed
before further comparisons; September has already been explored, so call this
exploratory out-of-time evaluation rather than an untouched final test.

There are 33 overlapping history windows per area, not 33 independent samples.
Nearby areas also overlap. Do not claim independent errors, confidence intervals,
causal coefficients, or generalization to unseen areas/cities. History lengths
outside 7/14/21 days are rate-normalized extrapolations of the training design.
No severity weights, closure/status, September-fitted coefficients, or future
crash counts enter feature construction or model fitting.

## Spatial design and caches

- 50 m grouping uses fixed local equirectangular coordinates at 40.7 N, -74.0
  longitude and mean Earth radius 6,371,008.8 m. This approximate local grouping
  grid is not a surveyed intersection grid. All radius/spacing decisions use
  spherical haversine meters, never Euclidean degree distances.
- Candidate representative: most frequently reported actual August coordinate
  in a cell; ties by ascending latitude then longitude. Target IDs encode cells.
- Angular buckets prefilter events; exact haversine decides membership. Store
  only event IDs/distances within 250 m. Daily count arrays speed model fitting.
- Radius boundaries are inclusive; targets less than 250 m apart are suppressed.
  Numerical tolerance is 0.0000001 m. Rank ties use ascending string target ID.
- Cache memberships and models, then calculate date/radius predictions as needed.
  No exhaustive hardcoding of slider results. Query caches are bounded.
- UI cache keys must include dates, radius, strategy, budget, and
  `get_config()["input_version"]`; evidence adds target ID. SHA-256 covers input
  contents, method source, configuration and runtime versions. Restart UI after backend code changes.
- Public result DataFrames/dictionaries are copied; one UI session cannot modify
  another session's cached data.

## Generated outputs

`python -m src.build` writes:

- `targets_scored.csv`: default Aug 25-31 history, all targets/radii/strategies.
- `selected_targets.csv`: matching spaced shortlists for all three budgets.
- `evaluation.csv`: 252 weekly radius/strategy/budget/outcome-week combinations.
- `model_metrics.csv`: 140 rows: seven radii, four estimators, four weeks plus pooled.
- `model_predictions.csv`: complete per-area weekly forecast/actual audit, generated locally.
- `analysis_manifest.json`: provenance signature, configuration, all seven fitted
  model coefficients, runtime versions, training/evaluation cutoffs, exclusions and output row counts.

Small final handoff snapshots are tracked; the larger per-area prediction audit
is ignored but reproducible. Inputs and the original cleaner are unchanged.

Reports are not verified defects; proximity does not establish signal-caused
crashes. Predicted counts are not crash probabilities, emergency repair urgency,
traffic-exposure-adjusted risk, or evidence of injuries prevented. Larger circles
mechanically include more events. Show weak performance and baseline tradeoffs
honestly; the event rewards defensible analysis rather than model complexity.
