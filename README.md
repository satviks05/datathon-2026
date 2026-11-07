# NYC in Motion: Signal Complaints & Collisions

**Do traffic-signal complaints improve predictions of subsequent collision counts beyond recent collision history alone?**

A Rutgers Data Science Club Datathon 2026 project using NYC 311 signal reports and NYPD collision records from **August–September 2025**. The Streamlit dashboard combines an observed-event concentration map with reproducible forecasting comparisons and an exploratory week-over-week complaint-change experiment.

**Main finding:** adding complaint volume modestly lowers average prediction error, but does not improve top-50 collision coverage. Adding weekly complaint change produces a further negligible error reduction in a separate matched comparison. These are exploratory historical findings, not evidence that signal complaints cause crashes.

## Run the dashboard

Use **64-bit Python 3.12**. Python 3.15 caused dependency-installation failures for this project. Saved real-data outputs are committed, so you can launch without retraining.

### Windows PowerShell

```powershell
git clone https://github.com/satviks05/datathon-2026.git
cd datathon-2026
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt -r ui/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
git clone https://github.com/satviks05/datathon-2026.git
cd datathon-2026
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-model.txt -r ui/requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open **http://localhost:8501** if the browser does not open automatically. Stop the app with **Ctrl+C**. Internet access is needed for the CARTO basemap; no API key is required for this map setup.

## What the app does

| View | Purpose |
| --- | --- |
| **Observed map** | Explore real complaints in blue and collisions in orange. Area circles grow with recorded counts; individual-event mode shows dates, complaint type/status, and collision outcomes. |
| **Model comparison** | Compare weekly/pooled MAE, top-50 citywide collision coverage, and errors across historical activity bands. |
| **Prediction table** | Inspect prior-week inputs, all three model predictions, and observed outcomes by area; download real model rows. |
| **Complaint change** | Compare weekly complaint change against baselines refitted on the same available training windows. |
| **Data & method** | Review the analytical design, exclusions, limitations, and run provenance. |

The selected **Evaluation week** controls model outcomes. The observed map shows the **preceding seven days**, as labeled in the app. Its approximately 500 m display bins are presentation-only and are separate from the model's exact projected grid.

In the area-count map, circle area represents event count: five events have five times the area of one event, about 2.2 times the diameter. Sizes use the same rule across dates and datasets, capped above 100 events. Hover a circle for counts and date ranges; switch to individual events for record details. Complaint status is a source-snapshot attribute, not proof of historical repair.

There is no predicted-count map. Model outputs are kept in tables and evaluation panels, separate from observed event concentrations. Development fixtures are explicitly labeled, opt-in, and excluded from downloads and research findings.

## Data and exclusions

Only the two supplied challenge datasets are analytical inputs:

| Dataset | Source records | Project scope | Mapped records |
| --- | ---: | --- | ---: |
| [NYC 311 Service Requests](https://data.cityofnewyork.us/d/erm2-nwe9) | 34,676 DOT reports | 6,534 **Traffic Signal Condition** reports | 6,119 |
| [NYPD Motor Vehicle Collisions](https://data.cityofnewyork.us/d/h9gi-nx95) | 14,526 collisions | All recorded collision outcomes | 14,066 |

- Signal complaints: 415 excluded for unusable coordinates, about **6.35% of signal reports**. Other DOT categories are outside scope, not coordinate exclusions.
- Collisions: 119 missing coordinate pairs and 341 zero-coordinate pairs excluded.
- Original files remain in `data/raw/`; verified cleaned snapshots are in `data/processed/`.
- The cleaner uses a broad coordinate box, not an NYC boundary polygon or street-level location verification. Missing boroughs do not invalidate otherwise usable coordinates.

See [data sources](docs/DATA_SOURCES.md) and [cleaning instructions](pipelines/README.md). Weather and Citi Bike are not used.

## Main forecasting method

**Outcome:** the number of ALL mapped collisions in each eligible area over the next seven days, not injury-only crashes or per-traveler risk.

1. Project coordinates to **EPSG:32618** and create fixed **500 m cells**.
2. Freeze eligible areas using cells with a complaint **or** collision during **August 1–7**. This yields **1,180 cells**. Retain every cell in every window, including zero-event rows.
3. Use the preceding seven days' complaint and collision counts as historical features.
4. Train on target weeks **August 8–14, 15–21, and 22–28**: 3,540 area-week rows.
5. Evaluate on **September 1–7, 8–14, 15–21, and 22–28**: 4,720 rows. September 29–30 are outside complete target weeks.

| Model | Inputs / rule |
| --- | --- |
| Persistence | Predict next week equals the previous week's collision count. |
| Collision-only | Previous week's collision count. |
| Combined | Previous week's collision count and signal-complaint count. |

Both trained models use `log1p` count features → `StandardScaler` → `PoissonRegressor(alpha=1.0, max_iter=1000)`. Preprocessing and fitted parameters use August training rows only and remain frozen. Each September forecast may use earlier September observations available before its start: this is **weekly updated forecasting**, not an August-only prediction of the entire month.

Dates use half-open windows and a common NYC local-clock assumption. Crash timestamps combine `crash_date` and `crash_time`. Closure dates and current complaint status are not predictors.

## Main results

Pooled evaluation across the four September weeks, from [the committed metrics](data/processed/metrics.csv):

| Model | MAE ↓ | Top-50 citywide collision coverage ↑ |
| --- | ---: | ---: |
| Persistence | 0.937076 | 5.5107% |
| Collision-only | 0.845225 | 5.5107% |
| Combined | 0.841684 | 5.2461% |

**MAE** is the average absolute difference between predicted and observed collisions per area-week. Lower is better. The combined model reduces MAE by approximately **0.42% relative to collision-only**, but its coverage is approximately **0.26 percentage points lower**. Complaint counts add little predictive value in this experiment and do not improve this shortlist targeting measure.

**Top-50 coverage** ranks eligible cells by predicted count, takes 50 per week, and divides actual collisions in those cells by **all mapped citywide collisions that week**. It is not prediction accuracy or the probability of a crash. Ties use area ID. Pooled coverage divides pooled collision counts, rather than averaging weekly percentages.

There are **6,805** mapped citywide collisions in the four target weeks; **4,343** fall inside eligible cells. Thus eligible-universe coverage is **63.82%**, the maximum available within this study universe before selecting a shortlist. Activity-band MAE provides an additional comparison using August 1–7 collision counts: zero, one, or two-plus.

## Exploratory complaint-change add-on

```text
weekly complaint change = last 7 days' complaints − preceding 7 days' complaints
```

Fourteen days of history are required. The first main-model training window lacks that history, so **all experiment models are refitted on the same two remaining August windows**, not compared as if their training matched the original models.

| Matched experiment model | MAE ↓ | Top-50 coverage ↑ |
| --- | ---: | ---: |
| Collision-only | 0.844359 | 5.5107% |
| Complaint level | 0.840884 | 5.2755% |
| Complaint level + weekly change | 0.840650 | 5.2755% |

Adding change lowers MAE by approximately **0.028% versus complaint level**, with unchanged coverage: a negligible exploratory gain. Signed `log1p` preserves negative changes. This feature is a comparison of two weeks, **not a reliable long-term baseline or verified influx of defects**.

The original models and canonical outputs are preserved. Experiment code, saved outputs, methodology, and limitations are in [the complaint-change experiment](experiments/complaint_change/README.md).

## Reproduce the analysis

After installing dependencies, run from the repository root. Use `.venv/bin/python` instead of the Windows executable on macOS/Linux.

```powershell
# Optional: reproduce the existing cleaning step.
.\.venv\Scripts\python.exe pipelines/clean_data.py data/raw data/processed

# Main forecasting features, model fitting, evaluation, and exports.
.\.venv\Scripts\python.exe -m src.build_forecast

# Separate exploratory complaint-change experiment.
.\.venv\Scripts\python.exe -m experiments.complaint_change.build
```

Main exports: `data/processed/features.csv`, `predictions.csv`, `metrics.csv`, and `run_metadata.json`. Add-on exports remain separately under `experiments/complaint_change/results/`. Metadata records source hashes, settings, run IDs, and provenance; the UI reads saved results rather than training or calculating competing evaluation metrics.

If model inputs differ from the saved run, the UI flags the model panels and disables their downloads. A verified Windows CRLF-to-LF newline-only match is accepted with a provenance note. Other differences require checking/restoring inputs or rebuilding results; the observed map continues to read the current cleaned records.

### Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/backend -v
.\.venv\Scripts\python.exe -m unittest experiments.complaint_change.test_experiment -v
.\.venv\Scripts\python.exe -m unittest discover -s tests/ui -v
```

Checks cover source IDs and windows, eligibility and zero-event rows, training/evaluation separation, signed changes, metric consistency, reproducibility, hashes, map encodings, controls, and real-export startup. Streamlit AppTest does not visually verify browser rendering; inspect the map locally as well.

## Practical use and limitations

The prototype supports exploring records and prioritizing **further safety investigation**. It does not establish an emergency-dispatch policy, a verified signal defect, repair effectiveness, or a staffing shortage.

- Only two months of data and a few weekly periods are available. September was already explored; evaluation is exploratory, not an untouched final test.
- Traffic volume, population activity, road geometry, and a complete signal inventory are absent. Busy areas can generate both complaints and collisions. Predictive associations are not causal effects or exposure-adjusted risk.
- Repeated complaints may concern the same condition; reporting behavior differs across places.
- Missing/unusable locations are excluded, and fixed early-history eligibility omits later-active areas. Cells with no observed events are not necessarily safe.
- Neighboring cells and successive weeks are related; area-week rows are not independent experiments.
- A lower MAE does not automatically imply better hotspot targeting. Report the coverage tradeoff and avoid claims of lives saved or statistically significant improvements.

## Repository guide and team workflow

| Path | Purpose |
| --- | --- |
| `app.py`, `ui/`, `.streamlit/` | Streamlit interface, observation map, adapters, and theme |
| `src/forecast_*.py`, `src/build_forecast.py` | Current weekly forecasting backend |
| `experiments/complaint_change/` | Isolated add-on, saved results, and checks |
| `pipelines/`, `data/raw/`, `data/processed/` | Cleaner, source snapshots, and current exports |
| `tests/backend/`, `tests/ui/` | Backend and interface checks |
| `docs/model_contract.md`, `docs/ui.md` | Current backend and UI run/integration details |

The team uses two code owners (backend and UI) plus a methodology/presentation owner. Work on separate branches, coordinate shared contracts, and review/merge one PR at a time. Update from `main` before integration.

[Backend contract](docs/model_contract.md) · [UI guide](docs/ui.md) · [Project handoff](docs/PROJECT_HANDOFF.md)

The handoff documents the initial weekly-model specification; later UI and optional experiment details are documented in the guides above. Older radius-based modules (`src/engine.py`, `src/ranking.py`, `src/regression.py`), ranking CSVs, and four-person starter instructions are retained history, not the current workflow.
