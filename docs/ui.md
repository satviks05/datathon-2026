# Weekly collision-forecast UI

This replaces the earlier inspection-area/radius workflow. `docs/PROJECT_HANDOFF.md` is the authoritative specification. UI code does not train models, aggregate geographic features, or calculate MAE or coverage.

## Get and run this version

Python **3.12, 64-bit** is recommended. From the repository root, Windows PowerShell:

```powershell
git fetch origin
git switch feature/forecast-ui
git pull origin feature/forecast-ui
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ui/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Activation is unnecessary when invoking the virtual environment's Python directly. On macOS/Linux, use `python3.12 -m venv .venv`, then `.venv/bin/python` in place of the Windows executable. Stop Streamlit with Ctrl+C.

After the backend owner's revised implementation is available, its separate setup/build commands are:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m src.build_forecast
```

These depend on the new backend PR; the older `src.build`/`src.train` modules and legacy exports are not substitutes.

## State and demo readiness

The UI can be developed and checked before real model exports exist. In that case, startup shows an NYC-centered map of the real cleaned signal reports and collisions from the preceding week. The default **Area counts** view groups observations into approximately 500 m display bins, with larger circles indicating more events. **Individual events** shows equal-sized points for detailed inspection. Layer checkboxes and forecast-week selection update these observations. This is an explicitly labeled observation view; it does not invent predictions or metrics. Missing-file diagnostics and backend setup commands are tucked into **Forecast connection details**. Enable **Development: use example forecast** to inspect a tiny static fixture. All fixture values and locations are synthetic; warnings remain visible and downloads are disabled. Fixture JSON is explicitly marked and cannot be loaded as a production run.

The revised backend exports are now committed on `main` and integrated into this UI branch: 4,720 prediction rows, 1,180 eligible cells, four evaluation weeks, and 24 metric rows. The real-export schema/source-hash check and Streamlit startup test pass. All 20 UI tests pass with no skips. A visual browser demo is still required; Python/AppTest checks do not verify tile rendering. No UI PR merges are performed automatically; reviewed changes need team approval.

## Controls and behavior

- **Forecast week:** September 1–7, 8–14, 15–21, or 22–28, 2025.
- **Prediction model:** combined Poisson, collision-only Poisson, or persistence.
- **Map measure:** prior-week signal complaints, prior-week collisions, predicted collisions, or observed target-week collisions.
- **Top-50 filter:** presentation sorting uses predicted count descending and lexicographic area-ID tie-breaking. It does not recompute coverage or change the backend's top-50 rule.
- **Area selection:** dropdown; if filtering hides the selected area, reset to the top displayed cell. Map-click selection is not implemented.
- **Comparison period:** selected week or pooled four-week evaluation. Activity-band MAE is always labeled as pooled.

All dropdowns are selection-only with typing disabled using Streamlit's native `filter_mode=None`. This requires **Streamlit 1.67 or newer**, as specified in `ui/requirements.txt`; reinstall UI requirements when updating this branch. Short labels, wrapping menu options, and a wider desktop sidebar prevent clipping; explanatory text appears below the controls. Header spacing clears Streamlit's fixed toolbar.

### Observed-map concentrations and hover details

The explorer is also available in the **Observed events** tab when real forecasts or development fixtures are loaded, so event-level investigation remains available after backend integration. Its observations are always real cleaned records, separately labeled from synthetic forecast fixtures.

Observation bins use a local equirectangular display grid at latitude 40.7, with approximate 500 m sides. They are presentation-only; they are not the backend's exact EPSG:32618 eligibility/features grid and never change predictions or evaluation. All observations in the selected historical window are shown, not just model-eligible cells.

Blue outlined circles show signal-report counts; orange filled circles show collision counts. Larger circles indicate higher counts, sharing the same size scale across the enabled layers; the scale can change when a layer is toggled or the week changes. Overlapping circles do not establish a signal–crash relationship. Toggle layers to inspect one dataset. The **Browse the underlying records** expander retains source IDs and detailed fields.

Area hovers show event counts, date/time ranges, complaint types and status counts, or collision injury/death totals. Individual-event hovers show collision date/time and outcomes, or report creation date/time, type, status, and component. Complaint status is explicitly a source-snapshot attribute, not the status as of the creation date or proof of repair.

The selected-area panel shows both historical counts, all three model predictions, and the later observed outcome. Zero-event cells remain visible. All models predict ALL collisions, rather than injury-only outcomes.

Map color and marker size encode counts. Radius is in screen pixels, never a geographic neighborhood. Pydeck receives a literal `pixels` unit (not an accessor expression), forecast circles are capped at 17 px, and the selected outline at 20 px. The camera starts over NYC at zoom 10.3 with zoom limits 9–16. The predicted-count scale is fixed across all three models and all four weeks; other count measures each use a fixed scale across weeks. Gray marks zero. Legends specify units and periods. CARTO tiles require internet access; map hover shows cell ID and displayed count.

## Canonical backend contract

Only these production files are read, from `data/processed/`:

- `predictions.csv`
- `metrics.csv`
- `run_metadata.json`

The exact required fields are in the handoff and in `ui/data.py`. Predictions contain every eligible cell in every evaluation week and preserve zero rows. The UI verifies fixed cell IDs/centers/activity bands, adjacent seven-day input/target windows, nonnegative finite predictions, integer counts, all three models' metrics, pooled and activity-band rows, and supported metadata schema/grid/universe settings.

`combined` means the trained combined Poisson model. No old percentile score, ordinary linear regression, radius-based selection, `selected_targets.csv`, `evaluation.csv`, or `src.api.get_area_evidence()` is used.

Percentages in metrics are **0–100**, not fractions. Coverage is blank if its denominator is zero. Activity-band coverage fields are blank, not zero. Metrics must provide week and overall rows for all models and activity-band rows for all populated bands. Counts in metric rows must match the exported eligible universe.

### Metadata hash representation

The handoff specifies hashes but not their key representation. This UI accepts cleaned filenames or relative/absolute paths as keys; only their basename is matched. Digest values can be strings or objects with a `sha256` field. Example preferred representation:

```json
{
  "input_sha256": {
    "signal_complaints_clean.csv": "<64 hex characters>",
    "collisions_clean.csv": "<64 hex characters>"
  }
}
```

Every prediction/metric row must match metadata `run_id`; schema version must be integer `1`. Metadata also includes the handoff's timestamps, windows, model settings, source audit, limitations, EPSG:32618 / 500 m settings, eligible area count, and August 1–7 eligibility bounds. Extra fields are allowed.

The UI checks file modification times/sizes plus run ID for cache invalidation. Cleaned-source SHA-256 hashes are compared to metadata. Changed or unverifiable sources produce a **STALE SAVED RUN** warning; saved outputs can be inspected but are not labeled current and downloads are disabled. Mixed runs, malformed schemas, invalid values, or missing cell/week/model rows are rejected. Old output files never act as fallback data. No backend files are changed by the UI.

The backend owns export generation. Write a complete run before launching/reloading; the reader rejects inconsistent run IDs and detects file changes during reading. `features.csv` is an audit artifact and is not required for UI startup.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/ui -v
```

The suite covers:

- Startup and control changes using production-shaped synthetic exports.
- Area selection, top-50 tie-breaking, selection reset, displayed/downloaded week agreement.
- Fixed model-comparison map scales and retained zero-event rows.
- Explicit fixture labeling and disabled fixture downloads.
- Read-only dropdown settings, literal map-pixel serialization, NYC camera bounds, and an actual-source map when model files are absent.
- Descriptive event hover payloads, concentration/detail switching, larger markers for higher-count areas, and conservation of observed counts across display bins.
- Missing/malformed/mixed-run/stale states, unsupported schema versions, and input-hash checks.
- Cache invalidation when run IDs/files change.
- An additional real-output startup/hash check that runs once all canonical production exports exist; otherwise explicitly skipped.

Tests do not fit models or verify numerical claims about the real dataset. Streamlit AppTest validates Python rendering and interactions, not browser tile rendering. The managed session's supported browser-preview tool is unavailable, so a visual browser check remains part of the local demo.

## Integration checklist for the backend owner / team

1. Publish schema-compliant real exports with matching run IDs and hashes of the committed cleaned inputs.
2. Keep the four exact target windows, all eligible cells, and all required metric scopes.
3. Resolve generated-output tracking through the shared-file owner; the current `.gitignore` may ignore these new files. The UI does not edit that policy.
4. Merge reviewed PRs one at a time after team approval. Pull updated `main`, build exports, then run the UI suite.
5. Start Streamlit and verify no fixture/stale/error banner appears; switch each week/model/measure, inspect zero and high-count cells, toggle top-50, and download the visible rows.
6. Compare displayed weekly/pooled values to `metrics.csv` and verify the same run ID appears in metadata and downloads.

## Short demo walkthrough

1. Open the forecast map on combined / September 1–7 / predicted collisions.
2. Switch to collision-only, explaining the fixed visual scale and all-collision expected-count units.
3. Show preceding-week complaints and retrospective observed collisions, noting their different periods.
4. Select an area and compare its historical counts, three predictions, and observed outcome.
5. Open Model comparison: show MAE and top-50 coverage for the selected week, then pooled results and activity-band errors. Read the actual tradeoff honestly.
6. Show the eligible-universe coverage ceiling and limitations, then download real prediction rows/metrics.
