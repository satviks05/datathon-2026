# NYC complaints and collisions UI

The main view is the **Observed map**. The predicted-count grid map has been removed. Model results remain in **Model comparison** and **Prediction table**, separate from recorded events.

## Run on Windows

Use Python 3.12, 64-bit. From the repository root:

```powershell
git fetch origin
git switch fix/observed-map-priority
git pull origin fix/observed-map-priority
.\.venv\Scripts\python.exe -m pip install -r ui/requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

If needed, create the environment first with `py -3.12 -m venv .venv`. On macOS/Linux use `python3.12` and `.venv/bin/python`. Stop Streamlit with Ctrl+C. Selection-only dropdowns require Streamlit 1.67 or newer.

## Observed map

- Blue circles represent Traffic Signal Condition complaints. Orange circles represent ALL recorded collisions.
- **Area counts** is the default. It groups real records into approximately 500 m display bins using a local equirectangular grid at reference latitude 40.7.
- Circle radius is `6 * sqrt(count)` pixels, capped at 60 px above 100 events. Consequently, five events have five times the circle area of one event and about 2.2 times its diameter. One event has 12 px diameter; five have about 27 px; ten have about 38 px.
- This size rule stays fixed across dates and layer toggles. Colors encode event type; circle area encodes recorded count. Blue outlines/pale fill and orange circles can be toggled separately where they overlap.
- **Individual events** uses equal-sized points. Hover shows collision date/time and injury/death counts, or complaint creation time, type, source-snapshot status, and component. Source status does not establish the status at the historical creation date or proof of repair.
- Group hovers show counts and date/time ranges plus complaint status summaries or collision outcomes. Underlying-record tables preserve IDs without making opaque IDs the primary map labels.
- The camera starts over NYC at zoom 10.3, with limits 9–16. Marker size is a screen encoding, not a geographic radius. CARTO tiles require internet access.

The map shows the seven days preceding the selected **Evaluation week**. The sidebar explains this explicitly. Observation bins include all mapped events in that window, not only model-eligible cells. They are presentation-only and never change backend features, predictions, or evaluation.

## Model panels

The selected model applies to comparison metrics and the prediction table, not map colors or observation counts. The old map-measure control is gone. Prediction rows retain the authoritative expected counts and actual outcomes for the selected week. The optional top-50 table filter sorts by prediction descending, then area ID lexicographically. Models are not fitted and evaluation metrics are not recomputed in UI code.

Backend files read from `data/processed/`:

- `predictions.csv`: all 1,180 eligible areas across four evaluation weeks in the current real export (4,720 rows).
- `metrics.csv`: weekly, pooled, and activity-band metrics (24 rows in the current real export).
- `run_metadata.json`: schema version, run ID, source hashes, configuration, and source audit.

`docs/PROJECT_HANDOFF.md` remains the analytical contract. The UI follows its exact model IDs, schemas, windows, and eligibility. A map removal changes presentation only. Missing/malformed/mixed-run exports never fall back to old ranking files. The observed map remains available if model files cannot load.

Development fixtures are opt-in, synthetic model tables, explicitly labeled, with downloads disabled. The observed map always uses real cleaned inputs, even when model fixtures are enabled.

## What the stale warning means

The earlier **STALE SAVED RUN** banner meant at least one local cleaned input failed the file-content hash recorded when the model exports were created. It did not establish whether the difference was actual records, formatting, line endings, or missing files. It did not indicate that the observed event map was stale.

The updated UI first compares exact SHA-256 hashes. If those differ, it also checks whether converting Windows CRLF newlines to LF yields the exact recorded digest. That accepts a provably matching LF representation without accepting changed record bytes. A provenance note explains when this normalization was used. No source files or backend outputs are modified.

Other differences or missing inputs remain flagged. **MODEL INPUT MISMATCH** appears inside model panels, while the observed map continues to show the current cleaned files. Model downloads stay disabled until the inputs are restored or exports rebuilt. Inspect the filename in the panel and use the backend's documented rebuild command if appropriate:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m src.build_forecast
```

Do not dismiss real mismatches merely to enable downloads. If a recorded expected hash was itself calculated on CRLF content and neither exact nor LF-normalized local bytes match, rebuilding remains necessary.

Metadata hash keys may be cleaned filenames or paths, with digest strings or objects containing `sha256`. Schema version must be integer 1. All output rows must match metadata run ID. Input/export signatures and run ID invalidate caches. The UI validates fixed windows, all eligible cells including zeros, finite counts/predictions, required model/metric scopes, and row counts.

## Verification and demo

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/ui -v
```

Tests cover actual-export loading/startup, source hashes, safe CRLF normalization versus changed records, all controls and downloads, missing/stale/malformed/mixed-run states, top-50 ties and selection reset, retained zero rows, descriptive hovers, count conservation, exact 5:1 marker-area ratio, NYC camera bounds, and removal of the forecast map. Streamlit AppTest does not visually verify browser tile rendering.

Demo: open Observed map; hover a blue and orange concentration; use the 1/5/10-event size legend; toggle each layer; switch to Individual events for timestamps and complaint status. Then show Model comparison and Prediction table as the separate model evidence. Report the actual comparison honestly rather than assuming complaints improve results.

Only UI-owned files change. Model definitions, production source snapshots, and backend exports are maintained by the backend owner. PRs are reviewed and merged sequentially after team approval.

## Optional complaint-change add-on

The **Complaint change** tab reads a separate, real-data experiment from `experiments/complaint_change/results/`. It compares weekly change against collision-only and complaint-level models refitted on the same TWO available August training windows. The canonical three-window models and exports stay unchanged. The tab labels this reduced history and exploratory status, checks artifact/source hashes, and is disabled in synthetic main-fixture mode. Build and results are documented in `experiments/complaint_change/README.md`.
