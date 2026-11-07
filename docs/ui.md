# Streamlit UI

## Run from the repository root

Use Python 3.10 or newer:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r ui/requirements.txt
python -m streamlit run app.py
```

The separate UI dependency file avoids editing the backend owner's dependencies. The backend owner can later incorporate it into a root requirements file. The existing cleaner is unchanged.

## Current behavior

- Real August/September source layers from the committed cleaned CSVs, with explicit injury OR fatality filtering and source-event month controls.
- Radius 100–250 m in 25 m increments; crash-first strategy by default; shortlist sizes 25/50/100, default 50.
- Shortlist, target selection, evidence circles, downloads, comparison chart, quality notes, and missing/error states.
- No scoring, candidate generation, spacing, neighborhood matching, or evaluation calculations in UI code.
- When backend exports are missing, the app shows real source events and marks analysis as pending. The development checkbox enables three explicitly synthetic layout examples; evaluation and downloads are disabled in this mode.
- Target selection uses a dropdown. Map hover exposes event IDs and details; map-click selection is not implemented.
- Source layers show all mapped events for the chosen month, independently of target neighborhoods. August target evidence and September evaluation remain clearly separated.

## Backend integration

The interface follows the latest team build plan. Existing `docs/DATA_CONTRACTS.md` describes the earlier starter and needs updating by its owner; the actual UI export expectations are below.

`data/processed/selected_targets.csv` requires:

```text
target_id,latitude,longitude,radius_m,strategy,signal_report_count,harm_crash_count,
score,top_descriptor,suggested_inspection,selected_rank,budget
```

`data/processed/evaluation.csv` requires:

```text
strategy,radius_m,budget,selected_count,september_harm_crashes_covered,
september_harm_coverage_pct,august_signal_reports_covered
```

Percentages are 0–100, not fractions. Strategies are `crash_first`, `complaint_first`, and `combined`. Required numeric values must be populated. Each selected target is unique within radius/strategy/budget; each evaluation configuration is unique. Export all seven radii and three budgets. Missing configurations are displayed as unavailable; the UI does not fall back to another radius or invent numbers.

The app does not need `targets_scored.csv` to show the selected shortlist. It remains a backend-owned audit output.

Evidence helper:

```python
# Backend owner implements src/api.py:
def get_area_evidence(target_id, radius_m, period="august"):
    return {"reports": reports_dataframe, "collisions": collisions_dataframe}
```

Return original event IDs and source columns. Nonempty reports must include `unique_key` and `created_date`; nonempty collisions must include `collision_id` and `crash_date`. Keep the requested period only. Include `distance_m` and injury/fatality flags for useful evidence display; label whether collisions include all outcomes or harmful outcomes only. Empty DataFrames are accepted. The helper receives no ranking strategy because spatial membership is independent of ranking.

Changing radius, strategy, or budget selects matching backend exports and refreshes the map, shortlist, metrics, evidence, and comparison. If the selected target no longer appears, selection resets to the top target. Comparisons include all available strategies at the same radius/budget.

File modification time/size signatures for cleaned data, generated exports, and backend Python files invalidate UI caches. Evidence keys also include target/radius/period. Backend owners must regenerate exports after data/method changes: file caching cannot establish the analytical provenance of externally generated CSVs. Restart Streamlit after changing backend Python code so imported modules refresh.

Generated analysis CSVs are ignored by the current repository `.gitignore`. The backend owner must document a generation command or explicitly choose to track small final exports. UI code does not modify that policy.

## Validation

```bash
python -m unittest discover -s tests/ui -v
```

Tests use Streamlit AppTest and temporary in-memory export fixtures. They verify real-source startup, development warnings/download restrictions, control changes, strict configuration selection, fatal-only crash inclusion, and the real-export/evidence connection. They do not require the ranking backend to exist.

The basemap needs network access to CARTO tiles. AppTest checks Python rendering and widget behavior, not browser map-tile rendering.
