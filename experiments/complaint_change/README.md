# Exploratory add-on: week-over-week complaint change

This independent experiment leaves the canonical backend, production exports, and observed map unchanged. Its saved outputs are displayed in the dashboard's **Complaint change** tab.

## Question and feature

Does change in recent signal-report volume add predictive value beyond recent collision counts and complaint level?

```text
recent complaints = reports in [forecast_start - 7 days, forecast_start)
previous complaints = reports in [forecast_start - 14 days, forecast_start - 7 days)
complaint_change = recent complaints - previous complaints
```

Positive and negative changes are retained. This is a difference between two weeks, **not an estimate of an area's long-term normal**. It requires 14 days of observed history. No history before August 1 is fabricated or zero-filled.

## Fair comparison

The first original August training window lacks enough history, so ALL experiment models use the same two training target windows: **August 15–21 and August 22–28** (2,360 area-week rows). Evaluation uses the same four September weeks and 1,180 eligible cells as the main model (4,720 rows).

Compare within this experiment:

- `persistence`: last week's collision count.
- `collision_only_2w`: log1p prior-week collision count.
- `complaint_level_2w`: log1p prior-week collision and complaint counts.
- `complaint_change_2w`: those same features plus signed week-over-week complaint change.

Every trained model uses a training-only StandardScaler and PoissonRegressor with alpha 1.0 / max_iter 1000. Counts use log1p. Change uses `sign(x) * log1p(abs(x))` so decreases are valid. No September outcomes enter fitting; no tuning was performed for this add-on.

The original backend's validated EPSG:32618 projection, fixed early eligible universe, source loading, feature counts, and MAE/coverage evaluator are reused through imports. Its files, configuration, and canonical exports are never modified. Metrics reuse the authoritative evaluator with explicit model-label adapters; the Streamlit UI reads saved metrics rather than calculating competing ones.

## Saved results

Pooled September evaluation:

| Model | MAE | Top-50 citywide collision coverage |
| --- | ---: | ---: |
| Persistence | 0.937076 | 5.5107% |
| Collision history, matched training | 0.844359 | 5.5107% |
| Complaint level, matched training | 0.840884 | 5.2755% |
| Complaint level + weekly change | 0.840650 | 5.2755% |

**Weekly change adds a negligible improvement over complaint level:** about 0.000234 fewer absolute-error collisions per area-week, or **0.028% relative MAE improvement**, with unchanged top-50 coverage. It does not establish practical benefit or a significant effect. Relative to collision-only, MAE improves about 0.44% while coverage falls about 0.24 percentage points.

These numbers come from the saved real-data run, not the UI fixtures. Do not compare directly against the original three-training-window model and attribute the difference solely to complaint change. There are only two experiment training windows and four evaluation windows; September outcomes were already explored. This is hypothesis exploration, not an untouched confirmatory test or causal finding.

## Run and verify

From the repository root, using Python 3.12:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m experiments.complaint_change.build
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Saved files are committed under `results/`: features, predictions, metrics, and metadata. These are separate from `data/processed/` and do not change the main model schema. Metadata records source/implementation/output hashes, fit settings, matched windows, comparisons, and limitations. Result CSVs use LF line endings through this folder's `.gitattributes`.

```powershell
.\.venv\Scripts\python.exe -m unittest experiments.complaint_change.test_experiment -v
.\.venv\Scripts\python.exe -m unittest discover -s tests/ui -v
```

Checks verify real cell/window counts against source IDs, complete history, signed changes, training-only scaling/fitting, finite predictions, real output hashes, tab interactions, and fixture separation. A second full build reproduced all three CSVs byte for byte. Only metadata creation time can change. The UI refuses stale or modified experiment artifacts, and does not use the real add-on while synthetic main fixtures are enabled.
