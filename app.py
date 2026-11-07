"""Observed NYC map with separate model comparison and prediction tables."""

import json

import pandas as pd
import streamlit as st

from ui.data import (MODELS, PREDICTION_COLUMNS, WEEKS, ForecastDataError,
                     load_forecast_bundle, week_label, week_rows)
from ui.observations import render_observations

st.set_page_config(page_title="NYC | Collision Forecasts", page_icon="🚦", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 5rem; max-width: 1550px;}
[data-testid="stMainBlockContainer"] {padding-top: 5rem;}
h1 {letter-spacing: -.045em; font-weight: 700; line-height: 1.2;}
h2, h3 {letter-spacing: -.025em;}
@media (min-width: 768px) {section[data-testid="stSidebar"] {min-width: 19rem;}}
[data-baseweb="popover"] [role="option"] div {white-space: normal !important;
    text-overflow: clip !important; overflow: visible !important;}
[data-testid="stMetric"] {background: white; border: 1px solid #e2e8ed;
    border-radius: 12px; padding: 16px;}
.eyebrow {font-size: .75rem; letter-spacing: .15em; color: #176b5b; font-weight: 700;}
.legend {max-width: 340px; height: 9px; border-radius: 5px;
    background: linear-gradient(to right, #398ba0, #e2522d); margin: 6px 0;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">NYC IN MOTION / COMPLAINTS & COLLISIONS</div>', unsafe_allow_html=True)
st.title("NYC complaints & collisions")
st.write("Explore where signal complaints and collisions concentrate, then compare whether "
         "complaints improve forecasts beyond collision history alone.")
st.caption("Historical prototype · August training / September evaluation, 2025 · NYC local time")


with st.sidebar:
    st.markdown("### Explore NYC")
    forecast_week = st.selectbox("Evaluation week", WEEKS, format_func=week_label,
                                 key="forecast_week", filter_mode=None)
    st.caption("The map shows the preceding week’s real observations. Tables evaluate the selected week.")
    model = st.selectbox("Model for tables", list(MODELS), format_func=MODELS.get,
                         key="prediction_model", filter_mode=None)
    st.caption({"combined": "Poisson model using past collisions and signal complaints.",
                "collision_only": "Poisson model using past collisions only.",
                "persistence": "Predicts the previous week’s collision count."}[model])
    st.divider()
    fixture = st.checkbox("Development: use example forecast", key="fixture_mode",
                          help="Synthetic model tables only; the observed map always uses real records.")

bundle = None
load_problem = None
try:
    bundle = load_forecast_bundle(fixture=fixture)
except (FileNotFoundError, ForecastDataError, OSError, ValueError, KeyError, TypeError) as exc:
    load_problem = str(exc)
if bundle is None:
    render_observations(forecast_week)
    with st.expander("Model connection details"):
        st.caption(load_problem)
        st.caption("Observed records remain available. Model comparison needs a complete valid export set.")
        st.code("python -m src.build_forecast")
    st.stop()

if fixture:
    st.warning("DEVELOPMENT EXAMPLE — model tables and evaluation use synthetic fixtures, not research findings. "
               "Downloads are disabled. The observed map uses real cleaned records.")

meta = bundle.metadata
run_id = meta["run_id"]
all_week = week_rows(bundle, forecast_week, model)
first = all_week.iloc[0]
feature_label = week_label(first["feature_start"])
forecast_label = week_label(forecast_week)
weekly_metrics = bundle.metrics.loc[
    bundle.metrics["scope"].eq("week") & bundle.metrics["forecast_start"].eq(forecast_week)
]
can_download = not fixture and not bundle.stale_reasons
observed, compare, table, method = st.tabs(["Observed map", "Model comparison", "Prediction table", "Data & method"])
with observed:
    render_observations(forecast_week)

with compare:
    if bundle.stale_reasons:
        st.warning("MODEL INPUT MISMATCH — the saved model results were built from different input files. "
                   "The observed map still shows your current cleaned records. Model downloads are disabled.")
        for reason in bundle.stale_reasons:
            st.caption(reason)
        with st.expander("How to refresh model results"):
            st.write("A file-content hash differs from the run metadata. If only Windows CRLF line endings "
                     "differ from the expected LF file, the UI verifies that automatically. Other changes "
                     "require checking the inputs and rebuilding the model exports.")
            st.code("python -m src.build_forecast")
    for note in bundle.provenance_notes:
        st.caption(note)
    st.caption(f"Run {run_id} · Generated {meta['created_at_utc']}")
    current_metric = weekly_metrics.loc[weekly_metrics["model"].eq(model)].iloc[0]
    summary = st.columns(3)
    summary[0].metric("Weekly MAE · lower is better", f"{current_metric['mae']:.3f}")
    coverage = current_metric["top_k_coverage_pct"]
    summary[1].metric("Top-50 collision coverage", "N/A" if pd.isna(coverage) else f"{coverage:.1f}%")
    summary[2].metric("Mapped citywide collisions · target week", f"{int(current_metric['citywide_collisions']):,}")
    st.caption(f"{MODELS[model]} · {forecast_label}. Metrics are read from the backend exports.")
    st.subheader("Does adding complaints improve the forecast?")
    comparison_scope = st.radio("Comparison period", ["Selected week", "All four weeks"],
                                horizontal=True, key="comparison_scope")
    comparison = (weekly_metrics.copy() if comparison_scope == "Selected week"
                  else bundle.metrics.loc[bundle.metrics["scope"].eq("overall")].copy())
    st.caption(f"{forecast_label if comparison_scope == 'Selected week' else 'September 1–28, 2025 · pooled evaluation'} "
               "· Same eligible rows for all models · Authoritative backend metrics")
    comparison["Model"] = comparison["model"].map(MODELS)
    comparison = comparison.set_index("model").reindex(list(MODELS)).reset_index()
    charts = st.columns(2)
    with charts[0]:
        st.markdown("**Mean absolute error · lower is better**")
        st.bar_chart(comparison.set_index("Model")[["mae"]].rename(columns={"mae": "MAE (collisions per cell-week)"}),
                     horizontal=True, color="#176B5B")
    with charts[1]:
        st.markdown("**Top-50 collision coverage · higher is better**")
        st.bar_chart(comparison.set_index("Model")[["top_k_coverage_pct"]].rename(
            columns={"top_k_coverage_pct": "Citywide mapped collisions covered (%)"}),
            horizontal=True, color="#398BA0")
    st.dataframe(comparison[["Model", "mae", "top_k", "top_k_collisions", "citywide_collisions",
                             "eligible_collisions", "top_k_coverage_pct", "eligible_coverage_pct"]],
                 hide_index=True, width="stretch", column_config={
                     "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
                     "top_k_coverage_pct": st.column_config.NumberColumn("Top-50 coverage (%)", format="%.2f"),
                     "eligible_coverage_pct": st.column_config.NumberColumn("Eligible-universe coverage (%)", format="%.2f"),
                 })
    st.caption("Top-50 coverage counts collisions in the model’s 50 highest-predicted cells, divided "
               "by all mapped citywide collisions. Eligible-universe coverage shows the ceiling "
               "available within the study cells. Pooled coverage uses pooled counts, not an average of weekly percentages.")
    st.subheader("Does performance differ in historically busy areas?")
    bands = bundle.metrics.loc[bundle.metrics["scope"].eq("activity_band")].copy()
    bands["Model"] = bands["model"].map(MODELS)
    st.dataframe(bands[["Model", "activity_band", "area_week_rows", "mae"]],
                 hide_index=True, width="stretch", column_config={
                     "mae": st.column_config.NumberColumn("Pooled MAE", format="%.3f"),
                 })
    st.caption("Bands use August 1–7 collision counts: zero, one, or two_plus. "
               "They are fixed before evaluation and are not population or traffic-exposure estimates.")
    st.download_button("Download authoritative metrics", bundle.metrics.to_csv(index=False).encode(),
                       f"metrics_{run_id}.csv", "text/csv", disabled=not can_download, key="download_metrics")
    st.caption("Download includes all weeks, pooled results, and activity bands. "
               "Report worse results honestly; no model is assumed to win.")


with table:
    st.subheader("Model predictions by area")
    st.caption(f"Inputs {feature_label} → target {forecast_label}. Expected counts, not crash probabilities.")
    if bundle.stale_reasons:
        st.warning("Saved model table: input files differ from the recorded run. See Model comparison for details.")
    top_only = st.checkbox("Show model’s top 50 areas only", key="top_only")
    rows = week_rows(bundle, forecast_week, model, top_only)
    ids = rows["area_id"].astype(str).tolist()
    if st.session_state.get("selected_area") not in ids:
        st.session_state["selected_area"] = ids[0]
    selected_id = st.selectbox("Select an area", ids, key="selected_area", filter_mode=None)
    visible = ["area_id", "past_complaints", "past_collisions", *PREDICTION_COLUMNS.values(), "actual_collisions"]
    st.dataframe(rows[visible], hide_index=True, width="stretch")
    area = rows.loc[rows["area_id"].eq(selected_id)].iloc[0]
    st.caption(f"Selected area: {selected_id} · {area['latitude']:.5f}, {area['longitude']:.5f}")
    st.dataframe(pd.DataFrame([
        {"Model": MODELS[mid], "Expected collisions": area[column]}
        for mid, column in PREDICTION_COLUMNS.items()
    ]), hide_index=True, width="stretch")
    st.caption(f"Observed outcome: {int(area['actual_collisions'])} collisions. "
               f"Prior week: {int(area['past_complaints'])} complaints and {int(area['past_collisions'])} collisions.")
    st.download_button("Download displayed prediction rows", rows.to_csv(index=False).encode(),
                       f"predictions_{model}_{forecast_week}_{'top50' if top_only else 'all'}.csv",
                       "text/csv", disabled=not can_download, key="download_predictions")

with method:
    st.subheader("What the experiment measures")
    st.markdown("""
**Question.** Do traffic-signal complaint counts add predictive information beyond collision history?

**Geography.** Fixed 500 m cells in EPSG:32618, eligible only if they had a mapped complaint or
collision on August 1–7. Every eligible cell is retained every week, including zero-event cells.
Other NYC locations are outside the prediction universe, not necessarily safe.

**Forecasting.** The previous seven days of counts predict the next seven days. Training targets are
August 8–14, 15–21, and 22–28. Evaluation targets are September 1–7, 8–14, 15–21, and 22–28.
Preprocessing and fitted models stay frozen. Earlier September observations can inform later
September forecasts; this is not a frozen August-only forecast of the entire month.

**Models.** Persistence predicts the last week’s collision count. Collision-only and combined use
log1p count features → StandardScaler → PoissonRegressor, alpha 1.0, max_iter 1000. Only the combined
model adds complaint counts. The backend owns fitting and all evaluation calculations.

**Limits.** September evaluation is exploratory and covers only four weeks. Traffic exposure,
population activity, and road characteristics are not controlled. Predictive improvement would not
establish that signal defects caused crashes, that a staffing shortage exists, or that repairs would
prevent a quantified number of injuries. Complaints are reports, not verified defects.
""")
    st.markdown("[NYC 311 source](https://data.cityofnewyork.us/d/erm2-nwe9) · "
                "[NYPD collision source](https://data.cityofnewyork.us/d/h9gi-nx95)")
    st.caption("Original bundle audit: 6,534 signal complaints → 6,119 mapped (415 excluded); "
               "14,526 collisions → 14,066 mapped (119 missing coordinates and 341 zero-coordinate pairs). "
               "The active run’s source audit below takes precedence for changed inputs.")
    with st.expander("Active run provenance and source audit"):
        st.json(meta)
        st.caption("Canonical outputs only: predictions.csv, metrics.csv, run_metadata.json. "
                   "Old ranking exports are never a fallback. UI caches include file signatures and run ID.")
    st.download_button("Download run metadata", json.dumps(meta, indent=2).encode(),
                       f"run_metadata_{run_id}.json", "application/json", disabled=not can_download,
                       key="download_metadata")
