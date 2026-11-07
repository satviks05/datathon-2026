"""Streamlit presentation for canonical weekly collision forecast exports."""

import json

import pandas as pd
import streamlit as st

from ui.data import (
    MODELS, MEASURES, PREDICTION_COLUMNS, WEEKS, ForecastDataError,
    color_scale_max, load_forecast_bundle, week_label, week_rows,
)
from ui.map_view import make_map

st.set_page_config(page_title="NYC | Collision Forecasts", page_icon="🚦", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1550px;}
h1 {letter-spacing: -.045em; font-weight: 700;}
h2, h3 {letter-spacing: -.025em;}
[data-testid="stMetric"] {background: white; border: 1px solid #e2e8ed;
    border-radius: 12px; padding: 16px;}
.eyebrow {font-size: .75rem; letter-spacing: .15em; color: #176b5b; font-weight: 700;}
.legend {max-width: 340px; height: 9px; border-radius: 5px;
    background: linear-gradient(to right, #398ba0, #e2522d); margin: 6px 0;}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">NYC IN MOTION / WEEKLY COLLISION FORECASTS</div>', unsafe_allow_html=True)
st.title("Do complaints add predictive signal?")
st.write("Compare collision history alone with collision history plus traffic-signal complaints. "
         "Explore expected collision counts and what happened in the following week.")
st.caption("Historical prototype · August training / September evaluation, 2025 · NYC local time")

with st.sidebar:
    st.markdown("### Explore a forecast")
    forecast_week = st.selectbox("Forecast week", WEEKS, format_func=week_label, key="forecast_week")
    model = st.selectbox("Prediction model", list(MODELS), format_func=MODELS.get, key="prediction_model")
    measure = st.selectbox("Map measure", list(MEASURES), format_func=MEASURES.get, key="map_measure")
    top_only = st.checkbox("Show model’s top 50 areas only", key="top_only",
                           help="Sort by predicted count descending, then area ID. Does not change the backend’s evaluation.")
    st.caption("Fixed 500 m cells · No radius search or inspection-spacing rule")
    st.divider()
    fixture = st.checkbox("Development: use example forecast", key="fixture_mode",
                          help="Synthetic values for UI development only. Never use as findings.")

bundle = None
try:
    bundle = load_forecast_bundle(fixture=fixture)
except FileNotFoundError as exc:
    st.info(str(exc))
    st.write("The new forecasting backend has not supplied a complete export set yet. "
             "Older radius-ranking and monthly-regression files are intentionally not used.")
    st.caption("The backend owner supplies this build command with the new implementation:")
    st.code("python -m src.build_forecast")
    st.caption("Enable the development example in the sidebar to check the UI layout. "
               "It is not an analytical result.")
except (ForecastDataError, OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Forecast outputs rejected: {exc}")
    st.caption("Ask the backend owner to rebuild the canonical outputs together. "
               "No old exports or substitute values will be displayed.")
if bundle is None:
    st.stop()

if fixture:
    st.warning("DEVELOPMENT EXAMPLE — all counts, predictions, metrics, and locations below are "
               "synthetic layout fixtures. They are not model findings. Downloads are disabled.")
elif bundle.stale_reasons:
    st.warning("STALE SAVED RUN — these outputs do not match the available cleaned inputs. "
               "They are shown only as saved historical results, not current predictions. "
               "Downloads are disabled until the backend rebuilds or input provenance is restored.")
    for reason in bundle.stale_reasons:
        st.caption(reason)

meta = bundle.metadata
run_id = meta["run_id"]
rows = week_rows(bundle, forecast_week, model, top_only)
all_week = week_rows(bundle, forecast_week, model)
first = all_week.iloc[0]
feature_label = week_label(first["feature_start"])
forecast_label = week_label(forecast_week)
context = f"{MODELS[model]} · Forecast {forecast_label} · Inputs {feature_label}"
st.caption(context)
st.caption(f"{'SYNTHETIC EXAMPLE' if fixture else 'Saved run' if bundle.stale_reasons else 'Verified input hashes'} "
           f"· Run {run_id} · Generated {meta['created_at_utc']}")

weekly_metrics = bundle.metrics.loc[
    bundle.metrics["scope"].eq("week") & bundle.metrics["forecast_start"].eq(forecast_week)
]
current_metric = weekly_metrics.loc[weekly_metrics["model"].eq(model)].iloc[0]
summary = st.columns(4)
summary[0].metric("Eligible 500 m cells", f"{len(all_week):,}")
summary[1].metric("Weekly MAE · lower is better", f"{current_metric['mae']:.3f}")
coverage = current_metric["top_k_coverage_pct"]
summary[2].metric("Top-50 citywide collision coverage", "N/A" if pd.isna(coverage) else f"{coverage:.1f}%")
summary[3].metric("Mapped citywide collisions · target week", f"{int(current_metric['citywide_collisions']):,}")
st.caption("MAE and coverage are read from backend metrics, not recalculated here. "
           "Top-50 coverage uses all mapped citywide collisions in the selected target week.")

explore, compare, method = st.tabs(["Forecast map", "Model comparison", "Data & method"])
can_download = not fixture and not bundle.stale_reasons
with explore:
    map_column, table_column = st.columns([1.7, 1], gap="large")
    with table_column:
        st.subheader("Areas to explore")
        st.caption(f"Showing {len(rows):,} of {len(all_week):,} eligible cells. "
                   "Ordered by the selected model, with area-ID tie-breaking.")
        ids = rows["area_id"].astype(str).tolist()
        if st.session_state.get("selected_area") not in ids:
            st.session_state["selected_area"] = ids[0]
        selected_id = st.selectbox("Select an area", ids, key="selected_area")
        visible_columns = ["area_id", "past_complaints", "past_collisions",
                           PREDICTION_COLUMNS[model], "actual_collisions"]
        st.dataframe(rows[visible_columns], hide_index=True, width="stretch", column_config={
            "area_id": "Area", "past_complaints": "Prior-week complaints",
            "past_collisions": "Prior-week collisions", "actual_collisions": "Observed next week",
            PREDICTION_COLUMNS[model]: st.column_config.NumberColumn("Predicted next week", format="%.3f"),
        })
        st.download_button(
            "Download displayed prediction rows", rows.to_csv(index=False).encode(),
            f"predictions_{model}_{forecast_week}_{'top50' if top_only else 'all'}_{measure}.csv",
            "text/csv", disabled=not can_download, key="download_predictions",
        )
        st.caption("Downloads preserve the canonical columns and all three predictions. "
                   "The filename records the active model, week, map measure, and filter.")
    with map_column:
        st.subheader(MEASURES[measure])
        if measure in {"past_complaints", "past_collisions"}:
            st.caption(f"Observed inputs: {feature_label}. These precede the prediction period.")
        elif measure == "actual_collisions":
            st.caption(f"Retrospective outcome: {forecast_label}. Not a forecast or model input.")
        else:
            st.caption(f"Expected collisions: {forecast_label}. Fractional values represent expected counts.")
        scale = color_scale_max(bundle, measure)
        st.pydeck_chart(make_map(rows, model, measure, scale, selected_id), width="stretch", height=480)
        st.markdown('<div class="legend"></div>', unsafe_allow_html=True)
        st.caption(f"Count scale: 0 → {scale:.2f}. Gray = zero. Larger/brighter markers = higher count. "
                   "Scale fixed across models and weeks for this measure.")
        st.caption("One marker per eligible 500 m cell at its exported center. Marker size is a visual "
                   "encoding in pixels, not a geographic radius. CARTO / OpenStreetMap basemap.")

    st.divider()
    st.subheader("Selected area · inputs, predictions, and outcome")
    area = rows.loc[rows["area_id"].eq(selected_id)].iloc[0]
    st.markdown(f"**{selected_id}** · {area['latitude']:.5f}, {area['longitude']:.5f} "
                f"· Early-history activity band: `{area['activity_band']}`")
    st.caption(f"Inputs {feature_label} → forecast {forecast_label}. Exported end dates are exclusive.")
    inputs = st.columns(3)
    inputs[0].metric("Prior-week signal complaints", int(area["past_complaints"]))
    inputs[1].metric("Prior-week collisions", int(area["past_collisions"]))
    inputs[2].metric("Observed target-week collisions", int(area["actual_collisions"]))
    predictions = st.columns(3)
    for column, model_id in zip(predictions, MODELS):
        column.metric(MODELS[model_id], f"{area[PREDICTION_COLUMNS[model_id]]:.3f}")
    st.caption("All models predict ALL mapped collisions, not only injury/fatal crashes. "
               "These expected counts are not per-traveler risk or probabilities.")

with compare:
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
