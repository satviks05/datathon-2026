"""Read the isolated experiment; never fit models in Streamlit."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ui import data

RESULTS = data.ROOT / "experiments" / "complaint_change" / "results"
LABELS = {
    "persistence": "Previous week",
    "collision_only_2w": "Collision history · matched",
    "complaint_level_2w": "Complaint level · matched",
    "complaint_change_2w": "Complaint level + weekly change",
}


def _matches(path, expected):
    raw = Path(path).read_bytes()
    return expected in {sha256(raw).hexdigest(), sha256(raw.replace(b"\r\n", b"\n")).hexdigest()}


@st.cache_data(show_spinner=False)
def _load(directory, source_directory, signatures):
    directory, source_directory = Path(directory), Path(source_directory)
    meta = json.loads((directory / "metadata.json").read_text())
    if meta.get("schema_version") != 1 or meta.get("experiment") != "week_over_week_complaint_change":
        raise ValueError("Unsupported complaint-change experiment")
    for filename in ("predictions.csv", "metrics.csv"):
        if not _matches(directory / filename, meta["output_sha256"][filename]):
            raise ValueError(f"Experiment output hash differs: {filename}")
    for filename in data.SOURCES:
        if not _matches(source_directory / filename, meta["input_sha256"][filename]):
            raise ValueError(f"Experiment inputs differ: {filename}; rebuild the experiment")
    predictions = pd.read_csv(directory / "predictions.csv", dtype={"area_id": "string", "run_id": "string"})
    metrics = pd.read_csv(directory / "metrics.csv", dtype={"run_id": "string"})
    required = {"area_id", "forecast_start", "past_complaints", "previous_complaints", "complaint_change",
                "actual_collisions", "run_id", "complaint_change_2w_prediction"}
    if not required.issubset(predictions.columns) or not {"mae", "top_k_coverage_pct", "scope", "model", "run_id"}.issubset(metrics.columns):
        raise ValueError("Incomplete experiment schema")
    if predictions.run_id.isna().any() or metrics.run_id.isna().any() or predictions.run_id.ne(meta["run_id"]).any() or metrics.run_id.ne(meta["run_id"]).any():
        raise ValueError("Mixed experiment runs")
    if set(predictions.forecast_start) != set(data.WEEKS) or predictions.duplicated(["area_id", "forecast_start"]).any():
        raise ValueError("Incomplete or duplicated experiment weeks")
    if set(metrics.model) != set(LABELS):
        raise ValueError("Experiment models are incomplete")
    if not np.isfinite(predictions["complaint_change_2w_prediction"]).all() or predictions["complaint_change_2w_prediction"].lt(0).any():
        raise ValueError("Invalid experiment predictions")
    return predictions, metrics, meta


def load_experiment():
    paths = [RESULTS / name for name in ("metadata.json", "predictions.csv", "metrics.csv")]
    paths += [data.DATA / name for name in data.SOURCES]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("The optional complaint-change experiment has not been built for these inputs")
    return _load(str(RESULTS), str(data.DATA), tuple(data.signature(path) for path in paths))


def render_change_experiment(forecast_week, fixture=False):
    st.subheader("Exploratory add-on · week-over-week complaint change")
    st.write("Does a recent rise or fall in complaints add information beyond complaint volume and collision history?")
    st.caption("Weekly change = complaints in the last 7 days minus complaints in the preceding 7 days. "
               "This is not a long-term average or a verified increase in defects.")
    if fixture:
        st.info("The real-data add-on is disabled while synthetic model fixtures are selected.")
        return
    try:
        predictions, metrics, meta = load_experiment()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(f"Optional experiment unavailable: {exc}")
        st.code("python -m experiments.complaint_change.build")
        return
    st.warning("Exploratory comparison: only TWO August training windows (Aug 15 and 22 starts). "
               "All trained baselines below are refitted on those same rows. Compare within this panel, "
               "not directly against the original three-window model as a feature-effect test.")
    period = st.radio("Add-on evaluation", ["All four weeks", "Selected week"], horizontal=True, key="change_scope")
    scope = metrics.loc[metrics.scope.eq("overall")].copy() if period == "All four weeks" else metrics.loc[
        metrics.scope.eq("week") & metrics.forecast_start.eq(forecast_week)].copy()
    scope["Model"] = scope.model.map(LABELS)
    st.bar_chart(scope.set_index("Model")[["mae"]].rename(columns={"mae": "MAE · lower is better"}),
                 horizontal=True, color="#176B5B")
    st.dataframe(scope[["Model", "mae", "top_k_coverage_pct", "top_k_collisions", "citywide_collisions"]],
                 hide_index=True, width="stretch", column_config={
                     "mae": st.column_config.NumberColumn("MAE", format="%.4f"),
                     "top_k_coverage_pct": st.column_config.NumberColumn("Top-50 coverage (%)", format="%.2f"),
                 })
    st.caption("Coverage uses the same citywide denominator, fixed eligible cells, and top-50 tie-break "
               "as the main backend. No September outcomes were used to fit the models; September had already been explored.")
    selected = st.session_state.get("selected_area")
    area = predictions.loc[predictions.forecast_start.eq(forecast_week) & predictions.area_id.eq(selected)]
    if not area.empty:
        st.markdown(f"**Selected area: {selected}**")
        st.dataframe(area[["previous_feature_start", "previous_feature_end", "feature_start", "feature_end",
                           "previous_complaints", "past_complaints", "complaint_change",
                           "complaint_change_2w_prediction", "actual_collisions"]],
                     hide_index=True, width="stretch")
    with st.expander("Add-on method and saved results"):
        st.write("Each count feature uses log1p; signed weekly change uses sign(x) × log1p(abs(x)). "
                 "StandardScaler and Poisson regression fit training rows only. Regularization is fixed at alpha=1.")
        st.json({"training_windows": meta["train_starts"], "rows": meta["rows"],
                 "pooled_comparisons": meta["comparisons"], "limitations": meta["limitations"]})
