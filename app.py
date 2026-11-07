"""Run from the repository root: streamlit run app.py."""

import pandas as pd
import streamlit as st

from ui.data import (
    STRATEGIES, TARGET_COLUMNS, EVALUATION_COLUMNS, area_evidence,
    development_fixture, input_version, load_export, load_source_events,
    select_configuration,
)
from ui.map_view import make_map


st.set_page_config(page_title="Signal Review | NYC", page_icon="🚦", layout="wide")
st.markdown("""<style>
    .block-container {padding-top: 2rem; max-width: 1550px;}
    h1 {letter-spacing: -0.045em; font-weight: 700;}
    h2, h3 {letter-spacing: -0.025em;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e2e8ed;
        border-radius: 12px; padding: 16px;}
    .eyebrow {font-size: .75rem; letter-spacing: .16em; color: #176b5b; font-weight: 700;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">NYC IN MOTION / HISTORICAL SAFETY REVIEW</div>', unsafe_allow_html=True)
st.title("A closer look at signal safety")
st.markdown("Explore reported-signal areas alongside nearby injury and fatal crashes. "
            "Use August evidence to compare review priorities against September outcomes.")
st.caption("AUG–SEP 2025 · NYC local time · Review areas, not verified individual intersections")

with st.sidebar:
    st.markdown("### Review settings")
    radius = st.slider("Evidence radius (meters)", 100, 250, 250, 25, key="radius_m",
                       help="Changes neighborhood evidence, rankings, and evaluation together.")
    strategy = st.selectbox("Ranking strategy", list(STRATEGIES),
                            format_func=STRATEGIES.get, key="ranking_strategy")
    budget = st.selectbox("Shortlist size (review areas)", [25, 50, 100], index=1, key="shortlist_budget",
                         help="A scenario count, not an estimate of available crews.")
    st.caption("Fixed grouping: 50 m grid\n\nFixed target spacing: 250 m")
    st.divider()
    st.markdown("### Map layers")
    show_reports = st.checkbox("Signal reports", True, key="show_reports")
    show_crashes = st.checkbox("Injury or fatal crashes", True, key="show_crashes")
    map_period = st.radio("Source-event period", ["August", "September"], horizontal=True, key="map_period")
    st.caption("The map period changes source-event layers only. Ranking inputs remain August.")
    st.divider()
    fixture = st.checkbox("Development: show sample shortlist", False, key="fixture_mode",
                          help="Synthetic layout examples. Not analysis or recommendations.")

try:
    reports, crashes = load_source_events()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot load cleaned source data: {exc}")
    st.code("python pipelines/clean_data.py data/raw data/processed")
    st.stop()

version = input_version()
selected = pd.DataFrame(columns=sorted(TARGET_COLUMNS))
evaluation = pd.DataFrame(columns=sorted(EVALUATION_COLUMNS))
export_errors = []
for filename, required in (("selected_targets.csv", TARGET_COLUMNS), ("evaluation.csv", EVALUATION_COLUMNS)):
    try:
        table, notice = load_export(filename, required)
        if notice:
            export_errors.append(notice)
        if filename == "selected_targets.csv":
            selected = select_configuration(table, radius, budget, strategy)
        else:
            evaluation = select_configuration(table, radius, budget)
    except (OSError, ValueError, KeyError) as exc:
        export_errors.append(str(exc))

if fixture:
    selected = development_fixture(radius, budget, strategy)
    st.warning("DEVELOPMENT FIXTURE — locations, scores, and shortlist counts are synthetic layout "
               "examples. They do not recalculate from source events and are not research findings. "
               "Evaluation and recommendation downloads are disabled.")
elif selected.empty:
    st.info("Source data is ready. No backend shortlist is available for these settings yet. "
            "The map shows real source events; ranking and evaluation are pending.")
else:
    st.caption("Loaded backend exports. The backend must regenerate all outputs after input or "
               "method changes; this UI does not certify export provenance.")

aug_reports = reports.loc[reports["created_date"].dt.month.eq(8)]
sep_harm = crashes.loc[crashes["occurred_at"].dt.month.eq(9) & crashes["has_harm"]]
metrics = st.columns(4)
metrics[0].metric("August signal reports · mapped", f"{len(aug_reports):,}")
metrics[1].metric("September harm crashes · mapped", f"{len(sep_harm):,}")
metrics[2].metric("Sample areas" if fixture else "Selected review areas", str(len(selected)) if not selected.empty else "—")
metrics[3].metric("Evidence radius", f"{radius} m")
st.caption(f"{STRATEGIES[strategy]} · Requested shortlist: {budget} areas · "
           "August inputs / September evaluation · Harm = injury OR fatality")

map_tab, compare_tab, quality_tab = st.tabs(["Explore areas", "Compare strategies", "Data & method"])

with map_tab:
    left, right = st.columns([1.7, 1], gap="large")
    chosen = None
    with right:
        st.subheader("Sample shortlist" if fixture else "Review shortlist")
        if not selected.empty:
            selected = selected.sort_values(["selected_rank", "target_id"])
            ids = selected["target_id"].astype(str).tolist()
            if st.session_state.get("selected_target") not in ids:
                st.session_state["selected_target"] = ids[0]
            chosen = st.selectbox("Inspect an area", ids, key="selected_target")
            display = selected[["selected_rank", "target_id", "signal_report_count", "harm_crash_count", "score"]].rename(columns={
                "selected_rank": "Rank", "target_id": "Area", "signal_report_count": "Aug reports",
                "harm_crash_count": "Aug harm crashes", "score": "Priority score",
            })
            st.dataframe(display, hide_index=True, width="stretch",
                         column_config={"Priority score": st.column_config.NumberColumn(format="%.3f")})
            if not fixture:
                st.download_button("Download shortlist CSV", selected.to_csv(index=False).encode(),
                                   f"shortlist_{strategy}_{radius}m_k{budget}.csv", "text/csv")
            st.caption("Scores order review priorities; they are not crash probabilities. "
                       "Actual selection can be smaller than the requested shortlist.")
        else:
            st.caption("The backend's selected_targets.csv will populate this table. "
                       "You can preview the layout using the development checkbox.")
    with left:
        st.subheader("Reported-signal areas")
        month = 8 if map_period == "August" else 9
        map_reports = reports.loc[reports["created_date"].dt.month.eq(month)]
        map_crashes = crashes.loc[crashes["occurred_at"].dt.month.eq(month) & crashes["has_harm"]]
        st.pydeck_chart(make_map(
            map_reports if show_reports else map_reports.iloc[:0],
            map_crashes if show_crashes else map_crashes.iloc[:0], selected, radius, chosen,
        ), width="stretch", height=490)
        st.caption(f"● Green: {'sample' if fixture else 'selected'} review areas / evidence circles · "
                   f"● Blue: {map_period} signal reports · ● Orange: {map_period} injury/fatal crashes. "
                   "Source layers show the full month, not only matched events. Basemap: CARTO / OpenStreetMap.")

    st.divider()
    st.subheader("Area evidence")
    if chosen is None:
        st.caption("Select a review area once backend results are available.")
    else:
        row = selected.loc[selected["target_id"].eq(chosen)].iloc[0]
        st.markdown(f"**{chosen}** · {row['latitude']:.5f}, {row['longitude']:.5f} · {radius} m")
        cols = st.columns(3)
        cols[0].metric("Sample reports" if fixture else "August signal reports", int(row["signal_report_count"]))
        cols[1].metric("Sample harm crashes" if fixture else "August harm crashes", int(row["harm_crash_count"]))
        cols[2].metric("Reported component", str(row["top_descriptor"]))
        st.write(row["suggested_inspection"])
        st.caption("Reported component labels are not verified diagnoses. Historical reports do not "
                   "establish a current defect or show that signals caused nearby crashes.")
        evidence_period = st.radio("Evidence period", ["august", "september"], horizontal=True, key="evidence_period",
                                   format_func=lambda x: "August · ranking evidence" if x == "august" else "September · evaluation evidence")
        if fixture:
            st.info("Sample areas have no linked source evidence. Disable the fixture to inspect real backend results.")
        else:
            try:
                records = area_evidence(chosen, radius, evidence_period, version)
                if records is None:
                    st.info("Event details await src.api.get_area_evidence(). No nearby records are inferred by the UI.")
                else:
                    outcomes = records["collisions"]
                    context_columns = {
                        "number_of_persons_injured": "Injured people · nearby records",
                        "number_of_persons_killed": "Deaths · nearby records",
                        "number_of_cyclist_injured": "Cyclist injuries · nearby records",
                    }
                    if not outcomes.empty and set(context_columns).issubset(outcomes.columns):
                        context = st.columns(3)
                        for column, (field, label) in zip(context, context_columns.items()):
                            column.metric(label, int(pd.to_numeric(outcomes[field], errors="raise").sum()))
                        st.caption(f"{evidence_period.title()} context totals from backend-returned records. "
                                   "Cyclist injuries are part of total injuries, not an additional category.")
                    a, b = st.columns(2)
                    with a:
                        st.markdown("**Signal reports · original records**")
                        st.dataframe(records["reports"], hide_index=True, width="stretch")
                    with b:
                        st.markdown("**Collisions · original records and outcome flags**")
                        st.dataframe(records["collisions"], hide_index=True, width="stretch")
            except Exception as exc:
                st.error(f"Backend evidence could not be loaded: {exc}")

with compare_tab:
    st.subheader("Three strategies. One comparison setting.")
    st.caption(f"Radius {radius} m · Requested shortlist {budget} · August selection / September outcomes")
    if fixture:
        st.info("Evaluation is disabled for development fixtures.")
    elif evaluation.empty:
        st.info("Awaiting backend evaluation.csv for this radius and shortlist size.")
    else:
        labels = evaluation.copy()
        labels["Strategy"] = labels["strategy"].map(STRATEGIES)
        st.bar_chart(labels.set_index("Strategy")[["september_harm_coverage_pct"]].rename(
            columns={"september_harm_coverage_pct": "September harmful crashes covered (%)"}),
            color="#176B5B", horizontal=True)
        st.dataframe(labels[["Strategy", "selected_count", "september_harm_crashes_covered",
                              "september_harm_coverage_pct", "august_signal_reports_covered"]],
                     hide_index=True, width="stretch")
        missing = set(STRATEGIES) - set(evaluation["strategy"])
        if missing:
            st.warning("Comparison is incomplete; missing: " + ", ".join(STRATEGIES[s] for s in sorted(missing)))
        st.download_button("Download comparison CSV", evaluation.to_csv(index=False).encode(),
                           f"comparison_{radius}m_k{budget}.csv", "text/csv")
    st.markdown(f"Coverage denominator: **{len(sep_harm):,} mapped September injury/fatal crashes**. "
                "The backend must count each crash once across overlapping selected circles.")
    st.caption("Compare strategies at the same radius and shortlist size. Larger circles mechanically "
               "cover more events. September evaluation is exploratory, not an untouched test set. "
               "August report coverage is an in-sample burden measure.")

with quality_tab:
    st.subheader("What these records can tell us")
    st.dataframe(pd.DataFrame([
        ["DOT 311 reports", 34676, "6,534 traffic signal reports; 6,119 mapped; 415 excluded (6.35%)"],
        ["Police-reported crashes", 14526, "14,066 mapped; 119 missing coordinates; 341 zero-coordinate pairs"],
    ], columns=["Source", "Original rows", "Included scope and coordinate exclusions"]),
        hide_index=True, width="stretch")
    st.caption("Source audit counts apply to the original supplied bundle. Current loaded snapshots: "
               f"{len(reports):,} reports / {len(crashes):,} collisions. Re-audit if inputs change.")
    st.markdown("""
**Method.** August reports define candidates on a 50 m grid. The backend ranks report and harmful-crash
counts by percentile; combined gives each percentile equal weight. Shortlists use fixed 250 m spacing.
The radius controls evidence neighborhoods, not the grouping grid or spacing.

**Limits.** Broad coordinate screening is not intersection verification. Reports are not verified
defects; request closure does not prove repair. Nearby crashes do not establish signal causation.
There is no traffic-exposure denominator, complete signal inventory, or evidence of injuries prevented.
The shortlist size is a review scenario, not a claim about staffing capacity.

**Sources.** [NYC 311](https://data.cityofnewyork.us/d/erm2-nwe9) ·
[NYPD Motor Vehicle Collisions](https://data.cityofnewyork.us/d/h9gi-nx95).
The two challenge CSVs are the only analytical inputs.
""")
    with st.expander("Backend connection status", expanded=bool(export_errors)):
        for message in export_errors:
            st.write(message)
        st.caption("Expected exports: data/processed/selected_targets.csv and evaluation.csv. "
                   "Expected evidence helper: src.api.get_area_evidence(target_id, radius_m, period).")
        st.caption("Input/export file signatures and backend Python changes invalidate UI caches. "
                   "Restart Streamlit after changing backend implementation code.")
