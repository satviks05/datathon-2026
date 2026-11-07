"""Observed-data explorer; display grouping is independent of model analysis."""

from datetime import date, timedelta

import streamlit as st

from ui.data import source_events_for_week, week_label
from ui.map_view import make_source_map


def render_observations(forecast_week):
    st.subheader("NYC · observed signal reports and collisions")
    feature_start = (date.fromisoformat(forecast_week) - timedelta(days=7)).isoformat()
    st.caption(f"Observed records: {week_label(feature_start)} · preceding the selected forecast week. "
               "This map shows recorded events, not predictions.")
    source_view = st.radio("Map detail", ["Area counts", "Individual events"], horizontal=True,
                           key="source_view")
    layers = st.columns(2)
    show_reports = layers[0].checkbox("Show signal reports", True, key="source_reports")
    show_collisions = layers[1].checkbox("Show collisions", True, key="source_collisions")
    try:
        reports, crashes = source_events_for_week(forecast_week)
        st.pydeck_chart(make_source_map(reports if show_reports else reports.iloc[:0],
                                       crashes if show_collisions else crashes.iloc[:0], source_view),
                        width="stretch", height=520, key=f"source_map_{forecast_week}_{source_view}")
        st.caption(f"Blue: {len(reports):,} signal reports · Orange: {len(crashes):,} collisions.")
        if source_view == "Area counts":
            st.caption("Larger circles indicate more recorded events in an approximately 500 m display bin. "
                       "Blue outlines = signal reports; orange fill = collisions. Both layers share a count-size scale. "
                       "Toggle layers to inspect overlap. Display bins do not change model features or evaluation.")
        else:
            st.caption("Hover a point for dates, complaint type/status, or collision outcomes. "
                       "Equal-sized markers represent individual records.")
        with st.expander("Browse the underlying records"):
            st.markdown("**Signal reports**")
            st.dataframe(reports[["created_date", "complaint_type", "status", "descriptor", "unique_key"]],
                         hide_index=True, width="stretch")
            st.markdown("**Collisions**")
            st.dataframe(crashes[["crash_date", "crash_time", "number_of_persons_injured",
                                  "number_of_persons_killed", "collision_id"]], hide_index=True, width="stretch")
    except (OSError, ValueError, KeyError) as exc:
        st.warning("The observed-data map could not load the cleaned files.")
        st.caption(str(exc))
