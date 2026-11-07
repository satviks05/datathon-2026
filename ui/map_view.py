"""Map rendering only; spatial membership belongs to the backend."""

import pandas as pd
import pydeck as pdk


def event_points(reports, crashes):
    pieces = []
    if not reports.empty:
        frame = reports[["latitude", "longitude"]].copy()
        frame["label"] = "Signal report · " + reports["unique_key"].astype(str)
        frame["detail"] = reports["descriptor"].fillna("Unspecified component").astype(str)
        frame["color"] = [[29, 118, 171, 145]] * len(frame)
        pieces.append(frame)
    if not crashes.empty:
        frame = crashes[["latitude", "longitude"]].copy()
        frame["label"] = "Harm crash · " + crashes["collision_id"].astype(str)
        frame["detail"] = (crashes["number_of_persons_injured"].astype(str) + " injured · "
                           + crashes["number_of_persons_killed"].astype(str) + " deaths")
        frame["color"] = [[216, 100, 53, 160]] * len(frame)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def make_map(reports, crashes, targets, radius_m, selected_id):
    layers = []
    events = event_points(reports, crashes)
    if not events.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer", events, id="source-events", get_position="[longitude, latitude]",
            get_fill_color="color", get_radius=8, radius_min_pixels=2,
            radius_max_pixels=5, pickable=True,
        ))
    if not targets.empty:
        circles = targets.copy()
        circles["label"] = "Review area · " + circles["target_id"].astype(str)
        circles["detail"] = str(radius_m) + " m evidence radius · August ranking"
        layers.append(pdk.Layer(
            "ScatterplotLayer", circles, id="evidence-radius", get_position="[longitude, latitude]",
            get_radius=radius_m, get_fill_color=[23, 107, 91, 20],
            get_line_color=[23, 107, 91, 150], stroked=True, line_width_min_pixels=1,
            pickable=True,
        ))
        circles["color"] = [
            [18, 68, 59, 255] if tid == selected_id else [23, 135, 112, 235]
            for tid in circles["target_id"]
        ]
        layers.append(pdk.Layer(
            "ScatterplotLayer", circles, id="review-targets", get_position="[longitude, latitude]",
            get_radius=24, radius_min_pixels=5, get_fill_color="color", pickable=True,
        ))
    return pdk.Deck(
        layers=layers, map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=40.722, longitude=-73.96, zoom=10.4, pitch=0),
        tooltip={"text": "{label}\n{detail}"},
    )
