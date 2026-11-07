"""Map display encodings; marker sizes are pixels, not geographic radii."""

import numpy as np
import pydeck as pdk

from ui.data import map_values


def nyc_view():
    return pdk.ViewState(latitude=40.715, longitude=-73.96, zoom=10.3,
                         min_zoom=9, max_zoom=16, pitch=0)


def map_frame(rows, model, measure, scale_max):
    points = rows.copy()
    points["display_count"] = map_values(rows, model, measure)
    fraction = points["display_count"].to_numpy(dtype=float) / scale_max
    fraction = np.clip(fraction, 0, 1)
    points["marker_pixels"] = 3 + 14 * np.sqrt(fraction)
    low, high = np.array([57, 139, 160]), np.array([226, 82, 45])
    colors = low[None, :] + fraction[:, None] * (high - low)[None, :]
    points["color"] = [[180, 190, 198, 135] if count == 0 else [*color.astype(int).tolist(), 195]
                       for count, color in zip(points["display_count"], colors)]
    points["display_label"] = points["display_count"].map(lambda count: f"{count:.2f}")
    return points


def make_map(rows, model, measure, scale_max, selected_id):
    points = map_frame(rows, model, measure, scale_max)
    layers = [pdk.Layer(
        "ScatterplotLayer", points, id="forecast-cells", get_position="[longitude, latitude]",
        get_fill_color="color", get_radius="marker_pixels", radius_units=pdk.types.String("pixels"),
        radius_min_pixels=3, radius_max_pixels=17, pickable=True,
    )]
    selected = points.loc[points["area_id"].eq(selected_id)].copy()
    if not selected.empty:
        selected["outline_pixels"] = selected["marker_pixels"] + 3
        layers.append(pdk.Layer(
            "ScatterplotLayer", selected, id="selected-cell", get_position="[longitude, latitude]",
            get_radius="outline_pixels", radius_units=pdk.types.String("pixels"),
            radius_max_pixels=20, filled=False, stroked=True,
            get_line_color=[19, 50, 67, 255], line_width_min_pixels=2, pickable=False,
        ))
    return pdk.Deck(
        layers=layers, map_provider="carto", map_style="light",
        initial_view_state=nyc_view(),
        tooltip={"text": "{area_id}\nDisplayed count: {display_label}\nForecast: {forecast_start} → {forecast_end} (end exclusive)"},
    )


def make_source_map(reports, collisions):
    """Display individual observed events without constructing model features."""
    layers = []
    for name, frame, id_column, color in (
        ("Signal report", reports, "unique_key", [35, 132, 158, 165]),
        ("Collision", collisions, "collision_id", [218, 99, 55, 175]),
    ):
        if frame.empty:
            continue
        frame = frame.copy()
        frame["event_label"] = name + " · " + frame[id_column].astype(str)
        layers.append(pdk.Layer(
            "ScatterplotLayer", frame, id=name.replace(" ", "-"),
            get_position="[longitude, latitude]", get_fill_color=color,
            get_radius=4, radius_units=pdk.types.String("pixels"),
            radius_min_pixels=3, radius_max_pixels=5, pickable=True,
        ))
    return pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                    initial_view_state=nyc_view(), tooltip={"text": "{event_label}"})
