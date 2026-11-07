"""Map display encodings; marker sizes are pixels, not geographic radii."""

import numpy as np
import pydeck as pdk

from ui.data import map_values


def map_frame(rows, model, measure, scale_max):
    points = rows.copy()
    points["display_count"] = map_values(rows, model, measure)
    fraction = points["display_count"].to_numpy(dtype=float) / scale_max
    fraction = np.clip(fraction, 0, 1)
    points["marker_pixels"] = 3 + 25 * np.sqrt(fraction)
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
        get_fill_color="color", get_radius="marker_pixels", radius_units="pixels", pickable=True,
    )]
    selected = points.loc[points["area_id"].eq(selected_id)].copy()
    if not selected.empty:
        selected["outline_pixels"] = selected["marker_pixels"] + 3
        layers.append(pdk.Layer(
            "ScatterplotLayer", selected, id="selected-cell", get_position="[longitude, latitude]",
            get_radius="outline_pixels", radius_units="pixels", filled=False, stroked=True,
            get_line_color=[19, 50, 67, 255], line_width_min_pixels=2, pickable=False,
        ))
    return pdk.Deck(
        layers=layers, map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=40.715, longitude=-73.96, zoom=10.3, pitch=0),
        tooltip={"text": "{area_id}\nDisplayed count: {display_label}\nForecast: {forecast_start} → {forecast_end} (end exclusive)"},
    )
