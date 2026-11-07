"""Map display encodings; marker sizes are pixels, not geographic radii."""

import numpy as np
import pandas as pd
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


def source_map_frame(frame, kind, view="Area counts"):
    """Presentation-only approximate 500 m bins; never used for model features."""
    if frame.empty:
        return pd.DataFrame()
    points = frame.copy()
    points["kind"] = kind
    if kind == "Signal report":
        points["timestamp"] = pd.to_datetime(points["created_date"])
        points["event_detail"] = points.apply(lambda row:
            f"Signal report\nCreated: {row['timestamp']:%b %d, %Y at %I:%M %p}\n"
            f"Type: {row['complaint_type']}\nStatus (source snapshot): {row['status']}\n"
            f"Component: {row['descriptor']}", axis=1)
    else:
        points["timestamp"] = pd.to_datetime(
            points["crash_date"].astype(str).str[:10] + " " + points["crash_time"].astype(str))
        points["event_detail"] = points.apply(lambda row:
            f"Collision\nOccurred: {row['timestamp']:%b %d, %Y at %I:%M %p}\n"
            f"Injured people: {row['number_of_persons_injured']}\n"
            f"Deaths: {row['number_of_persons_killed']}", axis=1)
    if view == "Individual events":
        points["display_count"] = 1
        return points

    # Local equirectangular display grid at NYC's reference latitude. This is
    # deliberately separate from the backend's exact EPSG:32618 prediction cells.
    meters_per_radian = 6_371_008.8
    lon_scale = meters_per_radian * np.cos(np.deg2rad(40.7))
    points["display_x"] = np.floor(np.deg2rad(points["longitude"]) * lon_scale / 500).astype(int)
    points["display_y"] = np.floor(np.deg2rad(points["latitude"]) * meters_per_radian / 500).astype(int)
    groups = []
    for (x, y), records in points.groupby(["display_x", "display_y"], sort=True):
        start, end = records["timestamp"].min(), records["timestamp"].max()
        count = len(records)
        detail = (f"{count} {'signal reports' if kind == 'Signal report' else 'collisions'} in this display area\n"
                  f"Dates: {start:%b %d, %Y %I:%M %p} – {end:%b %d, %Y %I:%M %p}\n")
        if kind == "Signal report":
            statuses = records["status"].fillna("Unknown").value_counts()
            detail += "Statuses (source snapshot): " + ", ".join(f"{label}: {n}" for label, n in statuses.items())
            detail += "\nType: " + ", ".join(records["complaint_type"].dropna().unique())
        else:
            detail += (f"Injured people: {int(records['number_of_persons_injured'].sum())}\n"
                       f"Deaths: {int(records['number_of_persons_killed'].sum())}")
        detail += "\nApprox. 500 m display bin · observed events, not predicted risk"
        groups.append(dict(latitude=np.rad2deg((y + .5) * 500 / meters_per_radian),
                           longitude=np.rad2deg((x + .5) * 500 / lon_scale),
                           display_count=count, event_detail=detail, kind=kind))
    return pd.DataFrame(groups)


def make_source_map(reports, collisions, view="Area counts"):
    """Show concentration circles by default, with an individual-event detail view."""
    layers = []
    report_points = source_map_frame(reports, "Signal report", view)
    crash_points = source_map_frame(collisions, "Collision", view)
    maximum = max((float(frame["display_count"].max()) for frame in (report_points, crash_points)
                   if not frame.empty), default=1)
    for name, frame, color in (
        ("Signal report", report_points, [35, 132, 158, 180]),
        ("Collision", crash_points, [218, 99, 55, 140]),
    ):
        if frame.empty:
            continue
        frame = frame.copy()
        frame["marker_pixels"] = (5 + 24 * np.sqrt(frame["display_count"] / maximum)
                                  if view == "Area counts" else 4)
        # Outlined report circles stay distinguishable from filled crash circles.
        layers.append(pdk.Layer(
            "ScatterplotLayer", frame, id=name.replace(" ", "-"),
            get_position="[longitude, latitude]",
            get_fill_color=[*color[:3], 45] if name == "Signal report" and view == "Area counts" else color,
            get_line_color=color, filled=True, stroked=True,
            line_width_min_pixels=2, get_radius="marker_pixels", radius_units=pdk.types.String("pixels"),
            radius_min_pixels=3, radius_max_pixels=29, pickable=True,
        ))
    return pdk.Deck(layers=layers, map_provider="carto", map_style="light",
                    initial_view_state=nyc_view(), tooltip={"text": "{event_detail}"})
