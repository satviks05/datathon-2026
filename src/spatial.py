"""Meter distances and bounded spatial lookups using NumPy, with no GIS service."""

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config

DISTANCE_TOLERANCE_M = 1e-7


def haversine_m(latitude, longitude, coordinates):
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    origin = np.radians([latitude, longitude])
    points = np.radians(coordinates)
    delta = points - origin
    a = np.sin(delta[:, 0] / 2) ** 2 + np.cos(origin[0]) * np.cos(points[:, 0]) * np.sin(delta[:, 1] / 2) ** 2
    return 2 * config.EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class RadiusIndex:
    """Angular buckets prefilter points; exact haversine decides membership.

    Query bounds enclose the spherical cap, including its longitude extrema.
    This index is intentionally scoped to the validated NYC input coordinates.
    """

    def __init__(self, coordinates):
        self.coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 2)
        self.width = math.degrees(max(config.RADII) / config.EARTH_RADIUS_M)
        cells = np.floor(self.coordinates / self.width).astype(np.int64)
        buckets = defaultdict(list)
        for index, cell in enumerate(cells):
            buckets[tuple(cell)].append(index)
        self.buckets = {cell: np.asarray(indices, dtype=np.int64) for cell, indices in buckets.items()}

    def query(self, latitude, longitude, radius_m):
        angle = (radius_m + DISTANCE_TOLERANCE_M) / config.EARTH_RADIUS_M
        lat_span = math.degrees(angle)
        lon_span = math.degrees(math.asin(min(1, math.sin(angle) / math.cos(math.radians(latitude)))))
        lower = np.floor(np.array([latitude - lat_span, longitude - lon_span]) / self.width).astype(int)
        upper = np.floor(np.array([latitude + lat_span, longitude + lon_span]) / self.width).astype(int)
        blocks = [self.buckets[(x, y)]
                  for x in range(lower[0], upper[0] + 1)
                  for y in range(lower[1], upper[1] + 1)
                  if (x, y) in self.buckets]
        if not blocks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
        indices = np.sort(np.concatenate(blocks))
        distances = haversine_m(latitude, longitude, self.coordinates[indices])
        keep = distances <= radius_m + DISTANCE_TOLERANCE_M
        return indices[keep], distances[keep]


def build_candidates(reports):
    """Fixed August catalog; grouping uses a documented local metric transform.

    Equirectangular x/y at 40.7 N, -74.0 E provide nominal 50 m cells.
    This approximate projection is used ONLY for grouping; radius and spacing
    decisions always use haversine meters. No street/intersection validation.
    """
    august = reports.loc[reports["created_date"].lt(pd.Timestamp("2025-09-01"))].copy()
    if august.empty:
        return pd.DataFrame(columns=["target_id", "latitude", "longitude"])
    x = config.EARTH_RADIUS_M * np.radians(august["longitude"] - config.GRID_ORIGIN_LON) * math.cos(math.radians(config.GRID_ORIGIN_LAT))
    y = config.EARTH_RADIUS_M * np.radians(august["latitude"] - config.GRID_ORIGIN_LAT)
    august["cell_x"] = np.floor(x / config.GRID_M).astype(int)
    august["cell_y"] = np.floor(y / config.GRID_M).astype(int)
    frequencies = august.groupby(["cell_x", "cell_y", "latitude", "longitude"]).size().reset_index(name="frequency")
    representatives = frequencies.sort_values(
        ["cell_x", "cell_y", "frequency", "latitude", "longitude"],
        ascending=[True, True, False, True, True], kind="stable",
    ).drop_duplicates(["cell_x", "cell_y"])
    representatives["target_id"] = [f"nyc50_{x}_{y}" for x, y in zip(representatives["cell_x"], representatives["cell_y"])]
    return representatives[["target_id", "latitude", "longitude"]].sort_values("target_id", kind="stable").reset_index(drop=True)
