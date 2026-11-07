"""Authoritative weekly/pooled evaluation with citywide coverage denominators."""

import numpy as np
import pandas as pd

from . import forecast_config as cfg
from .forecast_features import count_by_cell, in_window


def coverage(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def top_cells(frame, model):
    return frame.sort_values([f"{model}_prediction", "area_id"], ascending=[False, True],
                             kind="stable").head(cfg.TOP_K)


def evaluate(predictions, crashes):
    if predictions.empty or predictions.run_id.nunique() != 1 or predictions.run_id.isna().any():
        raise ValueError("Expected one nonempty prediction run")
    if predictions.duplicated(["area_id", "forecast_start"]).any():
        raise ValueError("Duplicate evaluation area/week")
    if not predictions.activity_band.isin(cfg.BANDS).all():
        raise ValueError("Unknown activity band")
    if predictions.groupby("area_id").activity_band.nunique().gt(1).any():
        raise ValueError("Activity bands must remain fixed across weeks")
    numeric = predictions[["actual_collisions", *[f"{m}_prediction" for m in cfg.MODELS]]].to_numpy()
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Invalid evaluation counts/predictions")
    if not predictions.actual_collisions.mod(1).eq(0).all():
        raise ValueError("Actual collision counts must be integers")
    universe = set(predictions.area_id)
    run_id = predictions.run_id.iloc[0]
    rows = []
    weekly = {model: [] for model in cfg.MODELS}
    for start, frame in predictions.groupby("forecast_start", sort=True):
        if set(frame.area_id) != universe or frame.forecast_end.nunique() != 1:
            raise ValueError("Every evaluation week must contain the same eligible cells")
        end = frame.forecast_end.iloc[0]
        if pd.Timestamp(end) - pd.Timestamp(start) != pd.Timedelta(days=7):
            raise ValueError("Expected a seven-day target window")
        citywide = int(in_window(crashes, start, end).collision_id.nunique())
        expected = count_by_cell(crashes, "collision_id", start, end).reindex(frame.area_id, fill_value=0)
        if not np.array_equal(expected.to_numpy(), frame.actual_collisions.to_numpy()):
            raise ValueError("Prediction actual counts disagree with source collision IDs")
        eligible = int(frame.actual_collisions.sum())
        for model in cfg.MODELS:
            numerator = int(top_cells(frame, model).actual_collisions.sum())
            row = dict(run_id=run_id, model=model, scope="week", forecast_start=start,
                       activity_band=None, area_week_rows=len(frame),
                       mae=float(np.abs(frame.actual_collisions - frame[f"{model}_prediction"]).mean()),
                       top_k=cfg.TOP_K, top_k_collisions=numerator, citywide_collisions=citywide,
                       eligible_collisions=eligible, top_k_coverage_pct=coverage(numerator, citywide),
                       eligible_coverage_pct=coverage(eligible, citywide))
            rows.append(row)
            weekly[model].append(row)
    for model in cfg.MODELS:
        error = (predictions.actual_collisions - predictions[f"{model}_prediction"]).abs()
        sums = {name: sum(w[name] for w in weekly[model]) for name in (
            "top_k_collisions", "citywide_collisions", "eligible_collisions")}
        rows.append(dict(run_id=run_id, model=model, scope="overall", forecast_start=None,
                         activity_band=None, area_week_rows=len(predictions), mae=float(error.mean()),
                         top_k=cfg.TOP_K, **sums,
                         top_k_coverage_pct=coverage(sums["top_k_collisions"], sums["citywide_collisions"]),
                         eligible_coverage_pct=coverage(sums["eligible_collisions"], sums["citywide_collisions"])))
        for band in cfg.BANDS:
            mask = predictions.activity_band.eq(band)
            if mask.any():
                rows.append(dict(run_id=run_id, model=model, scope="activity_band", forecast_start=None,
                                 activity_band=band, area_week_rows=int(mask.sum()), mae=float(error[mask].mean())))
    result = pd.DataFrame(rows, columns=cfg.METRIC_COLUMNS)
    for column in ("area_week_rows", "top_k", "top_k_collisions", "citywide_collisions", "eligible_collisions"):
        result[column] = result[column].astype("Int64")
    return result


def compare_models(metrics):
    """Derived presentation comparisons; metrics.csv remains authoritative."""
    comparisons = []
    for scope in ("week", "overall"):
        selected = metrics.loc[metrics.scope.eq(scope)]
        for start in selected.forecast_start.fillna("").unique():
            frame = selected.loc[selected.forecast_start.fillna("").eq(start)].set_index("model")
            combined = frame.loc["combined"]
            for reference in ("persistence", "collision_only"):
                ref = frame.loc[reference]
                gain = float(ref.mae - combined.mae)
                pct = 100.0 * gain / ref.mae if ref.mae else None
                delta = (combined.top_k_coverage_pct - ref.top_k_coverage_pct
                         if pd.notna(combined.top_k_coverage_pct) and pd.notna(ref.top_k_coverage_pct) else None)
                comparisons.append(dict(scope=scope, forecast_start=start or None, reference=reference,
                                        mae_reduction=gain, relative_mae_improvement_pct=pct,
                                        top_k_coverage_difference_pp=float(delta) if pd.notna(delta) else None))
    return comparisons
