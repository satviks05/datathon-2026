"""Materialize the August-only training files and September-only actuals."""

import json
from pathlib import Path

import pandas as pd

from . import config
from .loaders import input_version, load_events


def split_inputs(data_dir=config.DATA_DIR, output_dir=None):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir or data_dir)
    reports, crashes = load_events(data_dir)
    cutoff = pd.Timestamp(config.FORECAST_START)
    manifest = {"source_input_version": input_version(data_dir), "periods": {}}
    for name, is_train in (("august_train", True), ("september_actuals", False)):
        folder = output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        report_mask = reports["created_date"].lt(cutoff)
        crash_mask = crashes["occurred_at"].lt(cutoff)
        if not is_train:
            report_mask, crash_mask = ~report_mask, ~crash_mask
        selected_reports = reports.loc[report_mask].copy()
        if is_train:
            # These source attributes can reflect later September actions.
            # Remove them entirely from the training snapshot, not just features.
            selected_reports = selected_reports.drop(columns=["closed_date", "status"], errors="ignore")
        selected_crashes = crashes.loc[crash_mask].drop(columns=["occurred_at", "has_harm"])
        selected_reports.to_csv(folder / "signal_complaints_clean.csv", index=False)
        selected_crashes.to_csv(folder / "collisions_clean.csv", index=False)
        manifest["periods"][name] = dict(signal_reports=len(selected_reports), collisions=len(selected_crashes),
                                       role="training and forecast inputs" if is_train else "comparison only; never model inputs")
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
