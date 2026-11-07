"""Split inputs, freeze August-only forecasts, then evaluate September."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from . import config
from .api import _engine, get_config, get_data_quality
from .data_split import split_inputs
from .train import train_and_forecast


def _atomic_csv(table, path):
    temporary = path.with_suffix(".tmp.csv")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_outputs(data_dir=config.DATA_DIR, output_dir=config.DATA_DIR):
    started = perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = split_inputs(data_dir, output_dir)
    # The training command receives ONLY this directory, with no September rows.
    forecast_manifest = train_and_forecast(output_dir / "august_train", output_dir)
    engine = _engine(data_dir)
    all_scored, all_selected, all_evaluation, all_metrics, all_predictions = [], [], [], [], []
    for radius in config.RADII:
        for strategy in config.STRATEGIES:
            for budget in config.BUDGETS:
                scored, selected = engine.scored_and_selected(
                    config.FORECAST_START, config.FORECAST_END, radius, strategy, budget,
                )
                if budget == config.BUDGETS[0]:
                    all_scored.append(scored)
                all_selected.append(selected)
        for budget in config.BUDGETS:
            all_evaluation.append(engine.historical_evaluation(radius, budget))
        all_metrics.append(engine.model_metrics(config.FORECAST_START, config.FORECAST_END, radius))
        all_predictions.append(engine.area_comparison(config.FORECAST_START, config.FORECAST_END, radius))
    tables = {
        "targets_scored.csv": pd.concat(all_scored, ignore_index=True),
        "selected_targets.csv": pd.concat(all_selected, ignore_index=True),
        "evaluation.csv": pd.concat(all_evaluation, ignore_index=True),
        "model_metrics.csv": pd.concat(all_metrics, ignore_index=True),
        "model_predictions.csv": pd.concat(all_predictions, ignore_index=True),
    }
    for filename, table in tables.items():
        _atomic_csv(table, output_dir / filename)
    manifest = dict(
        input_version=engine.version, training_input_version=forecast_manifest["training_input_version"],
        method_version=config.METHOD_VERSION,
        runtime=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        config=get_config(data_dir=data_dir), data_quality=get_data_quality(data_dir=data_dir), split=split,
        training_period=[config.MIN_DATE, config.MODEL_TRAIN_END],
        forecast_period=[config.FORECAST_START, config.FORECAST_END],
        september_observations_used_for_predictions=False,
        models={str(radius): engine.regression.description(radius) for radius in config.RADII},
        coverage_definition="Unique original event IDs in union of selected circles",
        evaluation_kind="Full September, August-only forecast; exploratory because September previously explored",
        rows={name: len(table) for name, table in tables.items()},
    )
    temporary = output_dir / "analysis_manifest.tmp.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "analysis_manifest.json")
    print(json.dumps(dict(output_dir=str(output_dir.resolve()), seconds=round(perf_counter() - started, 3),
                          rows=manifest["rows"], input_version=engine.version), indent=2))
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()
    build_outputs(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
