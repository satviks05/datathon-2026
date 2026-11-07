"""Reproduce the seven-day regression backend handoff: python -m src.build."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import pandas as pd
import numpy as np

from . import config
from .api import _engine, get_config, get_data_quality


def _atomic_csv(table, path):
    temporary = path.with_suffix(".tmp.csv")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def build_outputs(data_dir=config.DATA_DIR, output_dir=config.DATA_DIR):
    started = perf_counter()
    engine = _engine(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_scored, all_selected, all_evaluation = [], [], []
    for radius in config.RADII:
        for strategy in config.STRATEGIES:
            for budget in config.BUDGETS:
                scored, selected = engine.scored_and_selected(
                    config.DEFAULT_START_DATE, config.DEFAULT_END_DATE, radius, strategy, budget,
                )
                if budget == config.BUDGETS[0]:
                    all_scored.append(scored)
                all_selected.append(selected)
        for budget in config.BUDGETS:
            all_evaluation.append(engine.historical_evaluation(radius, budget))
    tables = {
        "targets_scored.csv": pd.concat(all_scored, ignore_index=True),
        "selected_targets.csv": pd.concat(all_selected, ignore_index=True),
        "evaluation.csv": pd.concat(all_evaluation, ignore_index=True),
        "model_metrics.csv": pd.concat([engine.regression.test_metrics(radius) for radius in config.RADII], ignore_index=True),
        "model_predictions.csv": pd.concat([engine.regression.weekly_predictions(radius) for radius in config.RADII], ignore_index=True),
    }
    for filename, table in tables.items():
        _atomic_csv(table, output_dir / filename)
    manifest = dict(
        input_version=engine.version, method_version=config.METHOD_VERSION,
        runtime=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__),
        config=get_config(data_dir=data_dir), data_quality=get_data_quality(data_dir=data_dir),
        ranking_period=[config.DEFAULT_START_DATE, config.DEFAULT_END_DATE],
        evaluation_period=["2025-09-01", "2025-09-28"],
        evaluation_excluded_dates=["2025-09-29", "2025-09-30"],
        evaluation_exclusion_reason="Only complete nonoverlapping seven-day outcome weeks are evaluated.",
        models={str(radius): engine.regression.description(radius) for radius in config.RADII},
        candidate_period="August 2025; fixed across dates, strategies and radii",
        coverage_definition="Unique event IDs in the union of selected circles",
        evaluation_kind="August-fitted OLS; four complete September weeks; exploratory because September previously explored",
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
