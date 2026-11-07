"""Train and forecast without access to September: python -m src.train."""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config
from .loaders import input_version, load_events
from .regression import AugustOnlyRegression
from .spatial import build_candidates


def train_and_forecast(train_dir, output_dir, *, include_daily=False):
    reports, crashes = load_events(train_dir)
    model = AugustOnlyRegression(reports, crashes, build_candidates(reports))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecasts = [model.aggregate(config.FORECAST_START, config.FORECAST_END, radius) for radius in config.RADII]
    pd.concat(forecasts, ignore_index=True).to_csv(output / "september_forecasts.csv", index=False)
    manifest = dict(training_input_version=input_version(train_dir),
                    training_directory=str(Path(train_dir).resolve()), september_observations_used=False,
                    models={str(radius): model.description(radius) for radius in config.RADII})
    (output / "forecast_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if include_daily:
        pd.concat([model.daily_forecasts(radius) for radius in config.RADII], ignore_index=True).to_csv(
            output / "september_daily_forecasts.csv", index=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=config.DATA_DIR / "august_train")
    parser.add_argument("--output-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--daily", action="store_true")
    args = parser.parse_args()
    result = train_and_forecast(args.train_dir, args.output_dir, include_daily=args.daily)
    print(json.dumps(dict(training_input_version=result["training_input_version"], september_observations_used=False)))


if __name__ == "__main__":
    main()
