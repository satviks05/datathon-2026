"""Run: python -m experiments.complaint_change.build [--output DIRECTORY]."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from src import forecast_config as cfg
from src.forecast_features import build_features, count_by_cell, load_forecast_events, sha256_file
from src.forecast_metrics import evaluate

OUTPUT = Path(__file__).resolve().parent / "results"
TRAIN_STARTS = ("2025-08-15", "2025-08-22")
MODEL_FEATURES = {
    "collision_only_2w": ["past_collisions"],
    "complaint_level_2w": ["past_collisions", "past_complaints"],
    "complaint_change_2w": ["past_collisions", "past_complaints", "complaint_change"],
}


def signed_count_transform(values):
    """Same log1p for nonnegative counts; signed log1p preserves decreases."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def build_panel(reports, crashes):
    panel = build_features(reports, crashes)
    # Fourteen days of history do not exist for the first canonical train window.
    panel = panel.loc[panel["split"].ne("train") | panel.forecast_start.isin(TRAIN_STARTS)].copy()
    if set(panel.loc[panel.split.eq("train"), "forecast_start"]) != set(TRAIN_STARTS):
        raise ValueError("Expected the two history-complete August training windows")
    panel["previous_feature_start"] = ""
    panel["previous_feature_end"] = panel["feature_start"]
    panel["previous_complaints"] = 0
    for start, index in panel.groupby("forecast_start").groups.items():
        previous_start = (pd.Timestamp(start) - pd.Timedelta(days=14)).date().isoformat()
        previous_end = (pd.Timestamp(start) - pd.Timedelta(days=7)).date().isoformat()
        if previous_start < "2025-08-01":
            raise ValueError("Unavailable history must not be replaced with zero")
        counts = count_by_cell(reports, "unique_key", previous_start, previous_end)
        panel.loc[index, "previous_feature_start"] = previous_start
        panel.loc[index, "previous_complaints"] = counts.reindex(panel.loc[index, "area_id"], fill_value=0).to_numpy()
    panel["previous_complaints"] = panel.previous_complaints.astype("int64")
    panel["complaint_change"] = panel.past_complaints - panel.previous_complaints
    return panel.reset_index(drop=True)


def fit_and_predict(panel, run_id):
    train = panel.loc[panel.split.eq("train")]
    evaluation = panel.loc[panel.split.eq("evaluation")].copy()
    if set(train.forecast_start) != set(TRAIN_STARTS) or not train.forecast_end.lt("2025-09-01").all():
        raise ValueError("Only the matched two August training windows are permitted")
    if not train.actual_collisions.gt(0).any():
        raise ValueError("Training needs positive collision counts")
    evaluation["run_id"] = run_id
    evaluation["persistence_prediction"] = evaluation.past_collisions.astype(float)
    settings = {}
    with threadpool_limits(limits=1):
        for name, columns in MODEL_FEATURES.items():
            pipeline = Pipeline([
                ("signed_log1p", FunctionTransformer(signed_count_transform)),
                ("scaler", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=1.0, max_iter=1000)),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(train[columns], train.actual_collisions)
            prediction = pipeline.predict(evaluation[columns])
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise ValueError("Invalid experiment predictions")
            evaluation[f"{name}_prediction"] = prediction
            settings[name] = dict(features=columns, alpha=1.0, max_iter=1000,
                training_rows=len(train), training_starts=list(TRAIN_STARTS),
                scaler_mean=pipeline.named_steps["scaler"].mean_.tolist(),
                coefficients=pipeline.named_steps["poisson"].coef_.tolist(),
                converged=True, transform="signed log1p -> StandardScaler (counts remain ordinary log1p)")
    return evaluation.reset_index(drop=True), settings


def experiment_metrics(predictions, crashes):
    # Reuse the authoritative evaluator twice, without mutating its config or
    # recomputing a different definition of coverage/MAE in the UI.
    adapter = predictions.copy()
    adapter["collision_only_prediction"] = adapter.collision_only_2w_prediction
    frames = []
    for experiment in ("complaint_level_2w", "complaint_change_2w"):
        adapter["combined_prediction"] = predictions[f"{experiment}_prediction"]
        metrics = evaluate(adapter, crashes)
        if frames:
            metrics = metrics.loc[metrics.model.eq("combined")].copy()
        metrics["model"] = metrics.model.replace({"collision_only": "collision_only_2w", "combined": experiment})
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def run(output=OUTPUT):
    output = Path(output)
    reports, crashes = load_forecast_events()
    panel = build_panel(reports, crashes)
    inputs = {name: sha256_file(cfg.DATA_DIR / name) for name in cfg.INPUT_NAMES}
    implementation = {name: sha256_file(Path(__file__).resolve().parents[2] / "src" / name)
                      for name in ("forecast_config.py", "forecast_features.py", "forecast_metrics.py")}
    implementation["experiment_build.py"] = sha256_file(__file__)
    run_id = "change-" + hashlib.sha256(json.dumps([inputs, implementation], sort_keys=True).encode()).hexdigest()
    predictions, settings = fit_and_predict(panel, run_id)
    metrics = experiment_metrics(predictions, crashes)
    overall = metrics.loc[metrics.scope.eq("overall")].set_index("model")
    change = overall.loc["complaint_change_2w"]
    comparisons = []
    for reference in ("collision_only_2w", "complaint_level_2w", "persistence"):
        baseline = overall.loc[reference]
        comparisons.append(dict(reference=reference,
            mae_reduction=float(baseline.mae - change.mae),
            relative_mae_improvement_pct=float(100 * (baseline.mae - change.mae) / baseline.mae) if baseline.mae else None,
            top50_coverage_difference_pp=float(change.top_k_coverage_pct - baseline.top_k_coverage_pct)))
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("features.csv", panel), ("predictions.csv", predictions), ("metrics.csv", metrics)):
        frame.to_csv(output / name, index=False, lineterminator="\n")
    metadata = dict(schema_version=1, experiment="week_over_week_complaint_change", run_id=run_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(), input_sha256=inputs,
        implementation_sha256=implementation,
        output_sha256={name: sha256_file(output / name) for name in ("features.csv", "predictions.csv", "metrics.csv")},
        feature_definition="complaint_change = last 7 days' reports - previous 7 days' reports",
        train_starts=list(TRAIN_STARTS), evaluation_starts=list(cfg.EVALUATION_STARTS),
        model_settings=settings, comparisons=comparisons,
        rows={"features": len(panel), "predictions": len(predictions), "metrics": len(metrics)},
        limitations=["Exploratory add-on after September outcomes were already examined.",
                     "Only two training windows; this is week-over-week change, not a stable long-term normal.",
                     "Baselines are refitted on identical two-window training rows; compare within this experiment.",
                     "Do not attribute changes relative to the original three-window model solely to new features.",
                     "Negative changes are retained; no data tuning, causal claims, or significance claims."])
    # Metadata is published last so consumers can verify output hashes.
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(overall[["mae", "top_k_coverage_pct"]].to_string())
    print(json.dumps({"rows": metadata["rows"], "comparisons": comparisons}, indent=2))
    return panel, predictions, metrics, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
