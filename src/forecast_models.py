"""Fit once on the three August windows; apply frozen pipelines each week."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from . import forecast_config as cfg


def fit_models(features):
    train = features.loc[features["split"].eq("train")]
    if train.empty or set(train.forecast_start) != set(cfg.TRAIN_STARTS):
        raise ValueError("Expected exactly the three agreed August training windows")
    if not train.forecast_end.le("2025-09-01").all():
        raise ValueError("Training labels must precede September")
    if not train.actual_collisions.gt(0).any():
        raise ValueError("Poisson training needs a positive target count; no fallback model was agreed")
    models = {}
    with threadpool_limits(limits=1):
        for model, columns in cfg.MODEL_FEATURES.items():
            pipeline = Pipeline([
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scaler", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=1.0, max_iter=1000)),
            ])
            # A failed convergence is a failed build, never silently a usable fit.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(train[columns], train["actual_collisions"])
            models[model] = pipeline
    return models


def predict(features, models, run_id):
    evaluation = features.loc[features["split"].eq("evaluation")].copy()
    evaluation["run_id"] = run_id
    evaluation["persistence_prediction"] = evaluation["past_collisions"].astype(float)
    with threadpool_limits(limits=1):
        for model, columns in cfg.MODEL_FEATURES.items():
            evaluation[f"{model}_prediction"] = models[model].predict(evaluation[columns])
    values = evaluation[[f"{model}_prediction" for model in cfg.MODELS]].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Nonfinite or negative predictions; no rounding or clamping is permitted")
    return evaluation[cfg.PREDICTION_COLUMNS].reset_index(drop=True)


def model_settings(models):
    settings = {"persistence": {"method": "prediction = past_collisions"}}
    for name, pipeline in models.items():
        scaler, model = pipeline["scaler"], pipeline["poisson"]
        settings[name] = {
            "features": cfg.MODEL_FEATURES[name], "transform": "log1p -> StandardScaler",
            "estimator": "PoissonRegressor", "alpha": model.alpha, "max_iter": model.max_iter,
            "fit_intercept": model.fit_intercept, "tol": model.tol, "solver": model.solver,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "scaler_variance": scaler.var_.tolist(), "training_rows": int(scaler.n_samples_seen_),
            "coefficients": model.coef_.tolist(), "intercept": float(model.intercept_),
            "n_iter": int(model.n_iter_), "converged": True,
        }
    return settings
