"""Milestone 4 — train two reproducible RUL regression baselines on NASA FD001.

Safety/data-integrity:
 * training and validation subsets contain disjoint engine IDs
 * only the 130 current/past features created in Milestone 3 are inputs
 * variance filtering learns its mask on training engines only
 * NASA's official test files and final holdout labels are NEVER opened here
 * all validation plots describe held-out TRAINING trajectories, not NASA test
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.feature_engineering import feature_columns

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MODEL_FILES = {
    "Random Forest": "random_forest_fd001.joblib",
    "XGBoost": "xgboost_fd001.joblib",
}
TARGET_OPTIONS = {
    "capped": "rul_capped_125",
    "true": "rul_true",
}


def load_training_data(project_root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load ONLY past/current training features and separately exported train labels."""
    processed = Path(project_root) / "data" / "processed"
    fp, lp = processed / "train_fd001_features.csv", processed / "train_fd001_labels.csv"
    manifest_path = processed / "feature_manifest.json"
    for file in (fp, lp, manifest_path):
        if not file.is_file():
            raise FileNotFoundError(
                f"Required Milestone 3 file missing: {file}. Run run_feature_engineering.py first."
            )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = feature_columns()
    if manifest.get("milestone") != 3 or manifest.get("model_feature_columns") != features:
        raise ValueError("Feature manifest is not a compatible Milestone 3 export. Rerun feature engineering.")

    frame = pd.read_csv(fp)
    labels = pd.read_csv(lp)
    keys = ["engine_id", "cycle"]
    for label, df, required in (
        ("feature file", frame, [*keys, *features]),
        ("label file", labels, [*keys, "rul_true", "rul_capped_125"]),
    ):
        if not set(required).issubset(df.columns):
            raise ValueError(f"Missing columns in {label}: {sorted(set(required) - set(df.columns))}")
        if df.duplicated(keys).any():
            raise ValueError(f"Duplicate engine/cycle keys in {label}")

    if len(frame) != len(labels):
        raise ValueError("Feature and label row counts do not match")
    if not frame[keys].reset_index(drop=True).equals(labels[keys].reset_index(drop=True)):
        raise ValueError("Feature and label rows are misaligned; regenerate Milestone 3 outputs")
    if frame["engine_id"].nunique() < 4:
        raise ValueError("Need at least four different training engines for engine-group validation")
    if frame[keys].isna().any().any() or labels[["rul_true", "rul_capped_125"]].isna().any().any():
        raise ValueError("Missing identifier or target values")
    X = frame[features].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite numeric training feature found")
    if not np.isfinite(labels[["rul_true", "rul_capped_125"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite training target found")
    if (labels[["rul_true", "rul_capped_125"]] < 0).any().any():
        raise ValueError("Negative RUL target detected")
    return frame, labels, features


def engine_split(frame: pd.DataFrame, *, seed: int = SEED, validation_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Split complete histories; no engine is present in both subsets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
    indices = np.arange(len(frame))
    train_idx, val_idx = next(splitter.split(indices, groups=frame["engine_id"]))
    train_engines = set(frame.iloc[train_idx]["engine_id"].tolist())
    val_engines = set(frame.iloc[val_idx]["engine_id"].tolist())
    if train_engines & val_engines:
        raise AssertionError("Engine-ID leakage detected in train/validation split")
    return train_idx, val_idx


def create_models(*, seed: int = SEED, quick: bool = False) -> dict[str, Pipeline]:
    """Lightweight CPU models; pipelines fit variance masks on train engines ONLY."""
    rf_trees = 30 if quick else 90
    xgb_trees = 50 if quick else 200
    return {
        "Random Forest": Pipeline([
            ("variance", VarianceThreshold(threshold=0.0)),
            ("regressor", RandomForestRegressor(
                n_estimators=rf_trees, max_depth=16, min_samples_leaf=3,
                max_features=0.8, random_state=seed, n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("variance", VarianceThreshold(threshold=0.0)),
            ("regressor", XGBRegressor(
                n_estimators=xgb_trees, max_depth=5, learning_rate=0.065,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
                tree_method="hist", objective="reg:squarederror",
                random_state=seed, n_jobs=4,
            )),
        ]),
    }


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(y_true, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, pred))), 4),
        "r2": round(float(r2_score(y_true, pred)), 4),
    }


def train_and_export(project_root: Path = ROOT, *, target: str = "capped", quick: bool = False) -> tuple[dict, list[Path]]:
    """Fit/compare baselines and export validation evidence. No NASA test data used."""
    project_root = Path(project_root)
    if target not in TARGET_OPTIONS:
        raise ValueError(f"target must be one of {list(TARGET_OPTIONS)}")
    frame, labels, cols = load_training_data(project_root)
    target_name = TARGET_OPTIONS[target]
    tr_idx, va_idx = engine_split(frame)
    train = frame.iloc[tr_idx]
    val = frame.iloc[va_idx]
    X_train = train[cols]
    X_val = val[cols]
    y_train = labels.iloc[tr_idx][target_name].to_numpy(dtype=float)
    y_val = labels.iloc[va_idx][target_name].to_numpy(dtype=float)

    model_dir = project_root / "models"
    output_dir = project_root / "outputs"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = val[["engine_id", "cycle"]].copy().reset_index(drop=True)
    predictions["actual_rul"] = y_val
    metrics = {}
    importances = []
    paths = []
    print(f"Engine split: {train.engine_id.nunique()} train / {val.engine_id.nunique()} validation engines")
    print(f"Model target: {target_name}; {len(cols)} candidate features")

    for display_name, pipeline in create_models(quick=quick).items():
        print(f"Training {display_name}...", flush=True)
        pipeline.fit(X_train, y_train)
        raw_prediction = pipeline.predict(X_val)
        predicted = np.maximum(np.asarray(raw_prediction, dtype=float), 0.0)
        if target == "capped":
            predicted = np.minimum(predicted, 125.0)
        slug = "random_forest" if display_name == "Random Forest" else "xgboost"
        predictions[f"pred_{slug}"] = predicted
        metrics[display_name] = regression_metrics(y_val, predicted)
        selected_features = np.asarray(cols)[pipeline.named_steps["variance"].get_support()]
        importance_vals = pipeline.named_steps["regressor"].feature_importances_
        if len(selected_features) != len(importance_vals):
            raise AssertionError("Importance mapping mismatch after train-only feature filtering")
        importances.extend({
            "model": display_name, "feature": str(feature), "importance": float(importance)
        } for feature, importance in zip(selected_features, importance_vals))
        file_path = model_dir / MODEL_FILES[display_name]
        joblib.dump(pipeline, file_path, compress=3)
        paths.append(file_path)
        print(f"  MAE {metrics[display_name]['mae']:.2f} | RMSE {metrics[display_name]['rmse']:.2f} | R² {metrics[display_name]['r2']:.3f}")

    # Report the model with the lower validation RMSE; NASA test data remains untouched.
    winner = min(metrics, key=lambda model: metrics[model]["rmse"])
    prediction_path = output_dir / "validation_predictions.csv"
    importance_path = output_dir / "feature_importance.csv"
    predictions.sort_values(["engine_id", "cycle"]).to_csv(prediction_path, index=False)
    pd.DataFrame(importances).sort_values(["model", "importance"], ascending=[True, False]).to_csv(importance_path, index=False)
    paths.extend([prediction_path, importance_path])

    split_summary = {
        "seed": SEED,
        "split_strategy": "GroupShuffleSplit by engine_id (80/20 engines)",
        "training_engine_ids": sorted(map(int, train["engine_id"].unique())),
        "validation_engine_ids": sorted(map(int, val["engine_id"].unique())),
        "training_engines": int(train["engine_id"].nunique()),
        "validation_engines": int(val["engine_id"].nunique()),
        "training_rows": int(len(train)),
        "validation_rows": int(len(val)),
    }
    summary = {
        "project": "AeroPredict AI — NASA FD001",
        "milestone": 4,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": target_name,
        "target_description": (
            "RUL clipped to 125 cycles; higher physical RUL values treated as 125 for training/validation"
            if target == "capped" else "Uncapped known remaining life in cycles"
        ),
        "validation_metrics": metrics,
        "best_validation_model": winner,
        "model_files": MODEL_FILES,
        "feature_columns": cols,
        "candidate_feature_count": len(cols),
        "split": split_summary,
        "quick_training_mode": bool(quick),
        "disclaimer": (
            "Validation is by unseen engine histories from NASA FD001 training trajectories. "
            "Not evaluated on the official NASA test set; no real-world maintenance certification."
        ),
        "methodology": [
            "RUL targets and feature vectors stored in separate files and aligned by engine/cycle key.",
            "RUL targets and future failure-cycle data are excluded from model inputs.",
            "No random row-level train/validation split; complete engines are grouped.",
            "VarianceThreshold fitted only on training engines inside each model pipeline.",
            "Tree regressors do not require feature scaling.",
            "NASA official test/evaluation data were never loaded during training or model selection.",
            "Model quality figures refer to validation observations, not the NASA official test split.",
            "Feature importance measures model reliance, not causal explanations.",
        ],
    }
    summary_path = model_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    paths.append(summary_path)
    return summary, paths
