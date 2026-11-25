"""Milestone 6: reproducible, non-live engine-health *replay*.

The only histories available to this module are the 20 M5 validation engines
from the NASA *training* FD001 split; NASA's official test files and labels are
not opened. Saved optimized M5 pipelines perform fresh per-cycle inference.

All sensors/rolling features were engineered using current and prior readings.
All bands are illustrative rules based on *point estimates*, not probabilities
of failure or airworthiness assessments.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import feature_columns
from src.model_optimization import MODEL_FILENAMES, MODEL_SLUGS, postprocess_prediction

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["engine_id", "cycle"]
MONITORED_SENSORS = ("sensor_2", "sensor_4", "sensor_11", "sensor_15", "sensor_21")
REQUIRED_PRED_COLUMNS = {"engine_id", "cycle", "actual_rul", "pred_random_forest", "pred_xgboost"}


def health_band(predicted_rul: float, *, urgent_at: int = 30, watch_at: int = 65) -> dict[str, str]:
    """Human-readable *demonstration* bands; thresholds do not imply safety."""
    if not (0 <= urgent_at < watch_at <= 125):
        raise ValueError("Choose integer thresholds 0 <= urgent_at < watch_at <= 125")
    if not np.isfinite(predicted_rul):
        raise ValueError("Predicted RUL must be finite")
    if predicted_rul <= urgent_at:
        return {"label": "Priority review", "tone": "urgent", "note": "Illustrative high-urgency band; request a closer technical review."}
    if predicted_rul <= watch_at:
        return {"label": "Watch closely", "tone": "watch", "note": "Illustrative watch band; examine the recorded trends."}
    return {"label": "Routine observation", "tone": "routine", "note": "Illustrative low-urgency band, not a safety clearance."}


def validate_inputs(features: pd.DataFrame, prior_predictions: pd.DataFrame,
                    summary: dict) -> tuple[pd.DataFrame, list[str]]:
    """Enforce validation-only model replay, strict keys, and feature schema."""
    if summary.get("milestone") != 5:
        raise ValueError("Expected the Milestone 5 optimization_summary.json")
    if summary.get("target") not in ("rul_capped_125", "rul_true"):
        raise ValueError("Unknown target in optimization summary")
    split = summary.get("split", {})
    train_ids = set(map(int, split.get("training_engine_ids", [])))
    validation_ids = set(map(int, split.get("validation_engine_ids", [])))
    if not train_ids or not validation_ids or not train_ids.isdisjoint(validation_ids):
        raise ValueError("Invalid M5 engine split: train/validation engines must be disjoint")
    cols = feature_columns()
    required_features = set(KEYS + cols + list(MONITORED_SENSORS))
    if not required_features.issubset(features.columns):
        raise ValueError("M3 feature file is missing model or telemetry columns")
    if not REQUIRED_PRED_COLUMNS.issubset(prior_predictions.columns):
        raise ValueError("M5 validation predictions are missing required columns")
    if features.duplicated(KEYS).any() or prior_predictions.duplicated(KEYS).any():
        raise ValueError("Duplicate engine/cycle keys are not allowed")
    observed_ids = set(map(int, prior_predictions["engine_id"].unique()))
    if observed_ids != validation_ids:
        raise ValueError("Validation predictions do not match the M5 validation-engine list")
    if len(prior_predictions) == 0:
        raise ValueError("Empty validation predictions")
    joined = prior_predictions[KEYS].merge(features[["engine_id", *cols]], on=KEYS, how="left", validate="one_to_one", indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("Missing feature rows for some M5 validation observations")
    if not np.isfinite(joined[cols].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite monitoring feature values")
    if (joined["engine_id"].astype(int).isin(train_ids)).any():
        raise ValueError("Replay must use only validation engines, never fitted training engines")
    return joined.drop(columns="_merge").sort_values(KEYS).reset_index(drop=True), cols


def prepare_monitoring(project_root: Path = ROOT) -> tuple[dict, list[Path]]:
    """Verify M5 models reproduce M5 predictions, then export replay rows.

    Never reads data/raw/test_FD001.txt, RUL_FD001.txt or test holdout labels.
    """
    root = Path(project_root)
    summary_path = root / "models" / "optimization_summary.json"
    feature_path = root / "data" / "processed" / "train_fd001_features.csv"
    prediction_path = root / "outputs" / "m5_validation_predictions.csv"
    missing = [path for path in (summary_path, feature_path, prediction_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Milestone 3/5 files: " + ", ".join(str(p) for p in missing))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    features = pd.read_csv(feature_path)
    predictions = pd.read_csv(prediction_path)
    aligned, cols = validate_inputs(features, predictions, summary)
    # Reuse the exact validation observations from the saved M5 results.
    telemetry = features[KEYS + list(MONITORED_SENSORS)].merge(
        predictions[KEYS + ["actual_rul"]], on=KEYS, validate="one_to_one", how="inner"
    ).sort_values(KEYS).reset_index(drop=True)
    if len(telemetry) != len(predictions):
        raise ValueError("Telemetry rows do not cover every M5 validation observation")

    prediction_table = aligned[KEYS].copy()
    target_mode = "capped" if summary["target"] == "rul_capped_125" else "true"
    for family, slug in MODEL_SLUGS.items():
        expected_name = MODEL_FILENAMES[family]
        # Reject inconsistent or unexpected model filenames.
        if summary.get("model_files", {}).get(family) != expected_name:
            raise ValueError(f"Model filename mismatch for {family}. Rerun optimize_models.py")
        model_path = root / "models" / expected_name
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing optimized model {model_path}; rerun optimize_models.py")
        # Load ONLY artifacts produced locally by this project. joblib is pickle-based.
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            raise ValueError("Optimized model is not a fitted predictor")
        if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != cols:
            raise ValueError("Model feature list differs from Milestone 3 manifest")
        output = postprocess_prediction(model.predict(aligned[cols]), target_mode)
        if not np.isfinite(output).all():
            raise ValueError(f"Nonfinite predictions from {family}")
        prediction_table[f"pred_{slug}"] = output

    merged = telemetry.merge(prediction_table, on=KEYS, how="inner", validate="one_to_one")
    compare = merged.merge(predictions[KEYS + ["pred_random_forest", "pred_xgboost"]],
                           on=KEYS, validate="one_to_one", suffixes=("", "_m5"))
    for slug in MODEL_SLUGS.values():
        col = f"pred_{slug}"
        if not np.allclose(compare[col], compare[f"{col}_m5"], rtol=1e-4, atol=0.02):
            raise ValueError(f"Saved model and M5 {slug} predictions disagree; rerun optimize_models.py")
    if not np.isfinite(merged[list(MONITORED_SENSORS) + ["actual_rul", "pred_random_forest", "pred_xgboost"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in monitoring table")
    if (merged["cycle"] <= 0).any():
        raise ValueError("Engine operating cycles must be positive")

    result_dir = root / "outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    replay_path = result_dir / "m6_monitoring_replay.csv"
    meta_path = result_dir / "m6_monitoring_summary.json"
    merged.to_csv(replay_path, index=False)
    meta = {
        "project": "AeroPredict AI / NASA FD001",
        "milestone": 6,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Milestone 5 fixed engine-level validation holdout; NOT NASA official test data",
        "model_source": "Milestone 5 locally trained optimized models; precomputed inference for fast Streamlit replay",
        "target": summary["target"],
        "target_cap": 125 if target_mode == "capped" else None,
        "chosen_default_model": summary.get("best_validation_rmse_model", "XGBoost"),
        "quick_mode": bool(summary.get("quick_mode", False)),
        "validation_engines": sorted(map(int, merged["engine_id"].unique())),
        "row_count": int(len(merged)),
        "sensors": list(MONITORED_SENSORS),
        "guardrails": [
            "Only already-reviewed M5 validation histories are replayed; no independent test evidence is claimed.",
            "Sensor plots show observations through the selected cycle, not afterward.",
            "Actual validation RUL is available only behind an explicit reveal control.",
            "Illustrative urgency thresholds are not calibrated failure probabilities or maintenance instructions.",
            "This is offline historical replay, not real-time engine telemetry.",
            "NASA official test data and labels remain reserved for Milestone 7.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta, [replay_path, meta_path]
