"""Milestone 7 — ONE-TIME official NASA C-MAPSS FD001 test evaluation.

No model training/tuning or model selection uses the NASA test labels.
Every test engine contributes one prediction: its FINAL RECORDED observation.
The evaluated models are exactly the saved Milestone 5 optimized pipelines.
NASA test engine ID is a dataset-local key; its overlap with train engine IDs
is not leakage (train/test fleets are different trajectories).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import RUL_CAP, feature_columns
from src.model_training import ROOT, regression_metrics
from src.model_optimization import MODEL_FILENAMES, MODEL_SLUGS, postprocess_prediction

LABELS = ["engine_id", "cycle", "rul_true"]
KEYS = ["engine_id", "cycle"]
EXPECTED_FAMILIES = ("Random Forest", "XGBoost")


def _must_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prerequisite file: {path}. Follow the Milestone 7 README.")
    return pd.read_csv(path)


def _check_keys(df: pd.DataFrame, title: str) -> None:
    if not set(KEYS).issubset(df):
        raise ValueError(f"{title} missing engine_id/cycle")
    if df[KEYS].isna().any().any():
        raise ValueError(f"{title} contains missing engine_id/cycle")
    if df.duplicated(KEYS).any():
        raise ValueError(f"{title} has duplicate engine/cycle identifiers")
    for col in KEYS:
        vals = df[col].to_numpy(dtype=float)
        if not np.isfinite(vals).all() or (vals % 1 != 0).any() or (vals < 1).any():
            raise ValueError(f"{title} contains invalid {col} values")


def load_official_test_data(project_root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Require M3 last-cycle features and aligned, SEPARATELY exported NASA labels.

    Use the complete test feature CSV to verify that each selected row really
    is the final observation; never reconstruct a test target from an engine's
    last cycle. Test labels are NOT returned as feature columns.
    """
    root = Path(project_root)
    processed = root / "data" / "processed"
    meta_path = processed / "feature_manifest.json"
    if not meta_path.is_file():
        raise FileNotFoundError("Missing data/processed/feature_manifest.json; run run_feature_engineering.py")
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    cols = feature_columns()
    if manifest.get("milestone") != 3 or manifest.get("model_feature_columns") != cols:
        raise ValueError("Feature manifest does not match the current Milestone 3 feature schema")

    all_features = _must_read_csv(processed / "test_fd001_features.csv")
    final_features = _must_read_csv(processed / "test_fd001_last_features.csv")
    actual = _must_read_csv(root / "data" / "evaluation" / "test_fd001_holdout_labels.csv")
    for title, frame in (("complete test features", all_features),
                         ("last test features", final_features),
                         ("official test labels", actual)):
        _check_keys(frame, title)
    if not set(cols).issubset(all_features.columns) or not set(cols).issubset(final_features.columns):
        raise ValueError("Test feature files lack one or more expected causal model features")
    if not set(LABELS).issubset(actual):
        raise ValueError("Official test labels lack rul_true or engine/cycle keys")
    if actual["rul_true"].isna().any():
        raise ValueError("Missing official test RUL values")
    true_vals = actual["rul_true"].to_numpy(dtype=float)
    if not np.isfinite(true_vals).all() or (true_vals < 0).any() or (true_vals % 1 != 0).any():
        raise ValueError("Official test truth must be finite nonnegative integer cycles")
    if (all_features["engine_id"].nunique() != final_features["engine_id"].nunique()
            or final_features["engine_id"].nunique() != actual["engine_id"].nunique()):
        raise ValueError("Different numbers of engines across test feature and label files")
    if final_features.duplicated("engine_id").any() or actual.duplicated("engine_id").any():
        raise ValueError("Expected exactly one last observation and one RUL label per test engine")
    final_true = (all_features.sort_values(KEYS)
                  .groupby("engine_id", sort=True).tail(1).sort_values("engine_id")
                  .reset_index(drop=True))
    chosen = final_features.sort_values("engine_id").reset_index(drop=True)
    truth = actual.sort_values("engine_id").reset_index(drop=True)
    if not chosen[KEYS].equals(final_true[KEYS]):
        raise ValueError("Saved last-test rows do not match the final cycles in full test data")
    if not chosen[KEYS].equals(truth[KEYS]):
        raise ValueError("NASA ground-truth test labels are misaligned with final test cycles")
    arr_chosen = chosen[cols].to_numpy(dtype=float)
    arr_truth = final_true[cols].to_numpy(dtype=float)
    if not np.isfinite(arr_chosen).all() or not np.isfinite(arr_truth).all():
        raise ValueError("Nonfinite model feature in NASA official test data")
    if not np.allclose(arr_chosen, arr_truth, atol=1e-6, rtol=1e-8):
        raise ValueError("Saved last-test features differ from their complete histories")
    expected_count = manifest.get("test_engines")
    if expected_count is not None and len(chosen) != int(expected_count):
        raise ValueError("Unexpected number of official test engines vs feature manifest")
    return chosen, truth, manifest


def evaluate_and_export(project_root: Path = ROOT) -> tuple[dict, list[Path]]:
    """Score frozen M5 models on NASA FD001 one time; never refit or select on test."""
    root = Path(project_root)
    summary_file = root / "models" / "optimization_summary.json"
    if not summary_file.is_file():
        raise FileNotFoundError("Missing Milestone 5 summary. Run optimize_models.py first (before test evaluation).")
    m5 = json.loads(summary_file.read_text(encoding="utf-8"))
    if m5.get("milestone") != 5:
        raise ValueError("Expected Milestone 5 optimization_summary.json")
    target_name = m5.get("target")
    if target_name not in ("rul_capped_125", "rul_true"):
        raise ValueError(f"Unsupported model training target: {target_name}")
    selected_from_validation = m5.get("best_validation_rmse_model")
    if selected_from_validation not in EXPECTED_FAMILIES:
        raise ValueError("Missing preselected validation winner in Milestone 5 summary")
    if set(m5.get("model_files", {})) != set(EXPECTED_FAMILIES):
        raise ValueError("Milestone 5 model-file metadata is incomplete")
    if m5["model_files"] != MODEL_FILENAMES:
        raise ValueError("Milestone 5 model filenames do not match the current code")

    final_X, truth, manifest = load_official_test_data(root)
    cols = feature_columns()
    X = final_X[cols]
    actual_raw = truth["rul_true"].to_numpy(dtype=float)
    actual_cap = np.minimum(actual_raw, RUL_CAP)
    preds_frame = final_X[KEYS].copy()
    preds_frame["actual_rul_true"] = actual_raw.astype(int)
    preds_frame["actual_rul_capped_125"] = actual_cap.astype(int)

    scores = {}
    for family in EXPECTED_FAMILIES:
        path = root / "models" / m5["model_files"][family]
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved optimized model: {path}")
        # Only load locally generated/trusted joblib pipelines (pickle-based files).
        pipeline = joblib.load(path)
        if not hasattr(pipeline, "predict") or not hasattr(pipeline, "named_steps"):
            raise ValueError(f"Saved {family} model is not an expected fitted sklearn pipeline")
        if not {"variance", "regressor"}.issubset(pipeline.named_steps):
            raise ValueError(f"Saved {family} model is missing expected preprocessing/regressor steps")
        try:
            if int(pipeline.n_features_in_) != len(cols):
                raise ValueError(f"{family}: expected {len(cols)} feature inputs")
        except AttributeError:
            pass  # Older sklearn pipelines may not expose this property.
        raw_pred = np.asarray(pipeline.predict(X), dtype=float)
        if raw_pred.ndim != 1 or len(raw_pred) != len(final_X) or not np.isfinite(raw_pred).all():
            raise ValueError(f"{family} returned invalid test predictions")
        predicted = postprocess_prediction(raw_pred, "capped" if target_name == "rul_capped_125" else "true")
        preds_frame[f"pred_{MODEL_SLUGS[family]}"] = predicted
        # Report BOTH capped-target and physical-uncapped error, clearly separate.
        capped = regression_metrics(actual_cap, predicted)
        uncapped = regression_metrics(actual_raw, predicted)
        err = predicted - actual_raw
        scores[family] = {
            "capped_target": capped,
            "true_uncapped_rul": uncapped,
            "mean_signed_error_true": round(float(np.mean(err)), 4),
            "absolute_error_over_20_true": int((np.abs(err) > 20).sum()),
        }
        print(f"{family:14} | capped MAE {capped['mae']:.3f} RMSE {capped['rmse']:.3f} "
              f"| true MAE {uncapped['mae']:.3f} RMSE {uncapped['rmse']:.3f}", flush=True)

    # Important: do NOT choose a new model using these official test scores.
    metrics_rows = []
    for family, s in scores.items():
        for convention, met in (("capped_target", s["capped_target"]),
                                ("true_uncapped_rul", s["true_uncapped_rul"])):
            metrics_rows.append({"model": family, "evaluation_target": convention,
                                 "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"],
                                 "test_engines": len(final_X)})
    metric_table = pd.DataFrame(metrics_rows)
    # Convert to engine-level diagnostics; 1 row per test engine.
    for family in EXPECTED_FAMILIES:
        key = f"pred_{MODEL_SLUGS[family]}"
        preds_frame[f"abs_error_true_{MODEL_SLUGS[family]}"] = np.abs(preds_frame[key] - preds_frame["actual_rul_true"])
        preds_frame[f"signed_error_true_{MODEL_SLUGS[family]}"] = preds_frame[key] - preds_frame["actual_rul_true"]

    report = {
        "project": "AeroPredict AI — NASA C-MAPSS FD001",
        "milestone": 7,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": "Official FD001 final recorded cycle only; one prediction per test engine",
        "test_engine_count": int(len(final_X)),
        "model_training_target": target_name,
        "cap_cycles": RUL_CAP if target_name == "rul_capped_125" else None,
        "true_rul_exceeds_cap_count": int((actual_raw > RUL_CAP).sum()),
        "model_selected_before_official_test": selected_from_validation,
        "selection_rule": "Milestone 5: lowest RMSE on held-out training-engine validation; never choose from official test metrics",
        "milestone_5_validation_metrics": m5.get("validation_metrics", {}),
        "model_results": scores,
        "feature_count": len(cols),
        "feature_manifest_milestone": manifest.get("milestone"),
        "quick_training_mode": bool(m5.get("quick_mode", False)),
        "limitations": [
            "Test evaluation is one prediction at the last observed cycle of each NASA test engine (not every test cycle).",
            "Models were fit on the original Milestone 5 train portion, not refit on all 100 training engines.",
            "When trained on capped targets, capped-target errors are not equivalent to errors on true remaining physical life.",
            "These test scores are an evaluation snapshot; after inspecting them, the official test set is no longer untouched.",
            "Do not retune models, pick a winner, or repeatedly revise methodology based on official test performance.",
            "NASA C-MAPSS is simulated data. Predictions are not aircraft maintenance or safety instructions.",
            "Only load locally trusted joblib model files (pickle execution risk).",
        ],
    }
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "m7_official_test_predictions.csv",
             output_dir / "m7_official_test_metrics.csv",
             output_dir / "m7_official_test_summary.json"]
    preds_frame.to_csv(paths[0], index=False, float_format="%.5f")
    metric_table.to_csv(paths[1], index=False)
    paths[2].write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Preselected model (from M5 validation, NOT test): {selected_from_validation}")
    print(f"Official test engines evaluated: {len(final_X)}")
    print(f"True RUL values above {RUL_CAP}: {report['true_rul_exceeds_cap_count']}")
    return report, paths
