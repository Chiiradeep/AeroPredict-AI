"""AeroPredict AI / M5: engine-wise CV, guarded validation, and SHAP explanations.

Only Milestone 3 training features and labels are loaded. NASA's official test
trajectories and evaluation labels are intentionally never accessed here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.model_training import (
    ROOT, SEED, TARGET_OPTIONS, engine_split, load_training_data, regression_metrics,
)


MODEL_SPECS = {
    "Random Forest": {
        "RF / baseline": dict(n_estimators=90, max_depth=16, min_samples_leaf=3,
                              max_features=0.8),
        "RF / smoother": dict(n_estimators=140, max_depth=12, min_samples_leaf=5,
                              max_features=0.7),
    },
    "XGBoost": {
        "XGB / baseline": dict(n_estimators=200, max_depth=5, learning_rate=0.065,
                               subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
                               min_child_weight=1),
        "XGB / conservative": dict(n_estimators=300, max_depth=3, learning_rate=0.055,
                                   subsample=0.85, colsample_bytree=0.9, reg_lambda=5.0,
                                   min_child_weight=4),
    },
}
MODEL_SLUGS = {"Random Forest": "random_forest", "XGBoost": "xgboost"}
MODEL_FILENAMES = {
    "Random Forest": "optimized_random_forest_fd001.joblib",
    "XGBoost": "optimized_xgboost_fd001.joblib",
}


def make_pipeline(family: str, params: dict, *, quick: bool = False, seed: int = SEED) -> Pipeline:
    """Variance filter learns only from the relevant training-engine fold."""
    args = dict(params)
    if quick:
        args["n_estimators"] = min(int(args["n_estimators"]), 24 if family == "Random Forest" else 45)
    if family == "Random Forest":
        regressor = RandomForestRegressor(**args, n_jobs=-1, random_state=seed)
    elif family == "XGBoost":
        regressor = XGBRegressor(**args, n_jobs=4, random_state=seed,
                                tree_method="hist", objective="reg:squarederror")
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([("variance", VarianceThreshold(threshold=0.0)),
                     ("regressor", regressor)])


def postprocess_prediction(values: np.ndarray, target: str) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return np.minimum(values, 125.0) if target == "capped" else values


def grouped_search(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, *,
                   target: str, quick: bool = False, n_splits: int = 3) -> tuple[dict, pd.DataFrame]:
    """Select hyperparameters using *only* training engines, never validation engines."""
    group_ids = np.unique(groups)
    if len(group_ids) < n_splits:
        raise ValueError(f"Need at least {n_splits} training engines for grouped CV")
    cv = GroupKFold(n_splits=n_splits)
    rows = []
    selected = {}
    for family, spec_map in MODEL_SPECS.items():
        for name, params in spec_map.items():
            fold_scores = []
            for fold, (fit_i, check_i) in enumerate(cv.split(X, y, groups), start=1):
                if not set(groups[fit_i]).isdisjoint(set(groups[check_i])):
                    raise AssertionError("An engine appears in train and validation CV folds")
                model = make_pipeline(family, params, quick=quick)
                model.fit(X.iloc[fit_i], y[fit_i])
                preds = postprocess_prediction(model.predict(X.iloc[check_i]), target)
                metrics = regression_metrics(y[check_i], preds)
                fold_scores.append(metrics["rmse"])
                rows.append({"model":family, "candidate":name, "fold":fold,
                             "fold_train_engines":len(np.unique(groups[fit_i])),
                             "fold_validation_engines":len(np.unique(groups[check_i])),
                             "rmse":metrics["rmse"], "mae":metrics["mae"],
                             "r2":metrics["r2"]})
                print(f"  {name}: fold {fold}/{n_splits}, RMSE={metrics['rmse']:.3f}", flush=True)
            mean_rmse = float(np.mean(fold_scores))
            if family not in selected or mean_rmse < selected[family]["cv_rmse"]:
                selected[family] = {"candidate":name, "params":params,
                                    "cv_rmse":round(mean_rmse, 4)}
    return selected, pd.DataFrame(rows)


def choose_shap_rows(validation_frame: pd.DataFrame, samples_per_engine: int = 4) -> np.ndarray:
    """Deterministic within-engine positions cover early/mid/late cycles."""
    indices = []
    for _, grp in validation_frame.groupby("engine_id", sort=True):
        positions = np.unique(np.linspace(0, len(grp) - 1,
                                          num=min(samples_per_engine, len(grp)), dtype=int))
        indices.extend(grp.iloc[positions].index.tolist())
    return np.asarray(indices, dtype=int)


def explain_pipeline(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Tree SHAP for the unclipped regressor; postprocessing is not explained."""
    try:
        import shap
    except ImportError as e:
        raise ImportError("SHAP is missing: install requirements.txt first") from e
    support = pipeline.named_steps["variance"].get_support()
    names = X.columns[support].tolist()
    selected_X = pipeline.named_steps["variance"].transform(X)
    regressor = pipeline.named_steps["regressor"]
    explainer = shap.TreeExplainer(regressor, feature_perturbation="tree_path_dependent")
    contribution = np.asarray(explainer.shap_values(selected_X), dtype=float)
    if contribution.ndim != 2 or contribution.shape != selected_X.shape:
        raise ValueError(f"Unexpected SHAP shape: {contribution.shape}")
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    bases = np.full(len(X), base, dtype=float)
    raw = np.asarray(regressor.predict(selected_X), dtype=float)
    if not np.allclose(bases + contribution.sum(axis=1), raw, atol=0.1, rtol=1e-3):
        raise ValueError("SHAP contributions do not reconstruct raw regression predictions")
    return contribution, bases, raw, names


def optimize_and_export(project_root: Path = ROOT, *, target: str = "capped",
                        quick: bool = False, shap_samples_per_engine: int = 4) -> tuple[dict, list[Path]]:
    """Fit candidates on inner grouped CV, evaluate selected models on M4 engine holdout."""
    root = Path(project_root)
    if target not in TARGET_OPTIONS:
        raise ValueError("Target must be 'capped' or 'true'")
    frames, labels, cols = load_training_data(root)
    tr_idx, val_idx = engine_split(frames, seed=SEED)
    training = frames.iloc[tr_idx].copy()
    validation = frames.iloc[val_idx].copy().reset_index(drop=True)
    X_train, X_val = training[cols], validation[cols]
    target_name = TARGET_OPTIONS[target]
    y_train = labels.iloc[tr_idx][target_name].to_numpy(dtype=float)
    y_val = labels.iloc[val_idx][target_name].to_numpy(dtype=float)
    groups = training["engine_id"].to_numpy()

    print(f"Nested model selection: {training.engine_id.nunique()} training engines; "
          f"evaluation on {validation.engine_id.nunique()} held-out validation engines")
    print("Official NASA test files are never opened.")
    winners, cv_table = grouped_search(X_train, y_train, groups, target=target, quick=quick)
    out_dir = root / "outputs"
    model_dir = root / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    pred_frame = validation[["engine_id", "cycle"]].copy()
    pred_frame["actual_rul"] = y_val
    val_metrics = {}
    shap_global_rows = []
    shap_local_rows = []
    artifact_paths = []
    shap_pos = choose_shap_rows(validation, samples_per_engine=shap_samples_per_engine)

    for family in MODEL_SPECS:
        chosen = winners[family]
        print(f"Fitting selected {family} on all training engines ({chosen['candidate']})...", flush=True)
        pipeline = make_pipeline(family, chosen["params"], quick=quick)
        pipeline.fit(X_train, y_train)
        values = postprocess_prediction(pipeline.predict(X_val), target)
        pred_frame[f"pred_{MODEL_SLUGS[family]}"] = values
        val_metrics[family] = regression_metrics(y_val, values)
        model_path = model_dir / MODEL_FILENAMES[family]
        joblib.dump(pipeline, model_path, compress=3)
        artifact_paths.append(model_path)

        explain_X = X_val.iloc[shap_pos]
        contributions, bases, raw_preds, active_features = explain_pipeline(pipeline, explain_X)
        importance = np.mean(np.abs(contributions), axis=0)
        for name, imp in zip(active_features, importance):
            shap_global_rows.append({"model":family, "feature":name,
                                     "mean_absolute_shap":float(imp)})
        # Precompute a small set of sample explanations for an efficient hosted UI.
        for i, original_pos in enumerate(shap_pos):
            eng_id = int(validation.iloc[original_pos]["engine_id"])
            cycle = int(validation.iloc[original_pos]["cycle"])
            row_values = explain_X.iloc[i]
            strongest = np.argsort(np.abs(contributions[i]))[::-1][:10]
            for idx in strongest:
                name = active_features[int(idx)]
                shap_local_rows.append({
                    "model":family, "engine_id":eng_id, "cycle":cycle,
                    "feature":name, "feature_value":float(row_values[name]),
                    "shap_value":float(contributions[i,int(idx)]),
                    "base_value":float(bases[i]),
                    "raw_prediction":float(raw_preds[i]),
                    "display_prediction":float(postprocess_prediction([raw_preds[i]], target)[0]),
                    "actual_rul":float(y_val[original_pos]),
                })
        print(f"  Holdout MAE={val_metrics[family]['mae']:.3f} "
              f"RMSE={val_metrics[family]['rmse']:.3f}", flush=True)

    engine_rows = []
    for family in MODEL_SPECS:
        pred_col = f"pred_{MODEL_SLUGS[family]}"
        for eng, engine_group in pred_frame.groupby("engine_id"):
            engine_group = engine_group.sort_values("cycle")
            errors = engine_group[pred_col].to_numpy()-engine_group["actual_rul"].to_numpy()
            engine_rows.append({"model":family, "engine_id":int(eng),
                                "observations":int(len(engine_group)),
                                "engine_mae":float(np.mean(np.abs(errors))),
                                "engine_rmse":float(np.sqrt(np.mean(errors**2))),
                                "end_cycle_abs_error":float(abs(errors[-1]))})
    output_files = {
        "cv":out_dir/"m5_group_cv_results.csv",
        "predictions":out_dir/"m5_validation_predictions.csv",
        "engine_errors":out_dir/"m5_engine_errors.csv",
        "shap_global":out_dir/"m5_shap_global.csv",
        "shap_local":out_dir/"m5_shap_local.csv",
    }
    cv_table.to_csv(output_files["cv"], index=False)
    pred_frame.sort_values(["engine_id", "cycle"]).to_csv(output_files["predictions"], index=False)
    pd.DataFrame(engine_rows).sort_values(["model","engine_id"]).to_csv(output_files["engine_errors"],index=False)
    pd.DataFrame(shap_global_rows).sort_values(["model","mean_absolute_shap"],ascending=[True,False]).to_csv(output_files["shap_global"],index=False)
    pd.DataFrame(shap_local_rows).to_csv(output_files["shap_local"],index=False)
    artifact_paths.extend(output_files.values())

    summary = {
        "project":"AeroPredict AI / NASA C-MAPSS FD001",
        "milestone":5,
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "target":target_name,
        "quick_mode":bool(quick),
        "candidate_feature_count":len(cols),
        "model_files":MODEL_FILENAMES,
        "selected_cv_candidates":winners,
        "validation_metrics":val_metrics,
        "best_validation_rmse_model":min(val_metrics, key=lambda m:val_metrics[m]["rmse"]),
        "split":{
            "seed":SEED,
            "training_engine_ids":sorted(map(int,training.engine_id.unique())),
            "validation_engine_ids":sorted(map(int,validation.engine_id.unique())),
            "training_engines":int(training.engine_id.nunique()),
            "validation_engines":int(validation.engine_id.nunique()),
            "inner_folds":3,
            "strategy":"Inner GroupKFold by engine_id on M4 training engines; fixed M4 held-out engine split",
        },
        "explanations":{
            "sampled_validation_rows":int(len(shap_pos)),
            "samples_per_engine":int(shap_samples_per_engine),
            "method":"Tree SHAP, tree_path_dependent, feature values transformed using train-only variance mask",
            "note":"SHAP explains the unclipped raw tree regressor output; displayed RUL predictions are post-processed to [0,125] (or nonnegative for uncapped target). Positive SHAP raises predicted RUL; negative SHAP lowers it. Attributions are associations, not causes. Local file saves the 10 largest contributions per sample; omitted contributions can remain.",
        },
        "limitations":[
            "The engine holdout was already viewed during M4; use M5 holdout as a descriptive comparison, not as untouched final test evidence.",
            "The official NASA test set has not been accessed or used for tuning, training, or explanations.",
            "Validation metrics are per row; engine_error_summary provides per-engine diagnostics.",
            "Small 3-fold inner CV over two candidates per model family is illustrative, not an exhaustive hyperparameter search.",
            "Do not load untrusted joblib files; they use Python pickle.",
        ],
    }
    summary_path = model_dir/"optimization_summary.json"
    summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    artifact_paths.append(summary_path)
    return summary, artifact_paths
