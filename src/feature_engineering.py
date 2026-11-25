"""Milestone 3 — leakage-aware RUL labels and causal sensor-history features.

NASA FD001: training trajectories finish at failure, while RUL_FD001.txt
contains the remaining cycles at the *last observed* test cycle only.

Never use future failure cycles / labels / EDA-only life fractions as model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import clean_dataset

ROOT = Path(__file__).resolve().parents[1]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
WINDOWS = (5, 20)
RUL_CAP = 125


def feature_columns() -> list[str]:
    """Authoritative model-feature list. Engine ID and RUL labels are excluded."""
    cols = ["cycle", *SETTINGS, *SENSORS]
    for sensor in SENSORS:
        for window in WINDOWS:
            cols += [f"{sensor}_mean_{window}", f"{sensor}_std_{window}"]
        cols.append(f"{sensor}_delta_5")
    return cols


def add_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Generate features using CURRENT and PAST observations only.

    Rolling windows reset at engine boundaries. No scaler is fit here.
    The first five cycles get delta_5 = 0 because a five-cycle lag is
    unavailable. This is intentional, documented feature behavior.
    """
    ordered = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True).copy()
    if ordered.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle pair in feature input")
    if ordered[SENSORS + SETTINGS].isna().any().any():
        raise ValueError("Missing sensor or operating-setting value in feature input")

    grouped = ordered.groupby("engine_id", sort=False)
    engineered: dict[str, pd.Series] = {}
    for sensor in SENSORS:
        sensor_groups = grouped[sensor]
        for window in WINDOWS:
            engineered[f"{sensor}_mean_{window}"] = sensor_groups.transform(
                lambda series: series.rolling(window, min_periods=1).mean()
            )
            engineered[f"{sensor}_std_{window}"] = sensor_groups.transform(
                lambda series: series.rolling(window, min_periods=1).std(ddof=0)
            )
        engineered[f"{sensor}_delta_5"] = (
            ordered[sensor] - sensor_groups.shift(5)
        ).fillna(0.0)

    feature_frame = pd.concat([ordered, pd.DataFrame(engineered, index=ordered.index)], axis=1)
    return feature_frame[["engine_id", *feature_columns()]]


def training_rul_labels(train: pd.DataFrame) -> pd.DataFrame:
    """Training supervision only, not inference features."""
    ordered = train.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    final_cycle = ordered.groupby("engine_id")["cycle"].transform("max")
    true_rul = (final_cycle - ordered["cycle"]).astype("int64")
    if (true_rul < 0).any():
        raise ValueError("Negative training RUL detected")
    return pd.DataFrame({
        "engine_id": ordered["engine_id"].astype("int64"),
        "cycle": ordered["cycle"].astype("int64"),
        "rul_true": true_rul,
        "rul_capped_125": true_rul.clip(upper=RUL_CAP),
    })


def read_test_holdout_labels(raw_path: Path, last_test_rows: pd.DataFrame) -> pd.DataFrame:
    """Align NASA's one-RUL-per-test-engine file to final observed cycles.

    FD001 RUL lines correspond to test engine IDs in ascending numeric order.
    Store these ground-truth labels SEPARATELY from model-input features.
    """
    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing official test labels: {raw_path}")
    truth = pd.read_csv(raw_path, sep=r"\s+", header=None)
    if truth.shape[1] != 1 or truth.isna().any().any():
        raise ValueError("NASA RUL file must have exactly one numeric value per engine")
    if not np.isfinite(truth.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite NASA test RUL")
    ordered = last_test_rows.sort_values("engine_id").reset_index(drop=True)
    engine_ids = ordered["engine_id"].astype(int).tolist()
    if engine_ids != list(range(1, len(engine_ids) + 1)):
        raise ValueError("Expected contiguous 1-based NASA FD001 test engine IDs")
    if len(truth) != len(engine_ids):
        raise ValueError("Test RUL count does not match number of test engines")
    vals = truth.iloc[:, 0]
    if ((vals < 0) | (vals % 1 != 0)).any():
        raise ValueError("Test RUL labels must be nonnegative whole cycles")
    return pd.DataFrame({
        "engine_id": ordered["engine_id"].astype("int64"),
        "cycle": ordered["cycle"].astype("int64"),
        "rul_true": vals.astype("int64").to_numpy(),
    })


def build_and_export(project_root: Path = ROOT) -> tuple[dict, list[Path]]:
    """Create ML-ready features and labels from Milestone 2 cleaned files."""
    project_root = Path(project_root)
    proc_dir = project_root / "data" / "processed"
    raw_dir = project_root / "data" / "raw"
    train_file = proc_dir / "train_fd001_clean.csv"
    test_file = proc_dir / "test_fd001_clean.csv"
    for path in (train_file, test_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path.name} missing — run run_preprocessing.py first"
            )

    train = clean_dataset(pd.read_csv(train_file), label="FD001 train for M3")
    test = clean_dataset(pd.read_csv(test_file), label="FD001 test for M3")
    train_features = add_causal_features(train)
    test_features = add_causal_features(test)
    train_labels = training_rul_labels(train)
    last_test = test_features.groupby("engine_id", sort=False).tail(1).sort_values("engine_id").reset_index(drop=True)
    test_labels = read_test_holdout_labels(raw_dir / "RUL_FD001.txt", last_test)

    if not train_features[["engine_id", "cycle"]].equals(train_labels[["engine_id", "cycle"]]):
        raise ValueError("Train feature/label row alignment failed")
    if not last_test[["engine_id", "cycle"]].equals(test_labels[["engine_id", "cycle"]]):
        raise ValueError("Test holdout alignment failed")
    for df in (train_features, test_features):
        if not np.isfinite(df[feature_columns()].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite engineered model feature detected")

    eval_dir = project_root / "data" / "evaluation"
    proc_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        proc_dir / "train_fd001_features.csv",
        proc_dir / "train_fd001_labels.csv",
        proc_dir / "test_fd001_features.csv",
        proc_dir / "test_fd001_last_features.csv",
        eval_dir / "test_fd001_holdout_labels.csv",
        proc_dir / "feature_manifest.json",
    ]
    train_features.to_csv(paths[0], index=False)
    train_labels.to_csv(paths[1], index=False)
    test_features.to_csv(paths[2], index=False)
    last_test.to_csv(paths[3], index=False)
    test_labels.to_csv(paths[4], index=False)

    metadata = {
        "project": "AeroPredict AI — FD001",
        "milestone": 3,
        "train_rows": int(len(train_features)),
        "train_engines": int(train_features["engine_id"].nunique()),
        "test_rows": int(len(test_features)),
        "test_engines": int(test_features["engine_id"].nunique()),
        "test_last_rows": int(len(last_test)),
        "model_feature_count": len(feature_columns()),
        "model_feature_columns": feature_columns(),
        "non_model_identifier_columns": ["engine_id"],
        "train_target_columns": ["rul_true", "rul_capped_125"],
        "rul_cap_for_optional_modeling": RUL_CAP,
        "rolling_windows": list(WINDOWS),
        "notes": [
            "Each engineered feature uses only current/past readings of the same engine.",
            "Use engine-wise train/validation splits; never random split by rows.",
            "Fit scalers only on training engines of each split, not full data.",
            "Engine ID is an identifier, not a model input.",
            "Never use failure-cycle, RUL, life_fraction, life_stage or ground-truth test labels as features.",
            "Test truth is stored separately for final held-out evaluation.",
            "No model trained or predictions generated in Milestone 3.",
        ],
    }
    paths[5].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata, paths
