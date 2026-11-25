"""NASA C-MAPSS FD001 data audit and leak-safe Milestone 2 preparation.

Milestone 2 does not fit scalers, impute observations, or create RUL labels.
It verifies the supplied raw data and preserves every physical sensor column.
Any future normalizer must be fit on training engines only, after splitting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import COLUMNS, PROJECT_ROOT, load_fd001

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
NUMERIC_FEATURES = SETTINGS + SENSORS
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def audit_dataset(frame: pd.DataFrame, label: str) -> dict:
    """Return transparent data-quality statistics, without modifying data."""
    absent = [c for c in COLUMNS if c not in frame.columns]
    if absent:
        raise ValueError(f"{label}: missing expected columns: {absent}")

    relevant = frame[COLUMNS]
    null_counts = relevant.isna().sum()
    # All NASA telemetry columns should be numeric, including IDs and cycles.
    numeric = relevant.apply(pd.to_numeric, errors="coerce")
    invalid_numeric = int((numeric.isna() & relevant.notna()).sum().sum())
    finite_bad = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    duplicated_observations = int(relevant.duplicated().sum())
    duplicated_engine_cycles = int(relevant.duplicated(["engine_id", "cycle"]).sum())

    engine_ids = numeric["engine_id"].dropna()
    cycles = numeric["cycle"].dropna()
    invalid_identifiers = int(
        (engine_ids.lt(1) | (engine_ids.mod(1) != 0)).sum()
        + (cycles.lt(1) | (cycles.mod(1) != 0)).sum()
    )

    sorted_pairs = numeric[["engine_id", "cycle"]].dropna().sort_values(["engine_id", "cycle"])
    per_engine = sorted_pairs.groupby("engine_id")["cycle"].agg(["min", "max", "count"])
    gaps = int(((per_engine["max"] - per_engine["min"] + 1) != per_engine["count"]).sum())
    starts_after_one = int((per_engine["min"] != 1).sum())

    unique_values = numeric[NUMERIC_FEATURES].nunique(dropna=True)
    constants = [str(c) for c in unique_values.index if int(unique_values[c]) <= 1]
    missing_by_column = {str(col): int(v) for col, v in null_counts.items() if int(v) > 0}

    issues = []
    if int(null_counts.sum()) > 0:
        issues.append("missing values")
    if invalid_numeric > 0 or finite_bad > 0:
        issues.append("non-numeric or non-finite entries")
    if duplicated_observations > 0:
        issues.append("duplicate rows")
    if duplicated_engine_cycles > 0:
        issues.append("duplicate engine-cycle keys")
    if invalid_identifiers > 0:
        issues.append("invalid engine IDs or cycles")
    if gaps > 0 or starts_after_one > 0:
        issues.append("nonconsecutive or non-1-based engine cycles")

    return {
        "dataset": label,
        "rows": int(len(frame)),
        "engines": int(frame["engine_id"].nunique()),
        "columns": len(COLUMNS),
        "missing_cells": int(null_counts.sum()),
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicated_observations,
        "duplicate_engine_cycles": duplicated_engine_cycles,
        "invalid_numeric_cells": invalid_numeric,
        "nonfinite_cells": finite_bad,
        "invalid_engine_or_cycle_ids": invalid_identifiers,
        "engines_with_cycle_gaps": gaps,
        "engines_starting_after_cycle_1": starts_after_one,
        "constant_features": constants,
        "nonconstant_features": [str(c) for c in unique_values.index if int(unique_values[c]) > 1],
        "issues": issues,
        "status": "PASS" if not issues else "REVIEW",
    }


def clean_dataset(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Validate and deterministically order FD001 data, preserving all columns.

    We intentionally *do not* silently delete rows, fill values, scale sensors,
    or use knowledge of the engine's future life. That comes later, with proper
    train/validation separation. NASA FD001 should already be complete.
    """
    audit = audit_dataset(frame, label)
    if audit["issues"]:
        raise ValueError(f"{label} failed quality checks: {', '.join(audit['issues'])}")

    clean = frame.loc[:, COLUMNS].sort_values(["engine_id", "cycle"]).reset_index(drop=True).copy()
    clean["engine_id"] = clean["engine_id"].astype("int64")
    clean["cycle"] = clean["cycle"].astype("int64")
    for column in NUMERIC_FEATURES:
        clean[column] = clean[column].astype("float64")
    return clean


def training_life_stage(train: pd.DataFrame) -> pd.DataFrame:
    """Add EDA-only relative life stages based on *known* training end cycles.

    WARNING: `life_fraction` / `life_stage` use each engine's eventual failure
    cycle, and MUST NOT be used as model inputs or for inference on test engines.
    """
    max_cycles = train.groupby("engine_id")["cycle"].transform("max")
    result = train.copy()
    result["life_fraction"] = result["cycle"] / max_cycles
    result["life_stage"] = pd.cut(
        result["life_fraction"],
        bins=[0, 0.25, 0.75, 1.0],
        labels=["Early (0–25%)", "Middle (25–75%)", "Late (75–100%)"],
        include_lowest=True,
    )
    return result


def build_report(train: pd.DataFrame, test: pd.DataFrame, test_rul: pd.DataFrame) -> dict:
    train_audit = audit_dataset(train, "FD001 train")
    test_audit = audit_dataset(test, "FD001 test")
    return {
        "project": "AeroPredict AI",
        "milestone": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NASA C-MAPSS FD001",
        "train": train_audit,
        "test": test_audit,
        "test_rul_labels": int(len(test_rul)),
        "recommended_numeric_features": train_audit["nonconstant_features"],
        "excluded_constant_features_for_future_modeling": train_audit["constant_features"],
        "notes": [
            "Clean CSVs preserve the 26 original named columns and all sensor readings.",
            "Constant features are flagged, not deleted from exported datasets.",
            "No RUL targets or future-life-stage columns are added to the exported features.",
            "Do not fit a scaler on full training plus test; split engines first in Milestone 4.",
            "RUL_FD001.txt is held back for later test-set evaluation.",
        ],
    }


def process_and_export(output_dir: Path = PROCESSED_DIR) -> tuple[dict, list[Path]]:
    """Load raw data and write validated, sorted CSVs + JSON report."""
    train, test, test_rul = load_fd001()
    train_clean = clean_dataset(train, label="FD001 train")
    test_clean = clean_dataset(test, label="FD001 test")
    report = build_report(train_clean, test_clean, test_rul)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_fd001_clean.csv"
    test_path = output_dir / "test_fd001_clean.csv"
    report_path = output_dir / "data_quality_report.json"
    train_clean.to_csv(train_path, index=False, float_format="%.10g")
    test_clean.to_csv(test_path, index=False, float_format="%.10g")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report, [train_path, test_path, report_path]
