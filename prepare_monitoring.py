"""Create inference-backed validation replay for the M6 Streamlit dashboard."""
from __future__ import annotations

from src.engine_monitoring import prepare_monitoring


def main() -> None:
    try:
        meta, paths = prepare_monitoring()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        print("Try: python run_feature_engineering.py, then python optimize_models.py")
        raise SystemExit(1) from exc
    print("\nAeroPredict AI / Milestone 6 — offline engine monitoring replay")
    print(f"Validation engines: {len(meta['validation_engines'])}")
    print(f"Engine-cycle predictions: {meta['row_count']:,}")
    print(f"Target: {meta['target']}")
    if meta["quick_mode"]:
        print("WARNING: M5 models were created with --quick. Standard training is recommended.")
    for path in paths:
        print(f"Saved: {path}")
    print("SUCCESS — Milestone 6 monitoring replay prepared.")


if __name__ == "__main__":
    main()
