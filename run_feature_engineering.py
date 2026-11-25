"""AeroPredict AI — Milestone 3 offline feature engineering runner."""

from src.feature_engineering import build_and_export


def main() -> None:
    print("\nAEROPREDICT AI / MILESTONE 03 / RUL & FEATURE ENGINEERING")
    print("-" * 61)
    try:
        manifest, paths = build_and_export()
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"ERROR: {error}")
        print("If necessary, run run_preprocessing.py first; check the NASA FD001 files.")
        raise SystemExit(1) from error

    print(f"Train: {manifest['train_rows']:,} rows / {manifest['train_engines']} engines")
    print(f"Test: {manifest['test_rows']:,} rows / {manifest['test_engines']} engines")
    print(f"Causal numeric model features: {manifest['model_feature_count']}")
    print("\nSaved:")
    for path in paths:
        print(f"  {path}")
    print("\nSUCCESS — Milestone 3 feature engineering completed.")
    print("No model was trained. Test truth stays separate from test features.")


if __name__ == "__main__":
    main()
