"""AeroPredict AI — Milestone 4 training entry point.

Usage:
  python train_models.py                  # Default: cap RUL at 125
  python train_models.py --target true    # Train on uncapped RUL
  python train_models.py --quick          # Small model sizes for quick troubleshooting
"""

import argparse
from src.model_training import train_and_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AeroPredict AI FD001 models")
    parser.add_argument("--target", choices=["capped", "true"], default="capped")
    parser.add_argument("--quick", action="store_true", help="Smaller models, for setup verification")
    args = parser.parse_args()
    print("\nAEROPREDICT AI / MILESTONE 04 / MODEL TRAINING")
    print("-" * 58)
    try:
        summary, files = train_and_export(target=args.target, quick=args.quick)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nERROR: {error}")
        print("First run run_preprocessing.py and run_feature_engineering.py if needed.")
        raise SystemExit(1) from error
    print("\nComparison (held-out training engines; not official test):")
    for name, scores in summary["validation_metrics"].items():
        print(f"  {name}: MAE={scores['mae']:.2f}, RMSE={scores['rmse']:.2f}, R2={scores['r2']:.3f}")
    print(f"Best validation RMSE: {summary['best_validation_model']}")
    print("\nSaved files:")
    for file in files:
        print(f"  {file}")
    print("\nSUCCESS — Milestone 4 model training completed.")
    print("Official NASA test labels were NOT accessed or used to select the model.")


if __name__ == "__main__":
    main()
