"""Milestone 5 — tune FD001 RUL models with grouped CV; export SHAP explanations.

Run:
    python optimize_models.py
    python optimize_models.py --quick     # reduced trees for setup troubleshooting
"""
import argparse
from src.model_optimization import optimize_and_export


def main():
    parser = argparse.ArgumentParser(description="AeroPredict AI / model optimization")
    parser.add_argument("--target", choices=["capped", "true"], default="capped")
    parser.add_argument("--quick", action="store_true", help="Small trees for a smoke test; rerun without this for regular results")
    args = parser.parse_args()
    print("\nAEROPREDICT AI / MILESTONE 05 / OPTIMIZATION & SHAP")
    print("="*62)
    try:
        summary, paths = optimize_and_export(target=args.target, quick=args.quick)
    except (FileNotFoundError, ValueError, ImportError, OSError) as error:
        print("\nERROR:", error)
        print("Check that you ran run_feature_engineering.py and installed requirements.txt")
        raise SystemExit(1) from error
    print("\nThree-fold grouped cross-validation winners:")
    for name, row in summary["selected_cv_candidates"].items():
        print(f"  {name}: {row['candidate']} / mean CV RMSE={row['cv_rmse']:.2f}")
    print("\nHeld-out M4 engine validation (previously seen in M4; NOT official NASA test):")
    for name, vals in summary["validation_metrics"].items():
        print(f"  {name}: MAE={vals['mae']:.2f}, RMSE={vals['rmse']:.2f}, R2={vals['r2']:.3f}")
    print("\nExported:")
    for p in paths:
        print(f"  {p}")
    print("\nSUCCESS — Milestone 5 model optimization and SHAP completed.")
    print("Official NASA test labels were NOT used.")


if __name__ == "__main__":
    main()
