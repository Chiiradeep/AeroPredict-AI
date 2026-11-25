"""AeroPredict AI — M7 official NASA FD001 final holdout evaluation.
Run once after M5 optimization; DOES NOT train models or change model files.
"""

from pathlib import Path
from src.final_evaluation import evaluate_and_export


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    try:
        report, files = evaluate_and_export(root)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"MILESTONE 7 SETUP ERROR: {exc}") from exc
    print("\nGenerated reports:")
    for path in files:
        print(f"  {path.relative_to(root)}")
    print("\nSUCCESS — Milestone 7 official NASA test evaluation completed.")
    print("NOTE: Scores are now visible; do not tune or select models on this test set.")
