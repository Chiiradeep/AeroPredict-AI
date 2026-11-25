"""Run AeroPredict AI Milestone 2 data quality checks and preprocessing."""

from src.preprocessing import process_and_export


def main():
    print("\nAEROPREDICT AI  /  MILESTONE 02  /  DATA LABORATORY")
    print("-" * 59)
    try:
        report, outputs = process_and_export()
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"FAILED: {exc}")
        print("Check data/raw/ and run download_dataset.py if the files are missing.")
        raise SystemExit(1) from exc

    for label in ("train", "test"):
        info = report[label]
        print(
            f"{label.upper():5}: {info['rows']:,} records, "
            f"{info['engines']} engines, {info['missing_cells']} missing cells, "
            f"{info['duplicate_engine_cycles']} duplicate engine-cycle keys — {info['status']}"
        )
    print("Constant training features (flagged, not dropped): "
          + (", ".join(report["train"]["constant_features"]) or "none"))
    print("\nSaved files:")
    for path in outputs:
        print(f"  - {path}")
    print("\nSUCCESS — Milestone 2 quality checks and preprocessing completed.")
    print("No target labels, scaling, or future information were added to feature CSVs.")


if __name__ == "__main__":
    main()
