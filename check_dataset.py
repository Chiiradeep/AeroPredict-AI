"""Milestone 1: verify that NASA FD001 files load successfully."""

from src.data_loader import load_fd001


def main():
    try:
        train, test, rul = load_fd001()
    except (FileNotFoundError, ValueError) as exc:
        print(f"DATA CHECK FAILED: {exc}")
        raise SystemExit(1) from exc

    print("\n" + "=" * 60)
    print("AEROPREDICT AI - NASA C-MAPSS FD001 DATA CHECK")
    print("=" * 60)
    print(f"Train shape:            {train.shape[0]:,} rows x {train.shape[1]} columns")
    print(f"Test shape:             {test.shape[0]:,} rows x {test.shape[1]} columns")
    print(f"Training engines:       {train['engine_id'].nunique()}")
    print(f"Testing engines:        {test['engine_id'].nunique()}")
    print(f"Test RUL labels:        {len(rul)}")
    print(f"Missing train values:   {int(train.isna().sum().sum())}")
    print(f"Missing test values:    {int(test.isna().sum().sum())}")
    print("\nFirst 5 rows (selected columns):")
    print(train[["engine_id", "cycle", "setting_1", "sensor_2", "sensor_3"]].head().to_string(index=False))
    print("\nTraining engine lifetimes (cycles):")
    lengths = train.groupby("engine_id")["cycle"].max()
    print(lengths.describe().round(1).to_string())
    print("\nSUCCESS - Milestone 1 data setup is working.")
    print("Next: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
