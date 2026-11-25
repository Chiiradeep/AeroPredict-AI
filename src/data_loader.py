"""Load the official NASA C-MAPSS FD001 data without changing source values."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)
EXPECTED_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")


def missing_files(data_dir: Path = RAW_DATA_DIR) -> list[str]:
    """Return the required filename(s) that are not present."""
    return [name for name in EXPECTED_FILES if not (data_dir / name).is_file()]


def _read_trajectory(path: Path) -> pd.DataFrame:
    # NASA's text format has no header and uses variable-width whitespace.
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns; found {frame.shape[1]}. "
            "Check that you downloaded the original FD001 text file."
        )
    frame.columns = COLUMNS
    if frame.isna().any().any():
        raise ValueError(f"{path.name} contains missing readings.")
    frame["engine_id"] = frame["engine_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame


def load_fd001(data_dir: Path = RAW_DATA_DIR):
    """Return (train, test, test_rul) with named columns.

    `test_rul` is the true RUL *after the final observed test cycle* and
    is ordered by engine ID; it is not a row-level label for `test`.
    """
    absent = missing_files(data_dir)
    if absent:
        raise FileNotFoundError(
            "Missing NASA files in data/raw/: " + ", ".join(absent)
            + ". Run: python download_dataset.py"
        )

    train = _read_trajectory(data_dir / "train_FD001.txt")
    test = _read_trajectory(data_dir / "test_FD001.txt")
    rul = pd.read_csv(data_dir / "RUL_FD001.txt", sep=r"\s+", header=None)
    if rul.shape[1] != 1 or rul.isna().any().any():
        raise ValueError("RUL_FD001.txt must contain one numeric RUL per engine.")
    rul.columns = ["remaining_cycles"]
    rul["remaining_cycles"] = rul["remaining_cycles"].astype(int)
    if len(rul) != test["engine_id"].nunique():
        raise ValueError("Test RUL count does not match the number of test engines.")
    return train, test, rul
