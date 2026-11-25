"""Download and extract only the NASA C-MAPSS FD001 text files.

Source: NASA Prognostics Center of Excellence repository, dataset #6.
The NASA download is normally an outer ZIP containing CMAPSSData.zip.
"""

from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

from src.data_loader import EXPECTED_FILES, RAW_DATA_DIR

NASA_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)


def find_member(archive: ZipFile, filename: str) -> str:
    """Find a single archive member by basename, without extracting paths."""
    matches = [
        name for name in archive.namelist()
        if not name.endswith("/") and PurePosixPath(name.replace("\\", "/")).name == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one '{filename}' inside ZIP; found {len(matches)}.")
    return matches[0]


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if all((RAW_DATA_DIR / name).exists() for name in EXPECTED_FILES):
        print("FD001 files are already available in data/raw/. No download needed.")
        return

    print("Downloading NASA C-MAPSS dataset (~12 MB)...")
    print(f"Official source: {NASA_URL}")
    try:
        request = Request(NASA_URL, headers={"User-Agent": "AeroPredictAI-Educational/1.0"})
        with urlopen(request, timeout=90) as response:
            data = response.read()
        with ZipFile(BytesIO(data)) as outer:
            # Support either the nested archive NASA currently provides or a flat ZIP.
            inner = [n for n in outer.namelist() if PurePosixPath(n).name == "CMAPSSData.zip"]
            if inner:
                if len(inner) != 1:
                    raise ValueError("Found multiple CMAPSSData.zip archive members.")
                nested_data = outer.read(inner[0])
                with ZipFile(BytesIO(nested_data)) as inner_zip:
                    saved = save_fd001(inner_zip)
            else:
                saved = save_fd001(outer)
    except (OSError, ValueError, BadZipFile) as exc:
        print(f"Dataset download/extraction failed: {exc}")
        print("Manual alternative: visit NASA's dataset page (link in README),")
        print("extract the ZIP(s), and put train_FD001.txt, test_FD001.txt")
        print("and RUL_FD001.txt into data/raw/. Then run check_dataset.py.")
        raise SystemExit(1) from exc

    print("\nDataset ready. Extracted files:")
    for name in saved:
        print(f"  data/raw/{name}")
    print("Next command: python check_dataset.py")


def save_fd001(archive: ZipFile) -> list[str]:
    """Extract only three expected data files, ignoring archive paths."""
    loaded = {}
    for filename in EXPECTED_FILES:
        member = find_member(archive, filename)
        loaded[filename] = archive.read(member)
        if not loaded[filename]:
            raise ValueError(f"{filename} is empty.")
    for filename, content in loaded.items():
        destination = RAW_DATA_DIR / filename
        temp_file = destination.with_name(destination.name + ".tmp")
        temp_file.write_bytes(content)
        temp_file.replace(destination)
    return list(loaded)


if __name__ == "__main__":
    main()
