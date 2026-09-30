"""Download the real-world time-series datasets used for external validity.

All sources are public; no credentials are required. Files land in
``data/real/`` (git-ignored). Run once before ``run_real_benchmarks.py``::

    uv run python scripts/download_datasets.py
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
SOURCES = {
    "ETTh1.csv": f"{BASE}/ETTh1.csv",
    "ETTh2.csv": f"{BASE}/ETTh2.csv",
    "ETTm1.csv": f"{BASE}/ETTm1.csv",
}
JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists  {dest}")
        return
    print(f"fetching {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted, fixed URLs)
    print(f"saved   {dest}")


def main() -> int:
    out = Path(__file__).resolve().parent.parent / "data" / "real"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        _download(url, out / name)

    jena_zip = out / "jena_climate.zip"
    _download(JENA_URL, jena_zip)
    jena_csv = out / "jena_climate_2009_2016.csv"
    if not jena_csv.exists():
        with zipfile.ZipFile(jena_zip) as archive:
            archive.extractall(out)
        print(f"unzipped {jena_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
