#!/usr/bin/env python3
"""Download and extract the official CITRIS/iCITRIS video benchmarks.

The datasets are released on Zenodo by the CITRIS authors.  They are large
(gigabytes) and are therefore intentionally not committed to the repository.
This script mirrors the release into ``data/citris`` (or ``--root``) using the
same archive layout read by :mod:`gcmts.data.citris`.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATASETS = {
    "ball_in_boxes": (
        "https://zenodo.org/api/records/6637749/files/Ball_In_Boxes.zip/content",
        "Ball_In_Boxes.zip",
        "ball",
    ),
    "pong": (
        "https://zenodo.org/api/records/6637749/files/Interventional_Pong.zip/content",
        "Interventional_Pong.zip",
        "pong",
    ),
    "temporal_causal3d": (
        "https://zenodo.org/api/records/6637749/files/Temporal_Causal3dIdent_7shapes.zip/content",
        "Temporal_Causal3dIdent_7shapes.zip",
        "temporal_causal3d",
    ),
    "pinball": (
        "https://zenodo.org/api/records/6570441/files/pinball.zip/content",
        "pinball.zip",
        "pinball",
    ),
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination}")
        return
    print(f"[download] {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/citris"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATASETS),
        choices=sorted(DATASETS),
        help="Subset to download (default: all).",
    )
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        url, archive_name, extract_dir = DATASETS[name]
        archive = args.root / archive_name
        _download(url, archive)
        target = args.root / extract_dir
        marker = target / ".extracted"
        if not marker.exists():
            print(f"[extract] {archive} -> {target}")
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
            marker.write_text("ok\n")
        if not args.keep_archives:
            archive.unlink(missing_ok=True)
    print("CITRIS data ready under", args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
