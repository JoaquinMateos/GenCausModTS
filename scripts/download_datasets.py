"""Fetch or generate the datasets used by the benchmark suite.

Sources are public; no credentials are required for the automatic downloads.
Files land under ``data/`` (git-ignored). Usage::

    uv run python scripts/download_datasets.py                 # real series + IHDP
    uv run python scripts/download_datasets.py --generate      # CITRIS/Causal3DIdent tensors

ACIC 2016/2018 requires registration at the competition site and is therefore
documented, not fetched automatically: download the archives from
``github.com/vdorie/aciccomp`` (or the challenge site), convert them to CSV, and
place them under ``data/acic/<replication>/`` as ``x.csv``, ``z.csv``,
``y.csv``, ``mu0.csv`` and ``mu1.csv`` (see
:class:`gcmts.data.semi_synthetic.ACICLoader`).
"""

from __future__ import annotations

import argparse
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
IHDP_BASE = "https://www.fredjo.com/files"
IHDP_FILES = [
    "ihdp_npci_1-100.train.npz",
    "ihdp_npci_1-100.test.npz",
]


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists  {dest}")
        return
    print(f"fetching {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted, fixed URLs)
    print(f"saved   {dest}")


def download_real(out: Path) -> None:
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


def download_semi_synthetic(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in IHDP_FILES:
        _download(f"{IHDP_BASE}/{name}", out / name)
    print(
        "ACIC: not downloaded automatically (registration required); see the "
        "module docstring for the expected data/acic layout."
    )


def generate_synthetic(root: Path) -> None:
    """Materialise the CITRIS-style generators as tensors for cheap reuse."""
    import torch

    from gcmts.data.voronoi import VoronoiGenerator

    out = root / "generated"
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        generator = VoronoiGenerator(n_vars=5, horizon=16, seed=seed)
        batch = generator.sample(2048)
        payload = {
            "x": batch.x,
            "z": batch.z,
            "I": batch.context["I"] if batch.context else None,
            "adjacency": batch.adjacency,
        }
        torch.save(payload, out / f"voronoi_seed{seed}.pt")
        print(f"saved   {out / f'voronoi_seed{seed}.pt'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/generate benchmark datasets.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="also materialise the synthetic CITRIS/Voronoi tensors under data/generated/",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent / "data"
    download_real(root / "real")
    download_semi_synthetic(root / "semi_synthetic")
    if args.generate:
        generate_synthetic(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
