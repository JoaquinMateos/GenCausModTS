"""Download and convert the ACIC 2018 challenge data (Zenodo mirror).

The official ACIC distribution is an R package (2016) or requires a Synapse
account (2018). This script uses the open CC-BY Zenodo mirror of ACIC 2018
(record 19175621) and converts a reproducible subset of its 2,592 scaling
datasets into the layout expected by :class:`gcmts.data.semi_synthetic.ACICLoader`:
one directory per dataset containing headerless ``x.csv``, ``z.csv``,
``y.csv``, ``mu0.csv`` and ``mu1.csv``.

Note on ground truth: the mirror ships realised potential outcomes
(``y0``/``y1``), not the noiseless response surfaces, so ``mu*`` here are the
realised counterfactuals (the same convention used by caussim); PEHE therefore
measures error against realised potential outcomes.

Usage::

    uv run python scripts/fetch_acic_zenodo.py --download
    uv run python scripts/fetch_acic_zenodo.py --archive data/acic/acic2018.tar.gz
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://zenodo.org/api/records/19175621/files/acic2018.tar.gz/content"


def download(archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.stat().st_size > 0:
        print(f"{archive} already exists; skipping download")
        return
    print(f"downloading {URL} -> {archive}")
    with urllib.request.urlopen(URL) as response, archive.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            seen += len(chunk)
            if total:
                print(f"  {seen / 1e6:.0f}/{total / 1e6:.0f} MB", end="\r")
    print()


def extract(archive: Path, extract_dir: Path) -> Path:
    if not (extract_dir / "acic2018").exists():
        print(f"extracting {archive} -> {extract_dir}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
    return extract_dir / "acic2018"


def select_datasets(params: pd.DataFrame, n_replications: int) -> list[str]:
    """One dataset per DGP first (diversity), then the smallest remaining."""
    params = params.copy()
    params["size"] = pd.to_numeric(params["size"], errors="coerce")
    params = params.sort_values("size")
    chosen: list[str] = []
    seen_dgp: set[str] = set()
    for _, row in params.iterrows():
        if len(chosen) >= n_replications:
            break
        dgp = str(row.get("dgp", row["ufid"]))
        if dgp not in seen_dgp:
            seen_dgp.add(dgp)
            chosen.append(str(row["ufid"]))
    for _, row in params.iterrows():
        if len(chosen) >= n_replications:
            break
        ufid = str(row["ufid"])
        if ufid not in chosen:
            chosen.append(ufid)
    return chosen


def convert(root: Path, out_dir: Path, n_replications: int) -> None:
    scaling = root / "scaling" / "scaling"
    params = pd.read_csv(scaling / "params.csv")
    covariates = pd.read_csv(root / "x.csv")
    sample_column = covariates.columns[0]
    covariates = covariates.set_index(sample_column)
    chosen = select_datasets(params, n_replications)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, float | str]] = []
    for ufid in chosen:
        factual = pd.read_csv(scaling / "factuals" / f"{ufid}.csv")
        counterfactual = pd.read_csv(scaling / "counterfactuals" / f"{ufid}_cf.csv")
        merged = factual.merge(counterfactual, on="sample_id", how="inner")
        x = covariates.loc[merged["sample_id"]].to_numpy(dtype=np.float32)
        z = merged["z"].to_numpy(dtype=np.int64)
        y = merged["y"].to_numpy(dtype=np.float32)
        mu0 = merged["y0"].to_numpy(dtype=np.float32)
        mu1 = merged["y1"].to_numpy(dtype=np.float32)
        target = out_dir / ufid
        target.mkdir(parents=True, exist_ok=True)
        np.savetxt(target / "x.csv", x, delimiter=",")
        np.savetxt(target / "z.csv", z, delimiter=",")
        np.savetxt(target / "y.csv", y, delimiter=",")
        np.savetxt(target / "mu0.csv", mu0, delimiter=",")
        np.savetxt(target / "mu1.csv", mu1, delimiter=",")
        metadata.append(
            {
                "ufid": ufid,
                "n": float(x.shape[0]),
                "true_ate": float(np.mean(mu1 - mu0)),
                "treated_fraction": float(np.mean(z)),
            }
        )
        print(
            f"  {ufid}: n={x.shape[0]} p={x.shape[1]} ATE={metadata[-1]['true_ate']:.3f}"
        )
    pd.DataFrame(metadata).to_csv(out_dir / "metadata.csv", index=False)
    print(f"wrote {len(chosen)} replications to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/acic/acic2018.tar.gz"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/acic/extracted"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/acic"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--n-replications", type=int, default=100)
    args = parser.parse_args()
    if args.download:
        download(args.archive)
    root = extract(args.archive, args.extract_dir)
    convert(root, args.out_dir, args.n_replications)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
