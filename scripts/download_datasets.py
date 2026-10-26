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
OXCGRT_URL = (
    "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/"
    "data/OxCGRT_compact_national_v1.csv"
)
LALONDE_URL = "https://raw.githubusercontent.com/scunning1975/mixtape/master/nsw_mixtape.dta"
NHEFS_URL = (
    "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
    "causallib/datasets/data/nhefs/NHEFS.csv"
)
MTS_BASE = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master"
DENSE_SOURCES = {
    "exchange_rate": ("exchange_rate/exchange_rate.txt.gz", 8),
    "solar_al": ("solar-energy/solar_AL.txt.gz", 16),
    "electricity": ("electricity/electricity.txt.gz", 16),
    "traffic": ("traffic/traffic.txt.gz", 16),
}
PM25_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
    "PRSA_data_2010.1.1-2014.12.31.csv"
)
SCPERTURB_URL = (
    "https://zenodo.org/records/13350497/files/NormanWeissman2019_filtered.h5ad?download=1"
)
CLIMATEBENCH_BASE = "https://zenodo.org/api/records/7064308/files"
CLIMATEBENCH_FILES = ["test.tar.gz"]
STATIC_BASE = "https://www.fredjo.com/files"
STATIC_FILES = {
    "jobs_train.npz": f"{STATIC_BASE}/jobs_DW_bin.new.10.train.npz",
    "jobs_test.npz": f"{STATIC_BASE}/jobs_DW_bin.new.10.test.npz",
    "news.zip": f"{STATIC_BASE}/NEWS_csv.zip",
}
TWINS_URL = (
    "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/"
    "0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"
)


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"exists  {dest}")
        return
    print(f"fetching {url}")
    part = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, part)  # noqa: S310 (trusted, fixed URLs)
    part.replace(dest)
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


def download_static(out: Path) -> None:
    """Canonical real-covariate static effect benchmarks (Twins/Jobs/News)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, url in STATIC_FILES.items():
        _download(url, out / name)
    _download(TWINS_URL, out / "twins.csv.gz")
    news_dir = out / "news"
    marker = news_dir / "csv" / "topic_doc_mean_n5000_k3477_seed_1.csv.x"
    if not marker.exists():
        with zipfile.ZipFile(out / "news.zip") as archive:
            archive.extractall(out)
        extracted = out / "csv"
        if extracted.exists() and not news_dir.exists():
            extracted.rename(news_dir)
        print(f"unzipped {news_dir}")
    else:
        print(f"exists   {marker}")


def download_effects(out: Path) -> None:
    """Fetch the real effect-estimation cohorts (LaLonde RCT, NHEFS)."""
    out.mkdir(parents=True, exist_ok=True)
    _download(LALONDE_URL, out / "nsw_mixtape.dta")
    _download(NHEFS_URL, out / "NHEFS.csv")


def download_policy(out: Path) -> None:
    """Fetch the Oxford policy tracker and materialise the daily COVID panel."""
    from gcmts.data.real_panels import build_covid_panel

    out.mkdir(parents=True, exist_ok=True)
    raw = out / "OxCGRT_compact_national_v1.csv"
    _download(OXCGRT_URL, raw)
    panel = build_covid_panel(raw, out / "covid_policy.csv")
    print(f"built   {panel}")


def download_dense(out: Path) -> None:
    """Fetch the small standard multivariate benchmarks and convert to CSV."""
    from gcmts.data.real_formats import dense_text_to_csv, pm25_to_csv

    out.mkdir(parents=True, exist_ok=True)
    for name, (rel, max_channels) in DENSE_SOURCES.items():
        raw = out / f"{name}.txt.gz"
        _download(f"{MTS_BASE}/{rel}", raw)
        panel = dense_text_to_csv(raw, out / f"{name}.csv", max_channels=max_channels)
        print(f"built   {panel}")
    raw = out / "PRSA_data_2010.1.1-2014.12.31.csv"
    _download(PM25_URL, raw)
    panel = pm25_to_csv(raw, out / "beijing_pm25.csv")
    print(f"built   {panel}")


def download_single_cell(out: Path) -> None:
    """Fetch the standard Perturb-seq benchmark (Norman et al. 2019, scPerturb).

    The harmonised scPerturb file carries the recorded perturbations in
    ``obs['perturbation']``; it is the reference dataset for unseen-perturbation
    shift prediction (L2, rule R1).
    """
    out.mkdir(parents=True, exist_ok=True)
    _download(SCPERTURB_URL, out / "perturb_seq.h5ad")


def download_climate(out: Path) -> None:
    """Fetch ClimateBench v1.0 and build a spatio-temporal panel from ssp245."""
    import tarfile

    from gcmts.data.real_formats import climatebench_to_csv

    out.mkdir(parents=True, exist_ok=True)
    target = out / "climatebench"
    target.mkdir(exist_ok=True)
    for name in CLIMATEBENCH_FILES:
        archive = out / name
        _download(f"{CLIMATEBENCH_BASE}/{name}/content", archive)
        with tarfile.open(archive) as handle:
            handle.extractall(target, filter="data")
    panel = climatebench_to_csv(target / "outputs_ssp245.nc", out / "cesm2.csv")
    print(f"built   {panel}")


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


def document_manual() -> None:
    """Print the sources that need manual download or credentialed access."""
    rows = [
        ("weatherbench", "WeatherBench/ERA5", "github.com/pangeo-data/WeatherBench (TB-scale)",
         "weatherbench.zarr"),
        ("human36m", "Human3.6M", "vision.imar.ro/human3.6m (registration)", "human36m.npz"),
        ("humaneva", "HumanEva-I", "vision.cs.brown.edu/humaneva (registration)", "humaneva.npz"),
        ("mimic", "MIMIC-III/IV", "physionet.org/content/mimiciii (credentialed)",
         "mimic_cohort.csv"),
        ("ms_lesion", "MS lesion ISBI 2015",
         "kaggle.com/datasets/marwa96/isbi-ms-dataset", "ms_lesion.nii.gz"),
        ("oregon_rct", "Oregon Health Ins.",
         "dataverse.harvard.edu/dataverse/oregon", "dataverse_files/Oregon/oregon_puf.zip"),
        ("sutd_trafficqa", "SUTD-TrafficQA",
         "github.com/SUTDCV/SUTD-TrafficQA (video)", "n/a"),
    ]
    print(
        "\nDatasets NOT downloaded automatically (large, registered or credentialed)."
        "\nPlace the file under data/real/ with the target name; the loader"
        "\ngcmts.data.real_panels.load_real_dataset(name) then reads it:\n"
    )
    for name, label, source, target in rows:
        print(f"  {name:14s} {label:20s} {source:44s} -> {target}")


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
    download_effects(root / "real")
    download_policy(root / "real")
    download_dense(root / "real")
    download_single_cell(root / "real")
    download_static(root / "static")
    download_semi_synthetic(root / "semi_synthetic")
    try:
        download_climate(root / "real")
    except ImportError as error:
        print(
            f"skipping CESM2 CSV conversion: {error}. The archive is downloaded; "
            "install the optional real dependencies with `uv sync --extra real`."
        )
    if args.generate:
        generate_synthetic(root)
    document_manual()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
