"""Registry of the real (or real-derived) datasets named in the review.

Real-world datasets are catalogued here with the metadata that decides how they
may be used under rule R1 of the review: a dataset may populate a metric only
when it exposes the ground truth that metric needs. Three roles are possible:

* ``soundness`` --- the causal object is (partly) observed, so the dataset can
  certify a claim: recorded interventions (single-cell perturbation screens,
  L2) or randomisation (real RCTs, L3 marginal effects);
* ``external_validity`` --- no causal ground truth; the dataset can only support
  downstream-utility or out-of-distribution comparisons, never identifiability.

``level`` is the highest rung of Pearl's hierarchy the dataset exposes,
``ground_truth`` describes what (if anything) is observed, and ``access`` records
whether the data can be fetched automatically (``open``), needs a free
registration (``registration``) or is credentialed (``credentialed``).

The registry is the single source of truth for the external-validity harness
(:mod:`scripts.run_real_benchmarks`), the download helper
(:mod:`scripts.download_datasets`) and the paper tables
(:mod:`scripts.make_paper_tables`).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["RealDataset", "REAL_DATASETS", "real_label", "real_by_role"]


@dataclass(frozen=True)
class RealDataset:
    """Paper-facing metadata for one real or real-derived dataset."""

    name: str
    label: str
    level: str
    ground_truth: str
    modality: str
    role: str  # "soundness" | "external_validity"
    access: str  # "open" | "registration" | "credentialed"
    source: str
    loader: str  # "csv" | "covid_panel" | "rct" | "netcdf" | "single_cell" | "mocap" | "clinical"


REAL_DATASETS: dict[str, RealDataset] = {
    # -- Climate, earth and energy (external validity only) ------------------
    "etth1": RealDataset(
        "etth1", "ETTh1", "L1/L2", "partial (graph proxy)", "energy series",
        "external_validity", "open", "github.com/zhouhaoyi/ETDataset", "csv",
    ),
    "etth2": RealDataset(
        "etth2", "ETTh2", "L1/L2", "partial (graph proxy)", "energy series",
        "external_validity", "open", "github.com/zhouhaoyi/ETDataset", "csv",
    ),
    "ettm1": RealDataset(
        "ettm1", "ETTm1", "L1/L2", "partial (graph proxy)", "energy series",
        "external_validity", "open", "github.com/zhouhaoyi/ETDataset", "csv",
    ),
    "jena": RealDataset(
        "jena", "Jena Climate", "L1/L2", "partial (physics proxy)", "climate series",
        "external_validity", "open", "tf-keras-datasets/jena_climate", "csv",
    ),
    "weatherbench": RealDataset(
        "weatherbench", "WeatherBench (ERA5)", "L1/L2", "partial (physics proxy)",
        "climate field", "external_validity", "open (large)",
        "github.com/pangeo-data/WeatherBench", "netcdf",
    ),
    "cesm2": RealDataset(
        "cesm2", "CESM2 / ClimateBench", "L1/L2", "partial (simulator proxy)", "climate field",
        "external_validity", "open (large)", "github.com/duncanwp/ClimateBench", "netcdf",
    ),
    "exchange_rate": RealDataset(
        "exchange_rate", "Exchange rate", "L1", "none (proxy)", "financial series",
        "external_validity", "open", "github.com/laiguokun/multivariate-time-series-data", "dense",
    ),
    "solar_al": RealDataset(
        "solar_al", "Solar (AL)", "L1/L2", "partial (irradiance proxy)", "energy series",
        "external_validity", "open", "github.com/laiguokun/multivariate-time-series-data", "dense",
    ),
    "electricity": RealDataset(
        "electricity", "Electricity", "L1/L2", "partial (grid proxy)", "energy series",
        "external_validity", "open", "github.com/laiguokun/multivariate-time-series-data", "dense",
    ),
    "traffic": RealDataset(
        "traffic", "Traffic (San Francisco)", "L1/L2", "partial (road proxy)", "sensor panel",
        "external_validity", "open", "github.com/laiguokun/multivariate-time-series-data", "dense",
    ),
    "beijing_pm25": RealDataset(
        "beijing_pm25", "Beijing PM2.5", "L1", "none (proxy)", "air-quality series",
        "external_validity", "open", "archive.ics.uci.edu/dataset/381", "pm25",
    ),
    # -- Socio-economic and urban mobility (external validity only) ----------
    "covid_policy": RealDataset(
        "covid_policy", "COVID-19 policy panel", "L2/L3", "none (proxy)", "panel data",
        "external_validity", "open", "github.com/OxCGRT/covid-policy-dataset", "covid_panel",
    ),
    # -- Single-cell and biological perturbations (recorded interventions) ---
    "perturb_seq": RealDataset(
        "perturb_seq", "Perturb-seq / scPerturb", "L2", "recorded perturbations",
        "single-cell genomics", "soundness", "open",
        "github.com/sanderlab/scPerturb", "single_cell",
    ),
    # -- Clinical and healthcare (partial ground truth) ----------------------
    "mimic": RealDataset(
        "mimic", "MIMIC-III / IV", "L2/L3", "partial (cohort outcome)", "clinical series",
        "external_validity", "credentialed", "physionet.org/content/mimiciii", "clinical",
    ),
    "ms_lesion": RealDataset(
        "ms_lesion", "MS lesion (ISBI 2015)", "L2/L3", "partial (lesion mask)", "medical imaging",
        "external_validity", "open", "kaggle.com/datasets/marwa96/isbi-ms-dataset", "clinical",
    ),
    # -- Motion capture (physical, high-dimensional, GT pose) ----------------
    "human36m": RealDataset(
        "human36m", "Human3.6M", "L1", "3D pose", "skeleton MoCap",
        "soundness", "registration", "vision.imar.ro/human3.6m", "mocap",
    ),
    "humaneva": RealDataset(
        "humaneva", "HumanEva-I", "L1", "3D pose", "skeleton MoCap",
        "soundness", "registration", "vision.cs.brown.edu", "mocap",
    ),
    # -- Video reasoning (non-invertible projection, proxy) ------------------
    "sutd_trafficqa": RealDataset(
        "sutd_trafficqa", "SUTD-TrafficQA", "L1/L2", "none (proxy)", "driving video",
        "external_validity", "open", "github.com/SUTDCV/SUTD-TrafficQA", "csv",
    ),
    # -- Real RCTs (L3 marginal effects; randomisation identifies the ATE) ---
    "lalonde_rct": RealDataset(
        "lalonde_rct", "LaLonde RCT (NSW)", "L3", "experimental ATE", "socio-economic",
        "soundness", "open", "github.com/NBER/LaLonde", "rct",
    ),
    "oregon_rct": RealDataset(
        "oregon_rct", "Oregon Health Insurance", "L3", "experimental ATE", "health panel",
        "soundness", "open", "dataverse.harvard.edu/dataverse/oregon", "rct",
    ),
}


def real_label(key: str) -> str:
    """Return the paper-facing name of a real dataset key."""
    return REAL_DATASETS[key].label


def real_by_role(role: str) -> dict[str, RealDataset]:
    """Return the datasets whose role equals ``role``."""
    return {key: value for key, value in REAL_DATASETS.items() if value.role == role}
