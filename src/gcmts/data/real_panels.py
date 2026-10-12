"""Loaders for the real and real-derived datasets of the review's inventory.

These are the datasets that do *not* fit the synthetic generators: real panels
used for external validity (climate, energy, policy) and real effect-estimation
cohorts used for treatment-effect evaluation (a real RCT and a canonical
observational cohort). Everything here is dependency-light (``pandas``/
``numpy``) so the loaders run wherever the core library does; format-heavy
datasets (NetCDF, h5ad, NIfTI) are documented in :mod:`gcmts.data.real_registry`
and loaded through :func:`load_real_dataset`, which raises an informative error
when the optional dependency or the local file is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "EffectCohort",
    "build_covid_panel",
    "load_real_dataset",
    "load_lalonde",
    "load_oregon",
    "load_nhefs",
]


@dataclass(frozen=True)
class EffectCohort:
    """A real treatment-effect cohort with a reference (marginal) effect.

    Individual counterfactuals are unobserved, so only the marginal effect
    ``reference_ate`` (from randomisation or a published estimate) is exposed;
    individual-effect metrics (PEHE) are undefined for these datasets.
    """

    name: str
    x: np.ndarray
    t: np.ndarray
    y: np.ndarray
    reference_ate: float
    reference: str
    kind: str  # "rct" | "observational"


def load_lalonde(path: str | Path) -> EffectCohort:
    """Load the NSW/LaLonde experimental cohort (Dehejia--Wahba).

    The experimental benchmark exposes the ATE by randomisation; covariates are
    ``age, education, black, hispanic, married, nodegree, re74, re75`` and the
    outcome is ``re78`` (1978 earnings).
    """
    frame = pd.read_stata(path)
    covariates = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
    covariates = [column for column in covariates if column in frame.columns]
    x = frame[covariates].to_numpy(dtype=np.float64)
    t = frame["treat"].to_numpy(dtype=np.int64)
    y = frame["re78"].to_numpy(dtype=np.float64)
    treated, control = y[t == 1], y[t == 0]
    reference_ate = float(treated.mean() - control.mean())
    return EffectCohort(
        name="lalonde_rct",
        x=x,
        t=t,
        y=y,
        reference_ate=reference_ate,
        reference="experimental difference in means (randomised NSW)",
        kind="rct",
    )


def load_oregon(
    path: str | Path, *, outcome: str = "num_visit_cens_ed"
) -> EffectCohort:
    """Load the Oregon Health Insurance Experiment (OHIE) public-use files.

    The Medicaid lottery randomises the *offer* of coverage, so the difference
    in means of the outcome by ``treatment`` is an unbiased estimate of the
    intention-to-treat effect (L3, marginal). ``path`` is the ``oregon_puf.zip``
    archive; the emergency-department sample (``oregonhie_ed_vars``) is joined
    to the descriptive variables on ``person_id``.
    """
    import io
    import zipfile

    with zipfile.ZipFile(path) as archive:
        def read(name: str) -> pd.DataFrame:
            with archive.open(name) as handle:
                return pd.read_stata(io.BytesIO(handle.read()), convert_categoricals=False)

        descriptive = read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_descriptive_vars.dta")
        effects = read("OHIE_Public_Use_Files/OHIE_Data/oregonhie_ed_vars.dta")
    frame = descriptive.merge(effects[["person_id", outcome]], on="person_id", how="inner")
    frame = frame.dropna(subset=[outcome])
    covariates = [
        "female_list", "birthyear_list", "numhh_list", "have_phone_list",
        "english_list", "self_list", "pobox_list",
    ]
    covariates = [column for column in covariates if column in frame.columns]
    x = frame[covariates].to_numpy(dtype=np.float64)
    t = frame["treatment"].to_numpy(dtype=np.int64)
    y = frame[outcome].to_numpy(dtype=np.float64)
    reference_ate = float(y[t == 1].mean() - y[t == 0].mean())
    return EffectCohort(
        name="oregon_rct",
        x=x,
        t=t,
        y=y,
        reference_ate=reference_ate,
        reference="experimental intention-to-treat (Medicaid lottery)",
        kind="rct",
    )


def load_nhefs(path: str | Path) -> EffectCohort:
    """Load the NHEFS smoking-cessation cohort (Hernán--Robins).

    Observational, so the reference is the published confounder-adjusted
    estimate, not a randomised effect; the treatment is ``qsmk`` and the outcome
    ``wt82_71`` (weight change in kg).
    """
    frame = pd.read_csv(path)
    covariates = [
        "sex", "age", "race", "education", "smokeintensity", "smokeyrs",
        "exercise", "active", "wt71",
    ]
    covariates = [column for column in covariates if column in frame.columns]
    x = frame[covariates].to_numpy(dtype=np.float64)
    t = frame["qsmk"].to_numpy(dtype=np.int64)
    y = frame["wt82_71"].to_numpy(dtype=np.float64)
    return EffectCohort(
        name="nhefs",
        x=x,
        t=t,
        y=y,
        reference_ate=3.5,
        reference="published confounder-adjusted estimate (Hernán & Robins)",
        kind="observational",
    )


def build_covid_panel(oxcgrt_csv: str | Path, out_csv: str | Path) -> Path:
    """Aggregate the Oxford COVID-19 policy tracker into a global daily panel.

    Reads the compact national file and writes a wide CSV with a ``Date`` column
    and numeric channels (mean policy indices, summed cases and deaths) so that
    it can be windowed by :class:`gcmts.data.real.RealTimeSeries`.
    """
    frame = pd.read_csv(oxcgrt_csv, low_memory=False)
    frame = frame.rename(columns={"Date": "date"})
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["date"])
    index_columns = [
        column
        for column in ("StringencyIndex_Average", "ContainmentHealthIndex_Average")
        if column in frame.columns
    ]
    case_columns = [
        column for column in ("ConfirmedCases", "ConfirmedDeaths") if column in frame.columns
    ]
    grouped = frame.groupby("date")
    if index_columns:
        panel = grouped[index_columns].mean()
    else:
        panel = pd.DataFrame(index=grouped.size().index)
    for column in case_columns:
        panel[column] = grouped[column].sum()
    panel = panel.sort_index().reset_index().rename(columns={"date": "Date"})
    panel.to_csv(out_csv, index=False)
    return Path(out_csv)


def load_real_dataset(name: str, data_dir: str | Path) -> object:
    """Dispatch to the loader for ``name`` given a data directory.

    CSV panels (ETT, Jena, COVID) are returned as file paths for
    :class:`~gcmts.data.real.RealTimeSeries`; effect cohorts return an
    :class:`EffectCohort`. Datasets that need optional dependencies or manual
    access raise an informative error.
    """
    data_dir = Path(data_dir)
    csv_paths = {
        "etth1": data_dir / "ETTh1.csv",
        "etth2": data_dir / "ETTh2.csv",
        "ettm1": data_dir / "ETTm1.csv",
        "jena": data_dir / "jena_climate_2009_2016.csv",
        "covid_policy": data_dir / "covid_policy.csv",
        "exchange_rate": data_dir / "exchange_rate.csv",
        "solar_al": data_dir / "solar_al.csv",
        "electricity": data_dir / "electricity.csv",
        "traffic": data_dir / "traffic.csv",
        "beijing_pm25": data_dir / "beijing_pm25.csv",
        "cesm2": data_dir / "cesm2.csv",
    }
    if name in csv_paths:
        path = csv_paths[name]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run scripts/download_datasets.py (or place the file there)."
            )
        return path
    if name == "lalonde_rct":
        return load_lalonde(data_dir / "nsw_mixtape.dta")
    if name == "oregon_rct":
        return load_oregon(data_dir / "dataverse_files" / "Oregon" / "oregon_puf.zip")
    if name == "nhefs":
        return load_nhefs(data_dir / "NHEFS.csv")
    from gcmts.data.real_formats import (
        load_clinical_csv,
        load_mocap,
        load_nifti,
        load_single_cell,
        netcdf_to_csv,
    )

    if name == "weatherbench":
        source = data_dir / "weatherbench.zarr"
        if not source.exists():
            raise FileNotFoundError(
                f"{source} missing; see scripts/download_datasets.py for the source and layout."
            )
        return netcdf_to_csv(source, data_dir / "weatherbench_panel.csv")
    if name == "perturb_seq":
        return load_single_cell(data_dir / "perturb_seq.h5ad")
    if name in {"human36m", "humaneva"}:
        return load_mocap(data_dir / f"{name}.npz")
    if name == "ms_lesion":
        return load_nifti(data_dir / "ms_lesion.nii.gz")
    if name == "mimic":
        return load_clinical_csv(data_dir / "mimic_cohort.csv")
    raise NotImplementedError(
        f"Dataset '{name}' needs manual access (see gcmts.data.real_registry); "
        "provide the file under data/ and add a loader."
    )
