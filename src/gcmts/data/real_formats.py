"""Format-aware loaders for the remaining real datasets of the inventory.

These cover the datasets whose storage format is not a plain CSV: gridded
climate fields (NetCDF / Zarr), single-cell perturbation screens (``.h5ad``),
motion-capture skeletons (``.npz``/``.mat``) and medical volumes (NIfTI). The
heavy format libraries are optional (``pip install gcmts[real]``); importing
this module never fails, and each loader raises an informative error naming the
missing dependency or file.

The grid and imaging loaders are for external-validity use only: they expose no
causal ground truth (rule R1). The single-cell loader is the exception---the
perturbations are recorded interventions (L2), so it can support an
unseen-perturbation shift comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "PerturbationScreen",
    "netcdf_to_csv",
    "open_grid",
    "load_single_cell",
    "load_mocap",
    "load_nifti",
    "load_clinical_csv",
    "dense_text_to_csv",
    "pm25_to_csv",
    "climatebench_to_csv",
]


@dataclass(frozen=True)
class PerturbationScreen:
    """A single-cell perturbation screen (recorded interventions, L2).

    ``adata`` is the AnnData object; ``perturbations`` is the per-cell
    perturbation label taken from ``obs``. Because the perturbation is recorded
    but the latent cellular programs are not, evaluation targets the prediction
    of distributional shifts for *unseen* perturbations.
    """

    name: str
    adata: object
    perturbations: np.ndarray
    obs_key: str


def _require(module: str, extra: str = "real") -> object:
    try:
        return __import__(module)
    except ImportError as error:  # pragma: no cover - exercised only without extra
        raise ImportError(
            f"'{module}' is required for this dataset; install the optional "
            f"dependencies with `uv sync --extra {extra}`."
        ) from error


def open_grid(path: str | Path, *, variable: str | None = None, engine: str | None = None):
    """Open a gridded climate field (NetCDF or Zarr) as an xarray DataArray."""
    xr = _require("xarray")
    path = str(path)
    if path.endswith(".zarr") or engine == "zarr":
        dataset = xr.open_zarr(path)
    else:
        dataset = xr.open_dataset(path, engine=engine)
    name = variable or next(iter(dataset.data_vars))
    return dataset[name]


def netcdf_to_csv(
    path: str | Path,
    out_csv: str | Path,
    *,
    variable: str | None = None,
    time_dim: str = "time",
    max_cells: int = 8,
    engine: str | None = None,
) -> Path:
    """Flatten a gridded field into a multivariate CSV windowed by RealTimeSeries.

    The field is reduced to at most ``max_cells`` spatial locations (a leading
    spatial slice) so that the resulting panel has a tractable channel count;
    each remaining spatial cell becomes one channel indexed by time.
    """
    array = open_grid(path, variable=variable, engine=engine)
    if time_dim not in array.dims:
        raise ValueError(f"time dimension '{time_dim}' not found in {array.dims}.")
    spatial = [dim for dim in array.dims if dim != time_dim]
    for dim in spatial:
        array = array.isel({dim: 0}) if array.sizes[dim] > 1 else array
    frame = array.to_pandas()
    if frame.ndim == 1:
        frame = frame.to_frame()
    if frame.shape[1] > max_cells:
        frame = frame.iloc[:, :max_cells]
    frame = frame.reset_index().rename(columns={frame.index.name or "index": "Date"})
    frame.to_csv(out_csv, index=False)
    return Path(out_csv)


def load_single_cell(path: str | Path, *, obs_key: str = "perturbation") -> PerturbationScreen:
    """Load a ``.h5ad`` single-cell perturbation screen."""
    ad = _require("anndata")
    adata = ad.read_h5ad(path)
    if obs_key in adata.obs:
        perturbations = adata.obs[obs_key].to_numpy()
    else:
        perturbations = np.full(adata.n_obs, "unknown", dtype=object)
    return PerturbationScreen(
        name=Path(path).stem,
        adata=adata,
        perturbations=perturbations,
        obs_key=obs_key,
    )


def load_mocap(path: str | Path, *, key: str | None = None) -> np.ndarray:
    """Load a motion-capture skeleton array from ``.npz`` or ``.mat``.

    Expects an array shaped ``(n_frames, n_joints, 3)`` (or ``(n_frames,
    3*n_joints)``); the 3D pose is the ground truth used by the hierarchical
    temporal methods.
    """
    path = Path(path)
    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        name = key or next(iter(archive.files))
        array = archive[name]
    elif path.suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(path)
        candidates = [value for name, value in data.items() if not name.startswith("__")]
        if key is not None:
            candidates = [data[key]]
        array = candidates[0]
    else:
        raise ValueError(f"Unsupported motion-capture format: {path.suffix}.")
    return np.asarray(array, dtype=np.float64)


def load_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume (e.g. an MS lesion mask) as a NumPy array."""
    nib = _require("nibabel")
    image = nib.load(str(path))
    return np.asarray(image.dataobj, dtype=np.float32)


def load_clinical_csv(path: str | Path, *, max_channels: int = 16) -> np.ndarray:
    """Load a clinical time-series CSV (e.g. an extracted MIMIC cohort)."""
    import pandas as pd

    frame = pd.read_csv(path)
    frame = frame.select_dtypes(include=[np.number])
    return frame.to_numpy(dtype=np.float64)[:, :max_channels]


def dense_text_to_csv(
    path: str | Path,
    out_csv: str | Path,
    *,
    max_channels: int | None = None,
    sep: str = ",",
    compression: str = "infer",
) -> Path:
    """Convert a headerless dense matrix (``.txt[.gz]``) into a RealTimeSeries CSV.

    Used by the small standard multivariate benchmarks (exchange rate, solar,
    electricity, traffic) from the ``laiguokun/multivariate-time-series-data``
    collection: one row per timestep, one column per channel, no header.
    """
    import pandas as pd

    frame = pd.read_csv(
        path, header=None, sep=sep, compression=compression, engine="python"
    )
    if max_channels is not None:
        frame = frame.iloc[:, :max_channels]
    frame.insert(0, "Date", pd.RangeIndex(len(frame)))
    frame.to_csv(out_csv, index=False)
    return Path(out_csv)


def climatebench_to_csv(
    path: str | Path,
    out_csv: str | Path,
    *,
    member: int = 0,
    grid: int = 4,
    max_channels: int = 16,
) -> Path:
    """Flatten a ClimateBench output field into a spatio-temporal panel.

    Uses one ensemble member and a small ``grid`` x ``grid`` spatial patch per
    output variable (global-mean temperature, precipitation, ...), stacking the
    patch cells into channels so the panel stays small enough for a single GPU.
    """
    import pandas as pd

    xr = _require("xarray")
    dataset = xr.open_dataset(path)
    if "member" in dataset.dims:
        dataset = dataset.isel(member=member)
    frames = []
    names = []
    for variable in dataset.data_vars:
        array = dataset[variable]
        if not {"time", "lat", "lon"} <= set(array.dims):
            continue
        array = array.isel(lat=slice(0, grid), lon=slice(0, grid))
        stacked = array.stack(cell=("lat", "lon")).to_pandas()
        frames.append(stacked)
        names.extend(f"{variable}_{index}" for index in range(stacked.shape[1]))
    panel = pd.concat(frames, axis=1)
    panel.columns = names
    panel = panel.iloc[:, :max_channels].reset_index()
    panel = panel.rename(columns={panel.columns[0]: "Date"})
    panel.to_csv(out_csv, index=False)
    return Path(out_csv)


def pm25_to_csv(path: str | Path, out_csv: str | Path) -> Path:
    """Convert the UCI Beijing PM2.5 CSV into a RealTimeSeries CSV."""
    import pandas as pd

    frame = pd.read_csv(path)
    columns = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir", "cbwd"]
    columns = [column for column in columns if column in frame.columns]
    frame = frame[columns].copy()
    frame["cbwd"] = frame["cbwd"].astype("category").cat.codes
    frame = frame.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    frame.insert(0, "Date", pd.RangeIndex(len(frame)))
    frame.to_csv(out_csv, index=False)
    return Path(out_csv)
