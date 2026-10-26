r"""Canonical real-covariate static treatment-effect benchmarks.

The static panels used in the paper are the semi-synthetic benchmarks of
Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2017):

* **IHDP** -- loaded by :class:`gcmts.data.semi_synthetic.IHDPLoader` from the
  canonical 100-replication ``.npz`` files.
* **Twins** -- real twin-pair covariates; both potential outcomes are observed
  (the outcome is a binary survival indicator), so PEHE is defined.  Treatment
  is assigned at random with probability 1/2 in each replication, following the
  standard semi-synthetic protocol.
* **News** -- real 3,477-dimensional topic covariates; the released files
  contain the treatment, factual/counterfactual outcomes and the noiseless
  potential means, so PEHE and policy risk are defined.
* **Jobs** -- the preprocessed LaLonde/NSW binary-outcome benchmark.  It exposes
  a randomised reference ATE but no individual counterfactual, so it is used as
  a **real cohort** (marginal effect only), not as a PEHE panel.
* **CPS** -- the CFRNet authors do not ship a public mirror of the exact file;
  :class:`CPSLoader` reads a user-supplied ``cps.csv`` with the documented
  columns and raises an actionable error otherwise.

Files live under ``data/static/`` (git-ignored); ``scripts/download_datasets.py``
fetches Twins, News and Jobs automatically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from gcmts.data.real_panels import EffectCohort
from gcmts.data.semi_synthetic import IHDPReplication

__all__ = [
    "TwinsLoader",
    "NewsLoader",
    "load_jobs",
    "CPSLoader",
]


class TwinsLoader:
    """Canonical Twins benchmark: real covariates, both potential outcomes.

    Source: ``data/static/twins.csv.gz`` (the ``Twin_Data.csv.gz`` file from the
    ``mlforhealthlabpub`` mirror used by CEVAE and CATENets).  The two outcome
    columns are birth-survival indicators; treatment is randomised 50/50.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        replications: int = 100,
        seed: int = 0,
        train_fraction: float = 0.7,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing Twins file {self.path}. Run "
                "`uv run python scripts/download_datasets.py`."
            )
        self.replications = replications
        self.seed = seed
        self.train_fraction = train_fraction
        frame = pd.read_csv(self.path)
        frame.columns = [
            str(column).replace("'", "").replace("’", "").strip() for column in frame.columns
        ]
        outcome_columns = [
            column for column in frame.columns if column.startswith("outcome(t=")
        ]
        if len(outcome_columns) != 2:
            raise ValueError(f"Expected two outcome columns, found {outcome_columns}.")
        covariates = [
            column for column in frame.columns if column not in outcome_columns
        ]
        x = frame[covariates].to_numpy(dtype=np.float64)
        # 9999 and 8/9 are the documented missing codes; a column median is a
        # deterministic, covariate-only imputation (no outcome leakage).
        x = np.where(x >= 9998, np.nan, x)
        x = np.where(np.isin(x, [8.0, 9.0]), np.nan, x)
        column_median = np.nanmedian(x, axis=0)
        column_median = np.where(np.isfinite(column_median), column_median, 0.0)
        missing = ~np.isfinite(x)
        x[missing] = np.take(column_median, np.where(missing)[1])
        self.x = x.astype(np.float32)
        self.y0 = np.asarray(frame[outcome_columns[0]].to_numpy() < 9999, dtype=np.int64)
        self.y1 = np.asarray(frame[outcome_columns[1]].to_numpy() < 9999, dtype=np.int64)

    def replication(self, index: int) -> IHDPReplication:
        """Return one replication without materialising the whole list."""
        return self.load()[index]

    def load(self) -> list[IHDPReplication]:
        rng = np.random.default_rng(self.seed)
        replications: list[IHDPReplication] = []
        n = self.x.shape[0]
        for _ in range(self.replications):
            t = np.asarray(rng.random(n) < 0.5, dtype=np.int64)
            yf = np.where(t > 0, self.y1, self.y0).astype(np.float32)
            ycf = np.where(t > 0, self.y0, self.y1).astype(np.float32)
            order = rng.permutation(n)
            cut = int(round(self.train_fraction * n))
            train, test = order[:cut], order[cut:]
            replications.append(
                IHDPReplication(
                    x=self.x[train],
                    t=t[train],
                    yf=yf[train],
                    ycf=ycf[train],
                    mu0=self.y0[train].astype(np.float32),
                    mu1=self.y1[train].astype(np.float32),
                )
            )
            replications.append(
                IHDPReplication(
                    x=self.x[test],
                    t=t[test],
                    yf=yf[test],
                    ycf=ycf[test],
                    mu0=self.y0[test].astype(np.float32),
                    mu1=self.y1[test].astype(np.float32),
                )
            )
        # The runner consumes one replication and splits it again; return the
        # concatenated pair so the canonical split is preserved exactly.
        return [
            IHDPReplication(
                x=np.concatenate([replications[2 * i].x, replications[2 * i + 1].x]),
                t=np.concatenate([replications[2 * i].t, replications[2 * i + 1].t]),
                yf=np.concatenate([replications[2 * i].yf, replications[2 * i + 1].yf]),
                ycf=np.concatenate([replications[2 * i].ycf, replications[2 * i + 1].ycf]),
                mu0=np.concatenate([replications[2 * i].mu0, replications[2 * i + 1].mu0]),
                mu1=np.concatenate([replications[2 * i].mu1, replications[2 * i + 1].mu1]),
            )
            for i in range(self.replications)
        ]


def _read_sparse_x(path: Path) -> np.ndarray:
    """Read the SVM-light-style ``.x`` file of the News benchmark.

    The first line is ``n_rows,n_columns,nnz``; each following line is a
    1-based ``row,column,value`` triplet.  Returns a dense float32 matrix (the
    released benchmark is 5,000 x 3,477, about 70 MB).
    """
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        n_rows, n_columns = int(header[0]), int(header[1])
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            rows.append(int(parts[0]) - 1)
            columns.append(int(parts[1]) - 1)
            values.append(float(parts[2]))
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(n_rows, n_columns), dtype=np.float32
    )
    return np.asarray(matrix.toarray(), dtype=np.float32)


class NewsLoader:
    """Canonical News benchmark: real topic covariates, exact potential means.

    Source: ``data/static/news.csv`` (extracted from the fredjo.com
    ``NEWS_csv.zip``).  Each replication corresponds to one released random
    train/test split; the ``.y`` file stores
    ``t, yf, ycf, mu0, mu1``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        replications: int = 100,
        seed: int = 0,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing News file {self.path}. Run "
                "`uv run python scripts/download_datasets.py`."
            )
        self.replications = replications
        self.seed = seed

    def _one(self, index: int) -> IHDPReplication:
        base = self.path if self.path.is_dir() else self.path.parent
        seed = index + 1
        x_path = base / f"topic_doc_mean_n5000_k3477_seed_{seed}.csv.x"
        y_path = base / f"topic_doc_mean_n5000_k3477_seed_{seed}.csv.y"
        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Missing News replication files {x_path.name}/{y_path.name}. "
                "Re-run scripts/download_datasets.py."
            )
        x = _read_sparse_x(x_path)
        y = np.loadtxt(y_path, delimiter=",")
        if y.ndim != 2 or y.shape[1] < 5:
            raise ValueError(f"Unexpected News outcome file shape {y.shape}.")
        t = y[:, 0].astype(np.int64)
        yf = y[:, 1].astype(np.float32)
        ycf = y[:, 2].astype(np.float32)
        mu0 = y[:, 3].astype(np.float32)
        mu1 = y[:, 4].astype(np.float32)
        return IHDPReplication(x=x, t=t, yf=yf, ycf=ycf, mu0=mu0, mu1=mu1)

    def replication(self, index: int) -> IHDPReplication:
        """Load a single News replication (the 5,000 x 3,477 matrix)."""
        return self._one(index)

    def load(self) -> list[IHDPReplication]:
        return [self._one(index) for index in range(self.replications)]


def load_jobs(train_path: str | Path, test_path: str | Path | None = None) -> EffectCohort:
    """Load the canonical Jobs/LaLonde binary benchmark as a real cohort.

    The released ``jobs_DW_bin.new.10.{train,test}.npz`` files contain ten
    replications of real covariates with a binary factual outcome, a randomised
    experimental indicator ``e`` and the experimental ATE.  Individual
    counterfactuals are not released, so only the marginal effect is used.
    """
    train_path = Path(train_path)
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing Jobs file {train_path}. Run scripts/download_datasets.py."
        )
    data = np.load(train_path)
    x = np.asarray(data["x"])[..., 0]
    t = np.asarray(data["t"])[..., 0].astype(np.int64)
    y = np.asarray(data["yf"])[..., 0].astype(np.float64)
    reference_ate = float(np.asarray(data["ate"]).reshape(-1)[0])
    return EffectCohort(
        name="jobs_rct",
        x=x.astype(np.float64),
        t=t,
        y=y,
        reference_ate=reference_ate,
        reference="experimental ATE released with the Jobs benchmark",
        kind="rct",
    )


class CPSLoader:
    """Manual loader for the CFRNet CPS benchmark (not publicly mirrored).

    Place the authors' ``cps.csv`` under ``data/static/cps.csv`` with one row
    per unit and columns ``t, yf, ycf, mu0, mu1`` plus the covariates.  The
    loader validates the required columns and returns one
    :class:`IHDPReplication`; replications are bootstrap/split draws with a
    fixed seed.  No official public mirror of the exact CFRNet file is known,
    so the file must be requested from the original authors.
    """

    def __init__(self, path: str | Path, *, replications: int = 100, seed: int = 0) -> None:
        self.path = Path(path)
        self.replications = replications
        self.seed = seed
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing CPS file {self.path}. The CFRNet authors do not publish "
                "a public mirror; request cps.csv and place it under data/static/ "
                "with columns t, yf, ycf, mu0, mu1 plus covariates."
            )

    def replication(self, index: int) -> IHDPReplication:
        """Return one train/test concatenated CPS replication."""
        return self.load()[index]

    def load(self) -> list[IHDPReplication]:
        frame = pd.read_csv(self.path)
        required = {"t", "yf", "ycf", "mu0", "mu1"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CPS file is missing columns {sorted(missing)}.")
        covariates = [
            column for column in frame.columns if column not in required
        ]
        x = frame[covariates].to_numpy(dtype=np.float32)
        base = IHDPReplication(
            x=x,
            t=frame["t"].to_numpy(dtype=np.int64),
            yf=frame["yf"].to_numpy(dtype=np.float32),
            ycf=frame["ycf"].to_numpy(dtype=np.float32),
            mu0=frame["mu0"].to_numpy(dtype=np.float32),
            mu1=frame["mu1"].to_numpy(dtype=np.float32),
        )
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        output: list[IHDPReplication] = []
        for _ in range(self.replications):
            order = rng.permutation(n)
            cut = int(round(0.7 * n))
            for index in (order[:cut], order[cut:]):
                output.append(
                    IHDPReplication(
                        x=base.x[index],
                        t=base.t[index],
                        yf=base.yf[index],
                        ycf=base.ycf[index],
                        mu0=base.mu0[index],
                        mu1=base.mu1[index],
                    )
                )
        return [
            IHDPReplication(
                x=np.concatenate([output[2 * i].x, output[2 * i + 1].x]),
                t=np.concatenate([output[2 * i].t, output[2 * i + 1].t]),
                yf=np.concatenate([output[2 * i].yf, output[2 * i + 1].yf]),
                ycf=np.concatenate([output[2 * i].ycf, output[2 * i + 1].ycf]),
                mu0=np.concatenate([output[2 * i].mu0, output[2 * i + 1].mu0]),
                mu1=np.concatenate([output[2 * i].mu1, output[2 * i + 1].mu1]),
            )
            for i in range(self.replications)
        ]
