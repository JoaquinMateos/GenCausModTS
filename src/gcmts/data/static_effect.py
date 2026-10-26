r"""Additional *synthetic* static CATE panels (Twins/Jobs/News/CPS dimensions).

The canonical static benchmarks beyond IHDP/ACIC are the semi-synthetic CATE
datasets popularised by Louizos et al.\ (2017) and Shalit et al.\ (2017):
**Twins**, **Jobs** (LaLonde), **News** and **CPS**.  This module does **not**
bundle their real covariates.  It instantiates synthetic Gaussian replacements
calibrated to the source's covariate dimension and treated fraction, with a
known linear-plus-nonlinear outcome surface and exact potential outcomes.  They
are therefore *calibration panels*, not the canonical benchmarks: the results
must be labelled ``Synth-Twins`` etc., and no claim about the real benchmarks
may be based on them.  The canonical Twins loader (both twin outcomes observed)
and the CFRNet semi-synthetic files are a planned addition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcmts.data.semi_synthetic import IHDPReplication

__all__ = ["StaticCateLoader", "STATIC_CATE_BENCHMARKS"]


@dataclass(frozen=True)
class _Config:
    n_covariates: int
    treated_fraction: float
    nonlinearity: float
    noise: float


# Calibrated to the source benchmarks: Twins (~40 covariates, balanced),
# Jobs/LaLonde (~8 covariates, ~10% treated), News (~high-dimensional, balanced),
# CPS (~10 covariates, ~30% treated).
_CONFIGS: dict[str, _Config] = {
    "twins": _Config(n_covariates=40, treated_fraction=0.5, nonlinearity=0.5, noise=0.1),
    "jobs": _Config(n_covariates=8, treated_fraction=0.1, nonlinearity=0.2, noise=0.2),
    "news": _Config(n_covariates=20, treated_fraction=0.5, nonlinearity=0.8, noise=0.2),
    "cps": _Config(n_covariates=10, treated_fraction=0.3, nonlinearity=0.6, noise=0.2),
}

STATIC_CATE_BENCHMARKS = tuple(_CONFIGS)


class StaticCateLoader:
    """Semi-synthetic static CATE benchmark with known potential outcomes.

    Args:
        benchmark: one of ``twins``, ``jobs``, ``news``, ``cps``.
        replications: number of independent replications (each a benchmark
            instance for the paired tests).
        n_units: units per replication.
        seed: RNG seed.
    """

    def __init__(
        self,
        benchmark: str,
        *,
        replications: int = 100,
        n_units: int = 1000,
        seed: int = 0,
    ) -> None:
        if benchmark not in _CONFIGS:
            raise ValueError(f"Unknown static benchmark '{benchmark}'.")
        self.benchmark = benchmark
        self.config = _CONFIGS[benchmark]
        self.replications = replications
        self.n_units = n_units
        self.seed = seed

    def _one(self, rng: np.random.Generator) -> IHDPReplication:
        config = self.config
        p = config.n_covariates
        x = rng.normal(size=(self.n_units, p)).astype(np.float64)
        # Correlated, non-linear covariate structure.
        x = x + config.nonlinearity * np.tanh(x[:, :1]) * 0.5
        beta = rng.normal(size=p) / np.sqrt(p)
        propensity = np.clip(
            config.treated_fraction
            + config.nonlinearity * 0.1 * np.tanh(x @ beta),
            0.02,
            0.98,
        )
        t = (rng.random(self.n_units) < propensity).astype(np.float64)
        gamma = rng.normal(size=p) / np.sqrt(p)
        delta = rng.normal(size=p) / np.sqrt(p)
        mu0 = x @ gamma + config.nonlinearity * np.sin(x @ delta)
        mu1 = mu0 + 1.0 + config.nonlinearity * np.tanh(x @ gamma)
        y0 = mu0 + config.noise * rng.normal(size=self.n_units)
        y1 = mu1 + config.noise * rng.normal(size=self.n_units)
        yf = np.where(t > 0.5, y1, y0)
        ycf = np.where(t > 0.5, y0, y1)
        return IHDPReplication(x=x, t=t, yf=yf, ycf=ycf, mu0=mu0, mu1=mu1)

    def load(self) -> list[IHDPReplication]:
        """Return ``replications`` independent semi-synthetic replications."""
        rng = np.random.default_rng(self.seed)
        return [self._one(rng) for _ in range(self.replications)]
