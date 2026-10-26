r"""Semi-synthetic effect benchmarks with real covariates or validated simulators.

Three sources, all exposing ground-truth counterfactuals (rule R1):

* :class:`TumourGrowthSimulator` -- the pharmacokinetic tumour-growth model of
  Bica et al. (2020, ``bicaEstimatingCounterfactualTreatment2020``), ported from
  the reference implementation in
  ``github.com/ioanabica/Counterfactual-Recurrent-Network``
  (``utils/cancer_simulation.py``). Temporal, time-varying binary chemo/radio
  treatments, exact paired counterfactuals under shared exogenous noise.
* :class:`IHDPLoader` -- Hill's semi-synthetic IHDP benchmark with real
  covariates and simulated potential outcomes, 100 replications
  (``ihdp_npci_1-100.{train,test}.npz``). Static, binary treatment.
* :class:`ACICLoader` -- the ACIC 2016/2018 challenge data if downloaded by
  hand (the competition requires registration); static, binary treatment.

Notation follows Section 2 of the review: ``X_t`` observed state, ``A_t``
treatment, ``Y(a)`` potential outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["TumourGrowthSimulator", "IHDPLoader", "IHDPGenerator", "ACICLoader"]


def _sphere_volume(diameter: float | Tensor) -> Tensor | float:
    """Volume of a sphere given its diameter (tumour cells are assumed spherical)."""
    return 4.0 / 3.0 * float(np.pi) * (diameter / 2.0) ** 3.0


_TUMOUR_CELL_DENSITY = 5.8e8
_TUMOUR_DEATH_THRESHOLD = cast(float, _sphere_volume(13.0))

# Tumour stage lognormal parameters (mu, sigma, lower, upper) on the diameter.
_TUMOUR_SIZE_DISTRIBUTIONS: dict[str, tuple[float, float, float, float]] = {
    "I": (1.72, 4.70, 0.3, 5.0),
    "II": (1.96, 1.63, 0.3, 13.0),
    "IIIA": (1.91, 9.40, 0.3, 13.0),
    "IIIB": (2.76, 6.87, 0.3, 13.0),
    "IV": (3.86, 8.82, 0.3, 13.0),
}
_STAGE_OBSERVATIONS: dict[str, int] = {
    "I": 1432,
    "II": 128,
    "IIIA": 1306,
    "IIIB": 7248,
    "IV": 12840,
}


def _truncated_normal(
    size: tuple[int, ...],
    *,
    mean: float,
    std: float,
    low: float,
    high: float,
    generator: torch.Generator,
) -> Tensor:
    """Inverse-CDF sampler for ``N(mean, std)`` truncated to ``[low, high]``."""
    root2 = float(np.sqrt(2.0))
    lower = (low - mean) / std
    upper = (high - mean) / std
    cdf_low = 0.5 * (1.0 + torch.erf(torch.tensor(lower / root2)))
    cdf_high = 0.5 * (1.0 + torch.erf(torch.tensor(upper / root2)))
    uniform = torch.rand(size, generator=generator) * (cdf_high - cdf_low) + cdf_low
    z = root2 * torch.erfinv(2.0 * uniform - 1.0)
    return mean + std * z


class TumourGrowthSimulator(BaseDataGenerator):
    r"""Tumour-growth simulator with time-varying chemo/radio treatments.

    The state is ``X_t = (V_t, d_t, r_t)``: tumour volume, effective
    chemotherapy dosage and radiotherapy dosage. Following the reference
    implementation, the volume evolves as

    .. math::
        V_{t+1} = V_t\Big(1 + \rho \log\frac{K}{V_t}
                  - \beta_c d_t - \alpha r_t - \beta r_t^2 + \epsilon_t\Big),
        \quad \epsilon_t \sim \mathcal N(0, \sigma^2),

    the chemo dosage decays with a one-day half-life
    ``d_t = d_{t-1} e^{-\log 2} + 5 A^{chemo}_t`` and radiotherapy contributes
    ``r_t = 2 A^{radio}_t``. Treatment assignment is confounded: each treatment
    is applied with a sigmoid probability that grows with the running mean
    tumour diameter (coefficients ``chemo_coeff`` / ``radio_coeff`` control the
    strength of confounding, exactly as in Bica et al., 2020).

    Counterfactuals share the patient parameters, initial state, process noise
    and action draws with the factual trajectory and only replace the treatment
    mechanism (``do(A_{1:T}=0)`` by default), so the Abduction--Action--Prediction
    steps are exact.
    """

    def __init__(
        self,
        *,
        horizon: int = 15,
        chemo_coeff: float = 5.0,
        radio_coeff: float = 5.0,
        window_size: int = 5,
        process_noise: float = 0.01,
        chemo_dose: float = 5.0,
        radio_dose: float = 2.0,
        clamp_volume: bool = True,
        standardize: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(observed_dim=3, horizon=horizon)
        self.chemo_coeff = chemo_coeff
        self.radio_coeff = radio_coeff
        self.window_size = window_size
        self.process_noise = process_noise
        self.chemo_dose = chemo_dose
        self.radio_dose = radio_dose
        self.clamp_volume = clamp_volume
        self.standardize = standardize
        self.outcome_dim = 0
        self.rng = torch.Generator().manual_seed(seed or 0)
        self.d_max = 13.0
        self._stats: tuple[Tensor, Tensor] | None = None

    def _normalisation(self, horizon: int) -> tuple[Tensor, Tensor]:
        """Population per-dimension mean/std of the factual process."""
        if self._stats is None:
            parameters = self._patient_parameters(2048)
            noise = self.process_noise * torch.randn(2048, horizon, generator=self.rng)
            actions = self._policy_actions(parameters, noise, horizon)
            trajectory = self._roll(parameters, actions, noise, horizon=horizon)
            mean = trajectory.mean(dim=(0, 1))
            std = trajectory.std(dim=(0, 1)) + 1e-6
            self._stats = (mean, std)
        return self._stats

    # -- patient population ---------------------------------------------------

    def _patient_parameters(self, n_samples: int) -> dict[str, Tensor]:
        """Sample the per-patient simulation parameters of Bica et al. (2020)."""
        rng = self.rng
        patient_types = torch.randint(1, 4, (n_samples,), generator=rng)

        stages = sorted(_TUMOUR_SIZE_DISTRIBUTIONS)
        counts = torch.tensor([_STAGE_OBSERVATIONS[s] for s in stages], dtype=torch.float)
        probabilities = counts / counts.sum()
        stage_index = torch.multinomial(probabilities, n_samples, replacement=True, generator=rng)

        diameters = torch.empty(n_samples)
        for index, stage in enumerate(stages):
            mask = stage_index == index
            count = int(mask.sum().item())
            if count == 0:
                continue
            mu, sigma, low, high = _TUMOUR_SIZE_DISTRIBUTIONS[stage]
            bounds = ((np.log(low) - mu) / sigma, (np.log(high) - mu) / sigma)
            z = _truncated_normal(
                (count,),
                mean=0.0,
                std=1.0,
                low=bounds[0],
                high=bounds[1],
                generator=rng,
            )
            diameters[mask] = torch.exp(z * sigma + mu)
        initial_volumes = cast(Tensor, _sphere_volume(diameters))

        # Correlated (alpha, rho) drawn from a truncated bivariate normal.
        alpha_mean, alpha_std = 0.0398, 0.168
        rho_mean, rho_std = 7e-5, 7.23e-3
        correlation = 0.87
        covariance = torch.tensor(
            [
                [alpha_std**2, correlation * alpha_std * rho_std],
                [correlation * alpha_std * rho_std, rho_std**2],
            ]
        )
        distribution = torch.distributions.MultivariateNormal(
            torch.tensor([alpha_mean, rho_mean]), covariance
        )
        accepted: list[Tensor] = []
        total = 0
        while total < n_samples:
            draw = distribution.sample((n_samples,))
            valid = draw[(draw[:, 0] > 0) & (draw[:, 1] > 0)]
            accepted.append(valid)
            total += valid.shape[0]
        params = torch.cat(accepted)[:n_samples]
        alpha = params[:, 0] + torch.where(patient_types <= 1, 0.1 * alpha_mean, 0.0)
        rho = params[:, 1]
        beta = alpha / 10.0

        # beta_c > 0: the truncation bound comes from 0.028 + 0.0007 z > 0.
        beta_c = 0.028 + 0.0007 * _truncated_normal(
            (n_samples,), mean=0.0, std=1.0, low=-40.0, high=40.0, generator=rng
        )
        beta_c = beta_c + torch.where(patient_types >= 3, 0.1, 0.0)
        carrying_capacity = torch.full((n_samples,), cast(float, _sphere_volume(30.0)))
        return {
            "patient_types": patient_types,
            "initial_volumes": initial_volumes.clamp(min=1e-6),
            "alpha": alpha,
            "rho": rho,
            "beta": beta,
            "beta_c": beta_c,
            "K": carrying_capacity,
        }

    # -- simulation -----------------------------------------------------------

    def _roll(
        self,
        parameters: dict[str, Tensor],
        actions: Tensor,
        process_noise: Tensor,
        *,
        horizon: int,
    ) -> Tensor:
        """Roll trajectories under ``actions`` with fixed process noise."""
        n_samples = actions.shape[0]
        volume = parameters["initial_volumes"].clone()
        chemo_dosage = torch.zeros(n_samples)
        radio_dosage = torch.zeros(n_samples)
        trajectories = []
        for t in range(horizon):
            chemo_application = actions[:, t, 0]
            radio_application = actions[:, t, 1]
            chemo_dosage = chemo_dosage * np.exp(-np.log(2.0)) + self.chemo_dose * chemo_application
            radio_dosage = self.radio_dose * radio_application
            growth = (
                parameters["rho"] * torch.log(parameters["K"] / volume.clamp(min=1e-6))
                - parameters["beta_c"] * chemo_dosage
                - parameters["alpha"] * radio_dosage
                - parameters["beta"] * radio_dosage**2
                + process_noise[:, t]
            )
            volume = volume * (1.0 + growth)
            if self.clamp_volume:
                volume = volume.clamp(min=1e-6, max=_TUMOUR_DEATH_THRESHOLD)
            trajectories.append(torch.stack([volume, chemo_dosage, radio_dosage], dim=-1))
        return torch.stack(trajectories, dim=1)

    def _policy_probabilities(
        self,
        volume_history: Tensor,
        parameters: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Treatment probabilities of the reference confounding policy."""
        window = volume_history[:, -self.window_size :]
        diameters = 2.0 * (window * 3.0 / (4.0 * float(np.pi))) ** (1.0 / 3.0)
        metric = diameters.mean(dim=-1)
        intercept = self.d_max / 2.0
        chemo_prob = torch.sigmoid((self.chemo_coeff / self.d_max) * (metric - intercept))
        radio_prob = torch.sigmoid((self.radio_coeff / self.d_max) * (metric - intercept))
        return chemo_prob, radio_prob

    def _policy_actions(
        self,
        parameters: dict[str, Tensor],
        process_noise: Tensor,
        horizon: int,
    ) -> Tensor:
        """Roll the confounded assignment policy jointly with the tumour state."""
        n_samples = parameters["initial_volumes"].shape[0]
        uniforms = torch.rand(n_samples, horizon, 2, generator=self.rng)
        actions = torch.zeros(n_samples, horizon, 2)
        state = parameters["initial_volumes"].clone()
        chemo_dosage = torch.zeros(n_samples)
        history = state.unsqueeze(-1)
        for t in range(horizon):
            chemo_prob, radio_prob = self._policy_probabilities(history, parameters)
            apply_chemo = (uniforms[:, t, 0] < chemo_prob).float()
            apply_radio = (uniforms[:, t, 1] < radio_prob).float()
            actions[:, t, 0] = apply_chemo
            actions[:, t, 1] = apply_radio
            chemo_dosage = chemo_dosage * np.exp(-np.log(2.0)) + self.chemo_dose * apply_chemo
            radio_dosage = self.radio_dose * apply_radio
            growth = (
                parameters["rho"] * torch.log(parameters["K"] / state.clamp(min=1e-6))
                - parameters["beta_c"] * chemo_dosage
                - parameters["alpha"] * radio_dosage
                - parameters["beta"] * radio_dosage**2
                + process_noise[:, t]
            )
            state = state * (1.0 + growth)
            if self.clamp_volume:
                state = state.clamp(min=1e-6, max=_TUMOUR_DEATH_THRESHOLD)
            history = torch.cat([history, state.unsqueeze(-1)], dim=1)
        return actions

    def factual_actions(
        self, n_samples: int, horizon: int, *, seed: int | None = None
    ) -> Tensor:
        """Sample factual treatment paths from the confounded assignment policy."""
        rng = torch.Generator().manual_seed(seed) if seed is not None else self.rng
        saved, self.rng = self.rng, rng
        try:
            parameters = self._patient_parameters(n_samples)
            process_noise = self.process_noise * torch.randn(
                n_samples, horizon, generator=self.rng
            )
            return self._policy_actions(parameters, process_noise, horizon)
        finally:
            self.rng = saved

    def sample(
        self,
        n_samples: int,
        horizon: int | None = None,
        *,
        factual_action: Tensor | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Return factual trajectories and exact paired counterfactuals (no treatment).

        ``factual_action`` may override the policy with a ``(N, T, 2)`` tensor of
        binary chemo/radio applications; the counterfactual always treats with
        ``do(A_{1:T}=0)``.
        """
        horizon = horizon or self.horizon
        parameters = self._patient_parameters(n_samples)
        process_noise = self.process_noise * torch.randn(n_samples, horizon, generator=self.rng)
        if factual_action is None:
            actions = self._policy_actions(parameters, process_noise, horizon)
        else:
            actions = factual_action.float().reshape(n_samples, horizon, 2)
        counterfactual_actions = torch.zeros_like(actions)
        observed = self._roll(parameters, actions, process_noise, horizon=horizon)
        counterfactual = self._roll(
            parameters, counterfactual_actions, process_noise, horizon=horizon
        )
        if self.standardize:
            mean, std = self._normalisation(horizon)
            observed = (observed - mean) / std
            counterfactual = (counterfactual - mean) / std
        return Batch(
            x=observed,
            context={"A": actions},
            counterfactual=counterfactual,
            outcome=observed[:, -1, self.outcome_dim],
        )


@dataclass(frozen=True)
class IHDPReplication:
    """One replication of the IHDP semi-synthetic benchmark.

    All arrays are NumPy, with one row per unit: ``x`` covariates
    ``(N, 25)``, ``t`` binary treatment ``(N,)``, ``yf`` factual outcome,
    ``ycf`` counterfactual outcome, ``mu0``/``mu1`` noiseless potential means.
    """

    x: np.ndarray
    t: np.ndarray
    yf: np.ndarray
    ycf: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray

    def as_tensors(self) -> dict[str, Tensor]:
        return {
            "x": torch.from_numpy(self.x).float(),
            "t": torch.from_numpy(self.t).long(),
            "yf": torch.from_numpy(self.yf).float(),
            "ycf": torch.from_numpy(self.ycf).float(),
            "mu0": torch.from_numpy(self.mu0).float(),
            "mu1": torch.from_numpy(self.mu1).float(),
        }


class IHDPLoader:
    """Loader for the 100-replication IHDP benchmark of Hill (2011).

    Expects the canonical ``ihdp_npci_1-100.{train,test}.npz`` archives
    (``github.com/AMLab-Amsterdam/CEVAE`` / ``vdorie/npci`` mirrors; the
    download script fetches them from ``fredjo.com``). Each archive stores
    matrices ``(N, R)`` for ``R = 100`` replications.
    """

    def __init__(self, train_path: str | Path, test_path: str | Path | None = None) -> None:
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path is not None else None
        self.train = self._load(self.train_path)
        self.test = self._load(self.test_path) if self.test_path else None
        self.n_replications = int(self.train["t"].shape[1])
        self.n_covariates = int(self.train["x"].shape[1])

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        arrays = np.load(path)
        return {key: np.asarray(arrays[key]) for key in arrays.files}

    def replication(self, index: int, *, split: str = "train") -> IHDPReplication:
        """Return replication ``index`` of the requested split."""
        if not 0 <= index < self.n_replications:
            raise IndexError(f"Replication {index} out of range.")
        data = self.train if split == "train" else self.test
        if data is None:
            raise ValueError("No test split was provided to the loader.")
        x = data["x"][..., index]
        if x.ndim != 2:  # some mirrors transpose covariate blocks
            x = x.reshape(x.shape[0], -1)
        return IHDPReplication(
            x=x.astype(np.float32),
            t=data["t"][..., index].astype(np.int64),
            yf=data["yf"][..., index].astype(np.float32),
            ycf=data["ycf"][..., index].astype(np.float32),
            mu0=data["mu0"][..., index].astype(np.float32),
            mu1=data["mu1"][..., index].astype(np.float32),
        )

    def pehe(self, index: int, tau_hat: np.ndarray, *, split: str = "train") -> float:
        """PEHE of an estimated individual effect ``tau_hat`` on one replication."""
        item = self.replication(index, split=split)
        tau = item.mu1 - item.mu0
        return float(np.sqrt(np.mean((np.asarray(tau_hat) - tau) ** 2)))

    def ate_error(self, index: int, tau_hat: np.ndarray, *, split: str = "train") -> float:
        """Absolute error of the estimated average treatment effect."""
        item = self.replication(index, split=split)
        tau = item.mu1 - item.mu0
        return float(abs(np.mean(tau_hat) - np.mean(tau)))


class IHDPGenerator(BaseDataGenerator):
    r"""Batched single-step generator over one IHDP replication.

    The observed state is ``[x_1..x_p, y]``; sampling units with replacement
    yields factual batches and their paired counterfactuals (last coordinate
    replaced by the unobserved potential outcome). Covariates and outcome are
    standardised with the replication statistics, and
    :meth:`rescale_outcome` maps model outputs back to the original units so
    PEHE stays comparable across replications.

    The treatment is exposed per unit in ``batch.context['A']`` of shape
    ``(n, 1, 1)``; ``batch.counterfactual`` holds ``[x, y(1-a)]`` and is used
    for evaluation only, never by the training losses.
    """

    def __init__(
        self,
        replication: IHDPReplication,
        *,
        seed: int | None = None,
        standardize: bool = True,
        reference: IHDPGenerator | None = None,
    ) -> None:
        covariates = replication.x.astype(np.float32)
        super().__init__(observed_dim=covariates.shape[1] + 1, horizon=1)
        self.replication = replication
        self.standardize = standardize
        self.rng = torch.Generator().manual_seed(seed or 0)
        x = torch.from_numpy(covariates).float()
        yf = torch.from_numpy(replication.yf).float()
        ycf = torch.from_numpy(replication.ycf).float()
        self.x_mean: Tensor
        self.x_std: Tensor
        self.y_mean: Tensor
        self.y_std: Tensor
        if reference is not None:
            # Use the training split's standardisation for the held-out split,
            # so an estimated outcome can be mapped back with the same scale.
            self.x_mean, self.x_std = reference.x_mean, reference.x_std
            self.y_mean, self.y_std = reference.y_mean, reference.y_std
            x = (x - self.x_mean) / self.x_std
            yf = (yf - self.y_mean) / self.y_std
            ycf = (ycf - self.y_mean) / self.y_std
        elif standardize:
            self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-6
            self.y_mean, self.y_std = yf.mean(), yf.std() + 1e-6
            x = (x - self.x_mean) / self.x_std
            yf = (yf - self.y_mean) / self.y_std
            ycf = (ycf - self.y_mean) / self.y_std
        else:
            self.x_mean, self.x_std = torch.zeros(x.shape[1]), torch.ones(x.shape[1])
            self.y_mean, self.y_std = torch.zeros(()), torch.ones(())
        self.covariates = x
        self.outcomes = yf
        self.counterfactual_outcomes = ycf
        self.treatment = torch.from_numpy(replication.t).float()

    def _batch(self, index: Tensor) -> Batch:
        covariates = self.covariates[index]
        factual = self.outcomes[index]
        counterfactual = self.counterfactual_outcomes[index]
        x = torch.cat([covariates, factual.unsqueeze(-1)], dim=-1).unsqueeze(1)
        x_cf = torch.cat([covariates, counterfactual.unsqueeze(-1)], dim=-1).unsqueeze(1)
        action = self.treatment[index].reshape(-1, 1, 1)
        return Batch(
            x=x,
            context={"A": action},
            counterfactual=x_cf,
            outcome=factual,
        )

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        index = torch.randint(0, self.covariates.shape[0], (n_samples,), generator=self.rng)
        return self._batch(index)

    def sample_all(self) -> Batch:
        """Return every unit once, in order (canonical evaluation split)."""
        return self._batch(torch.arange(self.covariates.shape[0]))

    def rescale_outcome(self, outcome: Tensor) -> Tensor:
        """Map a standardised outcome estimate back to the original units."""
        return outcome * self.y_std.to(outcome.device) + self.y_mean.to(outcome.device)

    @property
    def outcome_dim(self) -> int:
        """Index of the outcome coordinate in the observed vector."""
        return self.observed_dim - 1


class ACICLoader:
    """Loader for hand-downloaded ACIC 2016/2018 challenge data.

    The competition data (``github.com/vdorie/aciccomp``) is distributed in R
    archives and requires registration, so it is not fetched automatically. The
    loader accepts a directory containing ``x.csv``, ``z.csv`` (treatment) and
    ``y.csv`` (factual outcome), plus ``mu0.csv``/``mu1.csv`` when a replication
    provides the ground-truth response surfaces, or a single ``.npz`` archive
    with the same keys.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(
                f"ACIC directory {self.directory} does not exist. Download the "
                "competition data by hand (see scripts/download_datasets.py) and "
                "place x.csv, z.csv, y.csv, mu0.csv and mu1.csv there."
            )

    def replication(self, name: str = "default") -> dict[str, np.ndarray]:
        """Load one replication directory (``<name>/``) or the root if flat."""
        import pandas as pd

        base = self.directory / name if (self.directory / name).is_dir() else self.directory
        result = {
            "x": pd.read_csv(base / "x.csv", header=None).to_numpy(dtype=np.float32),
            "t": pd.read_csv(base / "z.csv", header=None).to_numpy(dtype=np.int64).reshape(-1),
            "yf": pd.read_csv(base / "y.csv", header=None).to_numpy(dtype=np.float32).reshape(-1),
        }
        for key, filename in (("mu0", "mu0.csv"), ("mu1", "mu1.csv")):
            path = base / filename
            if path.exists():
                result[key] = (
                    pd.read_csv(path, header=None).to_numpy(dtype=np.float32).reshape(-1)
                )
        # ACIC outcomes are on a heavy-tailed, replication-specific scale (the
        # raw PEHE of the earlier run was 610 +/- 4654).  Standardise the
        # factual outcome, the realised potential outcomes and both response
        # surfaces with the *factual* mean/std; the effect is then reported in
        # standard-deviation units and is comparable across replications.
        factual = result["yf"]
        scale = float(np.std(factual))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        center = float(np.mean(factual))
        result["yf"] = (result["yf"] - center) / scale
        for key in ("mu0", "mu1"):
            if key in result:
                result[key] = (result[key] - center) / scale
        result["outcome_mean"] = np.asarray([center], dtype=np.float32)
        result["outcome_scale"] = np.asarray([scale], dtype=np.float32)
        return result

    def pehe(self, tau_hat: np.ndarray, name: str = "default") -> float:
        """PEHE on a replication that provides ``mu0``/``mu1``."""
        item = self.replication(name)
        if "mu0" not in item or "mu1" not in item:
            raise ValueError("This ACIC replication provides no ground-truth response surfaces.")
        tau = item["mu1"] - item["mu0"]
        return float(np.sqrt(np.mean((np.asarray(tau_hat) - tau) ** 2)))
