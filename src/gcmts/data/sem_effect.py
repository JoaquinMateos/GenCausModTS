r"""Time-varying-treatment structural-equation benchmark for effect estimation.

A fully synthetic benchmark in the style of Wu et al.\ (2024): a covariate vector
evolves under lagged causal influences of past treatments and covariates, a binary
treatment is assigned by a logistic propensity that depends on the covariate
history (time-varying confounding), and the outcome is a linear function of the
covariate and treatment history. Because every noise term is shared between the
factual and counterfactual worlds, the counterfactual trajectory under
``do(A_{1:T}=0)`` is exact, which makes PEHE and the counterfactual metrics
well defined.

.. math::
    X_t = \gamma_0 + \sum_\tau \gamma^A_\tau A_{t-\tau}
          + \sum_\tau \gamma^X_\tau X_{t-\tau} + \varepsilon_t,\quad
    Y_t = \alpha_0 + \sum_\tau \alpha^A_\tau A_{t-\tau}
          + \sum_\tau \alpha^X_\tau X_{t-\tau} + \nu_t,
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["TimeVaryingSEMGenerator"]


class TimeVaryingSEMGenerator(BaseDataGenerator):
    """Synthetic time-varying-treatment SEM with time-varying confounding.

    Args:
        n_covariates: number ``h`` of covariates.
        horizon: sequence length ``T``.
        history: number of lagged steps entering each mechanism.
        noise_std: process-noise std.
        seed: RNG seed.
    """

    def __init__(
        self,
        *,
        n_covariates: int = 4,
        horizon: int = 20,
        history: int = 2,
        noise_std: float = 0.1,
        max_radius: float = 0.8,
        seed: int | None = None,
    ) -> None:
        observed_dim = 1 + n_covariates + 1  # [Y, X_1..X_h, A]
        super().__init__(observed_dim=observed_dim, horizon=horizon)
        self.n_covariates = n_covariates
        self.history = history
        self.noise_std = noise_std
        self.outcome_dim = 0
        self.rng = torch.Generator().manual_seed(seed or 0)
        h = n_covariates
        self.gamma_a = 0.4 * torch.randn(history, generator=self.rng)
        gamma_x = 0.3 * torch.randn(history, h, h, generator=self.rng)
        # A random VAR(h) covariate process is often explosive; rescale the
        # lag blocks to a stationary companion radius.  Without this, a few
        # seeds produce outcomes two orders of magnitude larger than the rest
        # and dominate every aggregated PEHE.
        companion = torch.zeros(history * h, history * h)
        for lag in range(history):
            companion[:h, lag * h : (lag + 1) * h] = gamma_x[lag]
        for lag in range(1, history):
            companion[lag * h : (lag + 1) * h, (lag - 1) * h : lag * h] = torch.eye(h)
        radius = float(torch.linalg.eigvals(companion).abs().max())
        if radius > max_radius:
            gamma_x = gamma_x * (max_radius / radius)
        self.gamma_x = gamma_x
        self.beta_a = 0.8 * torch.randn(history, generator=self.rng)
        self.beta_x = 0.6 * torch.randn(history, h, generator=self.rng)
        self.alpha_a = 0.7 * torch.randn(history, generator=self.rng)
        self.alpha_x = 0.5 * torch.randn(history, h, generator=self.rng)

    def _roll(self, actions: Tensor, noise_x: Tensor, noise_y: Tensor) -> tuple[Tensor, Tensor]:
        """Roll the SEM for a given action sequence ``(B, T)``; shared noise."""
        batch, horizon = actions.shape
        h = self.n_covariates
        cov = torch.zeros(batch, horizon, h)
        out = torch.zeros(batch, horizon)
        cov[:, 0] = torch.rand(batch, h, generator=self.rng)
        for t in range(horizon):
            for tau in range(1, self.history + 1):
                if t - tau < 0:
                    continue
                cov[:, t] = cov[:, t] + self.gamma_a[tau - 1] * actions[:, t - tau].unsqueeze(-1)
                cov[:, t] = cov[:, t] + cov[:, t - tau] @ self.gamma_x[tau - 1]
                out[:, t] = out[:, t] + self.alpha_a[tau - 1] * actions[:, t - tau]
                out[:, t] = out[:, t] + cov[:, t - tau] @ self.alpha_x[tau - 1]
            if t > 0:
                cov[:, t] = cov[:, t] + noise_x[:, t]
                out[:, t] = out[:, t] + noise_y[:, t]
        return cov, out

    def _propensity(self, actions: Tensor, cov: Tensor) -> Tensor:
        logits = torch.zeros(actions.shape[0], actions.shape[1])
        for tau in range(1, self.history + 1):
            shifted = torch.zeros_like(actions)
            shifted[:, tau:] = actions[:, :-tau]
            shifted_cov = torch.zeros_like(cov)
            shifted_cov[:, tau:] = cov[:, :-tau]
            logits = logits + self.beta_a[tau - 1] * shifted
            logits = logits + shifted_cov @ self.beta_x[tau - 1]
        return torch.sigmoid(logits)

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        horizon = horizon or self.horizon
        noise_x = self.noise_std * torch.randn(
            n_samples, horizon, self.n_covariates, generator=self.rng
        )
        noise_y = self.noise_std * torch.randn(n_samples, horizon, generator=self.rng)
        # Factual actions: sample from the propensity of a first pass.
        cov0, _ = self._roll(torch.zeros(n_samples, horizon), noise_x, noise_y)
        propensity = self._propensity(torch.zeros(n_samples, horizon), cov0)
        actions = (torch.rand(n_samples, horizon, generator=self.rng) < propensity).float()
        cov, out = self._roll(actions, noise_x, noise_y)

        zeros = torch.zeros_like(actions)
        cov_cf, out_cf = self._roll(zeros, noise_x, noise_y)

        # Standardise the factual observations dimension-wise and apply the
        # *same* transform to the counterfactual.  The raw linear SEM can
        # produce outcomes an order of magnitude larger than the covariates,
        # which made the neural encoders emit NaNs for some seeds; the
        # component-wise affine transform leaves the causal effect identified
        # and is the same normalisation used by the other semi-synthetic
        # benchmarks.
        out_mean = out.mean()
        out_scale = out.std() + 1e-6
        cov_mean = cov.mean(dim=(0, 1), keepdim=True)
        cov_scale = cov.std(dim=(0, 1), keepdim=True) + 1e-6
        out = (out - out_mean) / out_scale
        cov = (cov - cov_mean) / cov_scale
        out_cf = (out_cf - out_mean) / out_scale
        cov_cf = (cov_cf - cov_mean) / cov_scale

        x = torch.cat([out.unsqueeze(-1), cov, actions.unsqueeze(-1)], dim=-1)
        x_cf = torch.cat([out_cf.unsqueeze(-1), cov_cf, zeros.unsqueeze(-1)], dim=-1)
        return Batch(
            x=x,
            context={"A": actions.unsqueeze(-1)},
            counterfactual=x_cf,
            outcome=out[:, -1],
        )
