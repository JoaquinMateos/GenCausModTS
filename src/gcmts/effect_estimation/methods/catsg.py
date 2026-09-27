"""CaTSG: score-guided conditional diffusion for counterfactual time series.

Reference: Xia et al., *Causal Time Series Generation via Diffusion Models*,
2025 — BibTeX ``yutongxiaCausalTimeSeries2025``.

A conditional denoising diffusion model is trained on observed trajectories.
Counterfactual inference follows the abduction--action--prediction recipe:
deterministic DDIM inversion abducts the factual trajectory into a noise code,
the treatment is replaced, and conditional DDIM sampling predicts the
counterfactual trajectory from that code.
"""

from __future__ import annotations

import logging
import math
from typing import cast

import torch
from torch import Tensor, nn

from gcmts.core.backbones import MLP
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["CaTSG"]


def _sinusoidal(positions: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device) / max(half, 1)
    )
    angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class CaTSG(BaseEffectEstimator):
    """Score-guided conditional diffusion estimator.

    Args:
        observed_dim: dimension ``D`` of the trajectory.
        treatment_dim: treatment dimension.
        horizon: trajectory length the denoiser operates on.
        n_steps: number of diffusion steps.
        hidden_dims: denoiser widths.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        horizon: int = 24,
        n_steps: int = 20,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        self.horizon = horizon
        self.n_steps = n_steps
        self.dim = horizon * observed_dim
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.alpha_bar: Tensor
        self.time_dim = 32
        self.denoiser = MLP(
            self.dim + self.time_dim + treatment_dim, self.dim, hidden_dims=hidden_dims
        )

    def _actions(self, batch: Batch) -> Tensor:
        x = batch.x
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(x.shape[0], self.treatment_dim, device=x.device)
        if action.ndim == 3:
            action = action.mean(dim=1)
        return action.float().to(x.device)

    def _eps(self, flat: Tensor, step: int, action: Tensor) -> Tensor:
        batch = flat.shape[0]
        t = torch.full((batch,), step, device=flat.device)
        inp = torch.cat([flat, _sinusoidal(t, self.time_dim), action], dim=-1)
        return cast(Tensor, self.denoiser(inp))

    def _ddim_sample(self, code: Tensor, action: Tensor) -> Tensor:
        flat = code
        for step in reversed(range(self.n_steps)):
            abar = self.alpha_bar[step]
            abar_prev = self.alpha_bar[step - 1] if step > 0 else torch.ones_like(abar)
            eps = self._eps(flat, step, action)
            x0 = (flat - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
            flat = torch.sqrt(abar_prev) * x0 + torch.sqrt(1.0 - abar_prev) * eps
        return flat.reshape(-1, self.horizon, self.observed_dim)

    def _ddim_invert(self, x: Tensor, action: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        for step in range(self.n_steps):
            abar = self.alpha_bar[step]
            abar_next = (
                self.alpha_bar[step + 1] if step + 1 < self.n_steps else torch.ones_like(abar)
            )
            eps = self._eps(flat, step, action)
            x0 = (flat - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
            flat = torch.sqrt(abar_next) * x0 + torch.sqrt(1.0 - abar_next) * eps
        return flat

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        return {"code": self._ddim_invert(batch.x, action), "action": action}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 3:
            new_action = new_action.mean(dim=1)
        return {"code": noise["code"], "action": new_action.float()}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        return self._ddim_sample(context["code"], context["action"])[:, :horizon]

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        action = intervention["A"]
        if action.ndim == 3:
            action = action.mean(dim=1)
        action = action.float().to(batch.x.device)
        code = torch.randn(batch.x.shape[0], self.dim, device=batch.x.device)
        return self._ddim_sample(code, action)

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        context = self.action(self.abduction(batch), counterfactual_action)
        counterfactual = self.prediction(context, batch.x.shape[1])
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=counterfactual,
            noise={"code": context["code"]},
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        flat = batch.x.reshape(batch.x.shape[0], -1)
        step = torch.randint(0, self.n_steps, (flat.shape[0],), device=flat.device)
        abar = self.alpha_bar[step].unsqueeze(-1)
        noise = torch.randn_like(flat)
        noisy = torch.sqrt(abar) * flat + torch.sqrt(1.0 - abar) * noise
        inp = torch.cat([noisy, _sinusoidal(step, self.time_dim), action], dim=-1)
        eps_pred = self.denoiser(inp)
        return {"eps": noise, "eps_pred": eps_pred, "action": action}

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        score = nn.functional.mse_loss(outputs["eps_pred"], outputs["eps"])
        return {"loss": score, "score": score}

    def identifiability_statement(self) -> str:
        return (
            "CaTSG: given a consistent conditional score and exact (deterministic) "
            "inversion for abduction, the counterfactual trajectory is identified "
            "up to the diffusion discretisation error."
        )
