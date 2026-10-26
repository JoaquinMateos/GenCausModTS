"""CaTSG: causality-guided conditional diffusion for time series.

Reference: Xia et al., *Causal Time Series Generation via Diffusion Models*,
2025 — BibTeX ``yutongxiaCausalTimeSeries2025``.

The audited implementation adds the two components that the earlier conditional
diffusion surrogate omitted: a learnable **environment bank** with an
``EnvInfer`` module that infers a posterior over latent environments from
``(x, c)``, and the **backdoor-adjusted guidance** of Eq. (6) of the paper,

    eps_guided = (1 + omega) * sum_k w_k eps_theta(x_t, t, c, e_k)
                 - omega * eps_theta(x_t, t),

where ``w = p(e | x, c)`` is the inferred environment posterior.  The denoiser
is trained with the guided score; counterfactual inference abducts the
environment posterior from the factual trajectory, replaces the treatment, and
samples with the same posterior (abduction--action--prediction).
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
    half = (dim + 1) // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=positions.device) / max(half, 1)
    )
    angles = positions.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return embedding[..., :dim]


class CaTSG(BaseEffectEstimator):
    """Environment-guided conditional diffusion estimator.

    Args:
        observed_dim: dimension ``D`` of the trajectory.
        treatment_dim: treatment dimension.
        horizon: trajectory length the denoiser operates on.
        n_steps: number of diffusion steps.
        n_environments: size ``K`` of the latent environment bank.
        environment_dim: dimension of each environment embedding.
        guidance: guidance strength ``omega`` in Eq. (6).
        hidden_dims: denoiser widths.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        horizon: int = 24,
        n_steps: int = 20,
        n_environments: int = 4,
        environment_dim: int = 8,
        guidance: float = 0.5,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        self.horizon = horizon
        self.n_steps = n_steps
        self.dim = horizon * observed_dim
        self.n_environments = n_environments
        self.environment_dim = environment_dim
        self.guidance = guidance
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.alpha_bar: Tensor
        self.time_dim = 32
        self.cond_dim = horizon * treatment_dim
        self.denoiser = MLP(
            self.dim + self.time_dim + self.cond_dim + environment_dim,
            self.dim,
            hidden_dims=hidden_dims,
        )
        self.environment_encoder = MLP(
            self.dim + self.cond_dim, n_environments, hidden_dims=hidden_dims
        )
        self.environment_bank = nn.Parameter(
            torch.randn(n_environments, environment_dim) * 0.1
        )

    def _actions(self, batch: Batch) -> Tensor:
        x = batch.x
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(x.shape[0], self.horizon, self.treatment_dim, device=x.device)
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, self.horizon, -1)
        return action.reshape(x.shape[0], -1).float().to(x.device)

    def _env_weights(self, flat_x: Tensor, action: Tensor) -> Tensor:
        logits = torch.as_tensor(self.environment_encoder(torch.cat([flat_x, action], dim=-1)))
        return torch.softmax(logits, dim=-1)

    def _eps(
        self,
        flat: Tensor,
        step: int | Tensor,
        action: Tensor,
        env_weights: Tensor | None,
    ) -> Tensor:
        batch = flat.shape[0]
        positions = (
            torch.full((batch,), step, device=flat.device)
            if isinstance(step, int)
            else step
        )
        time_embedding = _sinusoidal(positions, self.time_dim)
        if env_weights is None:
            env = torch.zeros(batch, self.environment_dim, device=flat.device)
            return cast(
                Tensor,
                self.denoiser(torch.cat([flat, time_embedding, action, env], dim=-1)),
            )
        per_environment = []
        for index in range(self.n_environments):
            env = self.environment_bank[index].unsqueeze(0).expand(batch, -1)
            per_environment.append(
                self.denoiser(torch.cat([flat, time_embedding, action, env], dim=-1))
            )
        conditioned = torch.stack(per_environment, dim=1)
        env_prediction = (env_weights.unsqueeze(-1) * conditioned).sum(dim=1)
        base = self.denoiser(
            torch.cat(
                [
                    flat,
                    time_embedding,
                    action,
                    torch.zeros(batch, self.environment_dim, device=flat.device),
                ],
                dim=-1,
            )
        )
        return cast(
            Tensor,
            (1.0 + self.guidance) * env_prediction - self.guidance * base,
        )

    def _ddim_sample(self, code: Tensor, action: Tensor, env_weights: Tensor) -> Tensor:
        flat = code
        for step in reversed(range(self.n_steps)):
            abar = self.alpha_bar[step]
            abar_prev = self.alpha_bar[step - 1] if step > 0 else torch.ones_like(abar)
            eps = self._eps(flat, step, action, env_weights)
            x0 = (flat - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
            flat = torch.sqrt(abar_prev) * x0 + torch.sqrt(1.0 - abar_prev) * eps
        return flat.reshape(-1, self.horizon, self.observed_dim)

    def _ddim_invert(self, x: Tensor, action: Tensor, env_weights: Tensor) -> Tensor:
        flat = x.reshape(x.shape[0], -1)
        for step in range(self.n_steps):
            abar = self.alpha_bar[step]
            abar_next = (
                self.alpha_bar[step + 1] if step + 1 < self.n_steps else torch.ones_like(abar)
            )
            eps = self._eps(flat, step, action, env_weights)
            x0 = (flat - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
            flat = torch.sqrt(abar_next) * x0 + torch.sqrt(1.0 - abar_next) * eps
        return flat

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        flat = batch.x.reshape(batch.x.shape[0], -1)
        env_weights = self._env_weights(flat, action)
        return {
            "code": self._ddim_invert(batch.x, action, env_weights),
            "action": action,
            "env_weights": env_weights,
        }

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, self.horizon, -1)
        flat = new_action.reshape(new_action.shape[0], -1).float()
        return {
            "code": noise["code"],
            "action": flat,
            "env_weights": noise["env_weights"],
        }

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        return self._ddim_sample(
            context["code"], context["action"], context["env_weights"]
        )[:, :horizon]

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        context = self.action(self.abduction(batch), counterfactual_action)
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=self.prediction(context, batch.x.shape[1]),
            noise={"code": context["code"], "env_weights": context["env_weights"]},
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        if batch.x.shape[1] != self.horizon:
            raise ValueError(
                f"CaTSG was built for horizon={self.horizon} but received T={batch.x.shape[1]}."
            )
        action = self._actions(batch)
        flat = batch.x.reshape(batch.x.shape[0], -1)
        env_weights = self._env_weights(flat, action)
        step = torch.randint(0, self.n_steps, (flat.shape[0],), device=flat.device)
        abar = self.alpha_bar[step].unsqueeze(-1)
        noise = torch.randn_like(flat)
        noisy = torch.sqrt(abar) * flat + torch.sqrt(1.0 - abar) * noise
        eps_pred = self._eps(noisy, step, action, env_weights)
        # Environment-balancing term: keep the bank diverse and avoid a
        # collapsed posterior (the paper's environment-bank regulariser).
        usage = env_weights.mean(dim=0)
        balance = (usage * torch.log(usage + 1e-8)).sum()
        return {
            "eps": noise,
            "eps_pred": eps_pred,
            "action": action,
            "env_weights": env_weights,
            "balance": balance,
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        score = nn.functional.mse_loss(outputs["eps_pred"], outputs["eps"])
        total = score - 0.01 * outputs["balance"]
        return {
            "loss": total,
            "score": score,
            "balance": outputs["balance"],
        }

    def identifiability_statement(self) -> str:
        return (
            "CaTSG: under the SCM with latent environment E and positivity, "
            "backdoor adjustment identifies the interventional score and "
            "abduction--action--prediction identifies the counterfactual score, "
            "up to the diffusion discretisation and environment-posterior error."
        )
