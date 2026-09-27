"""GANITE-style adversarial counterfactual estimator for time series.

Reference: Yoon et al., *GANITE: Estimation of Individualized Treatment
Effects*, NeurIPS 2018 — BibTeX ``yoonGANITEEstimationIndividualized2018``.

A generator produces (counterfactual) trajectories from an abducted noise
vector and a treatment, while a discriminator distinguishes generated from
factual trajectories. An inference network provides the abduction step so that
individual exogenous noise is preserved under the treatment replacement.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor, nn

from gcmts.core.backbones import MLP, GaussianHead, gaussian_reconstruction_nll
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["GANITE"]


class GANITE(BaseEffectEstimator):
    """Adversarial (GAN-enhanced) counterfactual estimator.

    Args:
        observed_dim: dimension ``D`` of the trajectory.
        treatment_dim: dimension of the treatment.
        noise_dim: dimension of the abducted exogenous noise.
        adversarial_weight: weight of the GAN term relative to reconstruction.
        hidden_dims: network widths.
        obs_noise: observation-noise std for the Gaussian NLL.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        noise_dim: int = 4,
        adversarial_weight: float = 0.1,
        hidden_dims: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.noise_dim = noise_dim
        self.adversarial_weight = adversarial_weight
        self.inference = GaussianHead(observed_dim, noise_dim, hidden_dims=hidden_dims)
        self.generator = MLP(noise_dim + treatment_dim, observed_dim, hidden_dims=hidden_dims)
        self.discriminator = MLP(observed_dim + treatment_dim, 1, hidden_dims=hidden_dims)
        self.register_buffer(
            "obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone()
        )
        self.obs_logvar: Tensor

    def _actions(self, batch: Batch) -> Tensor:
        x = batch.x
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(x.shape[0], x.shape[1], self.treatment_dim, device=x.device)
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, x.shape[1], -1)
        return action.to(x.device)

    def _generate(self, noise: Tensor, action: Tensor) -> Tensor:
        batch, time, _ = noise.shape
        inp = torch.cat([noise, action], dim=-1).reshape(batch * time, -1)
        return cast(Tensor, self.generator(inp)).reshape(batch, time, self.observed_dim)

    def _discriminate(self, x: Tensor, action: Tensor) -> Tensor:
        batch, time, _ = x.shape
        inp = torch.cat([x, action], dim=-1).reshape(batch * time, -1)
        logits = cast(Tensor, self.discriminator(inp)).reshape(batch, time)
        return logits.mean(dim=1)

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self.inference(batch.x.reshape(-1, self.observed_dim))
        noise = mean.reshape(batch.x.shape[0], batch.x.shape[1], -1)
        return {"noise": noise, "logvar": logvar, "action": action}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, noise["noise"].shape[1], -1)
        return {"noise": noise["noise"], "action": new_action.float()}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        return self._generate(context["noise"][:, :horizon], context["action"][:, :horizon])

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=self.intervene(batch, counterfactual_action),
            noise=self.abduction(batch),
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self.inference(batch.x.reshape(-1, self.observed_dim))
        noise = mean.reshape(batch.x.shape[0], batch.x.shape[1], -1)
        fake = self._generate(noise, action)
        return {
            "noise": noise,
            "logvar": logvar,
            "recon": fake,
            "action": action,
            "real_logits": self._discriminate(batch.x, action),
            "fake_logits": self._discriminate(fake, action),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        ones = torch.ones_like(outputs["real_logits"])
        zeros = torch.zeros_like(outputs["fake_logits"])
        generator_adv = nn.functional.binary_cross_entropy_with_logits(
            outputs["fake_logits"], ones
        )
        discriminator = 0.5 * (
            nn.functional.binary_cross_entropy_with_logits(outputs["real_logits"], ones)
            + nn.functional.binary_cross_entropy_with_logits(
                outputs["fake_logits"].detach(), zeros
            )
        )
        total = recon + self.adversarial_weight * (generator_adv + discriminator)
        return {
            "loss": total,
            "recon": recon,
            "gen_adv": generator_adv,
            "disc": discriminator,
        }

    def identifiability_statement(self) -> str:
        return (
            "GANITE: the adversarial discriminator matches the interventional "
            "trajectory distribution; identification relies on the preserved "
            "individual noise rather than on a structural guarantee."
        )
