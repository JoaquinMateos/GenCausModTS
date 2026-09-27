r"""CEPAE: Conditional Entropy-Penalised AutoEncoder for time-series counterfactuals.

Reference: Tomàs Garriga et al., *CEPAE: Conditional Entropy-Penalized
Autoencoders for Time Series Counterfactuals*, 2026 — BibTeX
``tomasgarrigaCEPAEConditionalEntropyPenalized2026``.

The model is a conditional VAE over trajectories. Abduction infers a latent
state from the factual trajectory, action replaces the treatment, and prediction
decodes the counterfactual from the abducted state. An adversarial action
classifier with a gradient-reversal layer penalises the conditional entropy
``I(z; a)`` so that the latent carries the individual trajectory but not the
treatment, which is what makes the replacement step valid.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor, nn

from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    GradientReversal,
    gaussian_kl,
    gaussian_reconstruction_nll,
)
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["CEPAE"]


class CEPAE(BaseEffectEstimator):
    """Conditional entropy-penalised autoencoder for counterfactual trajectories.

    Args:
        observed_dim: dimension ``D`` of the observed trajectory.
        treatment_dim: dimension of the treatment/action.
        latent_dim: dimension of the abducted latent state ``z``.
        entropy_weight: weight of the adversarial ``I(z; a)`` penalty.
        hidden_dims / decoder_hidden: network widths.
        obs_noise: observation-noise standard deviation for the Gaussian NLL.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        latent_dim: int = 4,
        entropy_weight: float = 1.0,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.z_dim = latent_dim
        self.entropy_weight = entropy_weight
        self.encoder = GaussianHead(observed_dim, latent_dim, hidden_dims=hidden_dims)
        self.decoder = MLP(latent_dim + treatment_dim, observed_dim, hidden_dims=decoder_hidden)
        self.action_classifier = MLP(latent_dim, treatment_dim, hidden_dims=hidden_dims)
        self.reversal = GradientReversal(scale=1.0)
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

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, _ = x.shape
        mean, logvar = self.encoder(x.reshape(batch * time, -1))
        return mean.reshape(batch, time, -1), logvar.reshape(batch, time, -1)

    def _decode(self, z: Tensor, action: Tensor) -> Tensor:
        batch, time, _ = z.shape
        inp = torch.cat([z, action], dim=-1).reshape(batch * time, -1)
        return cast(Tensor, self.decoder(inp)).reshape(batch, time, self.observed_dim)

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self._encode(batch.x)
        return {"z": mean, "logvar": logvar, "action": action}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, noise["z"].shape[1], -1)
        return {"z": noise["z"], "action": new_action.float()}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        z = context["z"][:, :horizon]
        action = context["action"][:, :horizon]
        return self._decode(z, action)

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        factual = batch.x
        counterfactual = self.intervene(batch, counterfactual_action)
        return CounterfactualOutput(
            factual=factual,
            counterfactual=counterfactual,
            noise=self.abduction(batch),
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        mean, logvar = self._encode(batch.x)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        recon = self._decode(z, action)
        return {
            "z": z,
            "mean": mean,
            "logvar": logvar,
            "recon": recon,
            "action": action,
            "action_logits": self.action_classifier(
                self.reversal(mean.reshape(-1, mean.shape[-1]))
            ),
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        kl = gaussian_kl(
            outputs["mean"].reshape(-1, self.z_dim),
            outputs["logvar"].reshape(-1, self.z_dim),
            torch.zeros_like(outputs["mean"]).reshape(-1, self.z_dim),
            torch.zeros_like(outputs["logvar"]).reshape(-1, self.z_dim),
        ).mean()
        target = outputs["action"].reshape(-1, self.treatment_dim)
        entropy = nn.functional.binary_cross_entropy_with_logits(
            outputs["action_logits"], target
        )
        total = recon + kl + self.entropy_weight * entropy
        return {"loss": total, "recon": recon, "kl": kl, "entropy": entropy}

    def identifiability_statement(self) -> str:
        return (
            "CEPAE: with an invertible decoder and an abducted latent state that "
            "is independent of the treatment (enforced adversarially), the "
            "counterfactual is identifiable up to the decoder's reparameterisation."
        )
