r"""CEPAE: Conditional Entropy-Penalised AutoEncoder for time-series counterfactuals.

Reference: Garriga, Sanz, Serrahima & Brando, *CEPAE: Conditional Entropy-
Penalized Autoencoders for Time Series Counterfactuals*, TMLR 2026 — BibTeX
``tomasgarrigaCEPAEConditionalEntropyPenalized2026``.

The audited implementation corrects a fidelity error: the earlier version was a
conditional *adversarial* autoencoder (CAAE).  CEPAE is a deterministic
conditional autoencoder regularised by the entropy penalty of Eq. (10) of the
paper,

    L_EP = sum_i std_batch(Z_i),

which upper-bounds the latent entropy and discourages the latent from
redundantly encoding information already present in the conditioners.  There is
no discriminator and no KL term.  Abduction encodes the factual trajectory with
its factual actions; action replaces the actions; prediction decodes the same
latent with the counterfactual actions.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from gcmts.core.backbones import MLP
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["CEPAE"]


class CEPAE(BaseEffectEstimator):
    """Deterministic conditional autoencoder with an entropy penalty.

    Args:
        observed_dim: dimension ``D`` of the observed trajectory.
        treatment_dim: dimension of the treatment/action.
        latent_dim: dimension of the abducted latent state ``z``.
        entropy_weight: weight of the batch-standard-deviation entropy penalty.
        hidden_dims / decoder_hidden: network widths.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        latent_dim: int = 4,
        entropy_weight: float = 0.01,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        self.z_dim = latent_dim
        self.entropy_weight = entropy_weight
        self.encoder = MLP(
            observed_dim + treatment_dim, latent_dim, hidden_dims=hidden_dims
        )
        self.decoder = MLP(
            latent_dim + treatment_dim, observed_dim, hidden_dims=decoder_hidden
        )

    def _actions(self, batch: Batch) -> Tensor:
        action = None
        if batch.context is not None and "A" in batch.context:
            action = batch.context["A"]
        if action is None:
            action = torch.zeros(
                batch.x.shape[0], batch.x.shape[1], self.treatment_dim, device=batch.x.device
            )
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, batch.x.shape[1], -1)
        return action.float().to(batch.x.device)

    def _encode(self, x: Tensor, action: Tensor) -> Tensor:
        batch, time, _ = x.shape
        inp = torch.cat([x, action], dim=-1).reshape(batch * time, -1)
        return torch.as_tensor(self.encoder(inp)).reshape(batch, time, self.z_dim)

    def _decode(self, z: Tensor, action: Tensor) -> Tensor:
        batch, time, _ = z.shape
        inp = torch.cat([z, action], dim=-1).reshape(batch * time, -1)
        return torch.as_tensor(self.decoder(inp)).reshape(batch, time, self.observed_dim)

    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        action = self._actions(batch)
        return {"z": self._encode(batch.x, action), "action": action}

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, noise["z"].shape[1], -1)
        return {"z": noise["z"], "action": new_action.float()}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        return self._decode(context["z"][:, :horizon], context["action"][:, :horizon])

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
        z = self._encode(batch.x, action)
        recon = self._decode(z, action)
        return {"z": z, "recon": recon, "action": action}

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        reconstruction = nn.functional.mse_loss(outputs["recon"], batch.x)
        flat = outputs["z"].reshape(-1, self.z_dim)
        # Eq. (10): sum of the batch standard deviations of the latent
        # coordinates.  This is the paper's stable surrogate for S(Z).
        entropy_penalty = flat.std(dim=0, unbiased=False).sum()
        total = reconstruction + self.entropy_weight * entropy_penalty
        return {
            "loss": total,
            "recon": reconstruction,
            "entropy": entropy_penalty,
        }

    def identifiability_statement(self) -> str:
        return (
            "CEPAE: the entropy penalty encourages the latent to be independent "
            "of the conditioners while retaining reconstruction information; the "
            "counterfactual is the decoder applied to the abducted latent with "
            "the replaced actions, up to the autoencoder's reparameterisation."
        )
