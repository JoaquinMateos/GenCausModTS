"""Counterfactual Recurrent Network (CRN) with a recurrent encoder/decoder.

Reference: Bica, Alaa, Lambert & van der Schaar, *Estimating Counterfactual
Treatment Outcomes over Time Through Adversarially Balanced Representations*,
ICLR 2020 — BibTeX ``bicaEstimatingCounterfactualTreatment2020``.

The audited implementation replaces the earlier linear state-space surrogate
with the architecture of the paper: an LSTM encoder over the factual history
``(x_t, a_{t-1})`` produces a representation ``Phi(h_t)``; an LSTM decoder is
initialised from the encoder state and rolls the representation forward under
the intended treatment sequence; an outcome head reconstructs the observations.
A treatment classifier is trained adversarially through a gradient-reversal
layer, so the representation is predictive of the outcome but not of the
assigned treatment (the balancing property that identifies the counterfactual).

For a trajectory counterfactual the encoder sees the factual prefix, the
counterfactual action sequence replaces the treatment from the split onward,
and the decoder holds the abducted recurrent state fixed.  A one-step model
(static benchmarks) uses a small ``[Phi(h), a]`` outcome head.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from gcmts.core.backbones import (
    MLP,
    GradientReversal,
    gaussian_reconstruction_nll,
)
from gcmts.effect_estimation.base import BaseEffectEstimator
from gcmts.typing import Batch, CounterfactualOutput

logger = logging.getLogger(__name__)

__all__ = ["CRN"]


class CRN(BaseEffectEstimator):
    """Adversarially balanced recurrent counterfactual estimator.

    Args:
        observed_dim: observation dimension ``D``.
        treatment_dim: treatment dimension ``A``.
        latent_dim: dimension of the balanced representation ``Phi``.
        horizon: trajectory length the model is trained on.
        hidden_dim: LSTM hidden width.
        adversarial_weight: weight of the treatment-classifier loss (through a
            gradient-reversal layer).
        obs_noise: observation-noise standard deviation.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        treatment_dim: int = 1,
        latent_dim: int = 6,
        horizon: int = 24,
        hidden_dim: int = 64,
        adversarial_weight: float = 1.0,
        obs_noise: float = 0.1,
    ) -> None:
        super().__init__(observed_dim=observed_dim, treatment_dim=treatment_dim)
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        self.state_dim = latent_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.adversarial_weight = adversarial_weight
        self.encoder = nn.LSTM(
            observed_dim + treatment_dim,
            hidden_dim,
            batch_first=True,
        )
        self.representation = MLP(hidden_dim, latent_dim)
        self.outcome_head = MLP(latent_dim, observed_dim)
        self.decoder = nn.LSTM(latent_dim + treatment_dim, hidden_dim, batch_first=True)
        self.decoder_head = MLP(hidden_dim, observed_dim)
        # One-step (static) path: the decoder LSTM needs a prefix to initialise.
        self.static_head = MLP(latent_dim + treatment_dim, observed_dim)
        self.classifier = MLP(latent_dim, treatment_dim)
        self.reversal = GradientReversal(scale=1.0)
        self.register_buffer(
            "obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone()
        )
        self.obs_logvar: Tensor

    # -- helpers ---------------------------------------------------------
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

    @staticmethod
    def _lagged_actions(actions: Tensor) -> Tensor:
        return torch.cat([torch.zeros_like(actions[:, :1]), actions[:, :-1]], dim=1)

    def _encode(self, x: Tensor, actions: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        inputs = torch.cat([x, self._lagged_actions(actions)], dim=-1)
        hidden, state = self.encoder(inputs)
        phi = self.representation(hidden)
        return phi, state

    def _decode(
        self,
        state: tuple[Tensor, Tensor],
        phi: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """Roll the decoder from a representation and action sequence."""
        outputs: list[Tensor] = []
        hidden_state = state
        current_phi = phi
        for step in range(actions.shape[1]):
            inp = torch.cat([current_phi, actions[:, step]], dim=-1).unsqueeze(1)
            hidden, hidden_state = self.decoder(inp, hidden_state)
            outputs.append(self.decoder_head(hidden[:, 0]))
            current_phi = self.representation(hidden[:, 0])
        return torch.stack(outputs, dim=1)

    # -- BaseEffectEstimator --------------------------------------------
    def abduction(self, batch: Batch) -> dict[str, Tensor]:
        actions = self._actions(batch)
        time = batch.x.shape[1]
        split = 1 if time <= 1 else max(1, time // 4)
        phi, state = self._encode(batch.x[:, :split], actions[:, :split])
        return {
            "state_h": state[0],
            "state_c": state[1],
            "phi": phi[:, -1],
            "split": torch.tensor(split, device=batch.x.device),
            "horizon": torch.tensor(time, device=batch.x.device),
            "action": actions,
            "prefix": batch.x[:, :split],
        }

    def action(
        self, noise: dict[str, Tensor], intervention: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        new_action = intervention["A"]
        if new_action.ndim == 2:
            new_action = new_action.unsqueeze(1).expand(-1, int(noise["horizon"].item()), -1)
        return {**noise, "action": new_action.float()}

    def prediction(self, context: dict[str, Tensor], horizon: int) -> Tensor:
        split = int(context["split"].item())
        if horizon <= split:
            return context["prefix"]
        # The decoder consumes a_{t-1} to predict x_t, so the suffix starts one
        # action earlier than the first decoded observation.
        actions = context["action"][:, split - 1 : horizon - 1]
        if actions.shape[1] == 0:
            return context["prefix"]
        state = (context["state_h"], context["state_c"])
        generated = self._decode(state, context["phi"], actions)
        return torch.cat([context["prefix"], generated], dim=1)

    def intervene(self, batch: Batch, intervention: dict[str, Tensor]) -> Tensor:
        context = self.action(self.abduction(batch), intervention)
        return self.prediction(context, batch.x.shape[1])

    def counterfactual(
        self, batch: Batch, counterfactual_action: dict[str, Tensor]
    ) -> CounterfactualOutput:
        if batch.x.shape[1] == 1:
            actions = self._actions(batch)
            phi, _ = self._encode(batch.x, actions)
            new_action = counterfactual_action["A"]
            if new_action.ndim == 2:
                new_action = new_action.unsqueeze(1)
            if new_action.ndim == 3 and new_action.shape[1] != 1:
                new_action = new_action[:, :1]
            prediction = self.static_head(torch.cat([phi[:, -1], new_action[:, 0]], dim=-1))
            return CounterfactualOutput(
                factual=batch.x,
                counterfactual=prediction.unsqueeze(1),
                noise={"phi": phi},
                intervention=counterfactual_action,
            )
        return CounterfactualOutput(
            factual=batch.x,
            counterfactual=self.intervene(batch, counterfactual_action),
            noise=self.abduction(batch),
            intervention=counterfactual_action,
        )

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        if batch.x.shape[1] != self.horizon:
            raise ValueError(
                f"CRN was built for horizon={self.horizon} but received T={batch.x.shape[1]}."
            )
        actions = self._actions(batch)
        phi, _ = self._encode(batch.x, actions)
        recon = self.outcome_head(phi)
        logits = self.classifier(self.reversal(phi.reshape(-1, phi.shape[-1])))
        time = batch.x.shape[1]
        split = 1 if time <= 1 else max(1, time // 4)
        if time > split:
            phi_prefix, state_prefix = self._encode(
                batch.x[:, :split], actions[:, :split]
            )
            decoded = self._decode(
                state_prefix,
                phi_prefix[:, -1],
                actions[:, split - 1 : time - 1],
            )
        else:
            decoded = None
        static = (
            self.static_head(torch.cat([phi[:, -1], actions[:, -1]], dim=-1))
            if time == 1
            else None
        )
        return {
            "phi": phi,
            "recon": recon,
            "decoded": decoded,
            "static": static,
            "action_logits": logits,
            "action": actions,
        }

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        recon = gaussian_reconstruction_nll(outputs["recon"], batch.x, self.obs_logvar)
        target = outputs["action"].reshape(-1, self.treatment_dim)
        adversarial = nn.functional.binary_cross_entropy_with_logits(
            outputs["action_logits"], target
        )
        total = recon + self.adversarial_weight * adversarial
        decoder_loss = torch.zeros((), device=batch.x.device)
        if outputs.get("decoded") is not None:
            split = max(1, batch.x.shape[1] // 4)
            decoder_loss = gaussian_reconstruction_nll(
                outputs["decoded"], batch.x[:, split:], self.obs_logvar
            )
            total = total + decoder_loss
        if outputs.get("static") is not None:
            static_loss = gaussian_reconstruction_nll(
                outputs["static"], batch.x[:, 0], self.obs_logvar
            )
            total = total + static_loss
        else:
            static_loss = torch.zeros((), device=batch.x.device)
        return {
            "loss": total,
            "recon": recon,
            "adversarial": adversarial,
            "decoder": decoder_loss,
            "static": static_loss,
        }

    def identifiability_statement(self) -> str:
        return (
            "CRN: under sequential ignorability and positivity, the recurrent "
            "representation is treatment-invariant (adversarially balanced) and "
            "the counterfactual roll-out holds the abducted recurrent state fixed; "
            "counterfactuals are identifiable up to the outcome decoder's "
            "reparameterisation."
        )
