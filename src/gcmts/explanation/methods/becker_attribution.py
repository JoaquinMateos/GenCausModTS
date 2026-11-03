"""Compact Becker et al. attribution model for branch III.C (explanation).

Reference: Becker, Niemietz & Bergs, *Explainable neural network for
time-series-based condition monitoring in fine blanking*, Journal of
Intelligent Manufacturing 37(3):1247-1263, 2026,
DOI ``10.1007/s10845-025-02596-3`` (CC BY 4.0).

The original method combines a beta-VAE with the Ada-GVAE pairing constraint of
Locatello et al. (2020) and a multilayer perceptron that predicts the target
from the posterior means. Pairing is motivated by the independent causal
mechanisms (ICM) principle: within a short window only a few mechanisms change,
so paired samples share most latent coordinates. Ada-GVAE averages the
posterior coordinates whose pairwise KL divergence lies below the midpoint of
the per-pair minimum and maximum KL.

This implementation is a **compact surrogate** on flattened time series, not a
reproduction of the convolutional architecture or the fine-blanking
experiment. It exposes two attribution heads evaluated at the posterior mean
``mu``:

* ``attributions_grad`` -- gradient of the predictor logit with respect to
  ``mu`` (local feature importance); and
* ``attributions`` -- mean-ablation importance, i.e. the logit difference when
  one coordinate of ``mu`` is replaced by its zero mean.

There is no identifiability theorem for this surrogate: Ada-GVAE supplies an
ICM-based inductive bias, and the attributions are external-validity evidence
relative to the learned predictor.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch.nn import functional as F

from gcmts.core.backbones import MLP, GaussianHead, gaussian_kl, reparameterise
from gcmts.explanation.base import BaseExplainer
from gcmts.typing import Batch, CausalLevel

logger = logging.getLogger(__name__)

__all__ = ["BeckerAttribution"]


class BeckerAttribution(BaseExplainer):
    """beta-VAE + Ada-GVAE pairing with gradient and mean-ablation heads.

    Args:
        n_channels: number of observed channels ``D``.
        horizon: sequence length ``T``.
        n_concepts: number of latent factors ``C`` (also the attribution
            width), default 4.
        n_classes: number of label classes (binary by default).
        latent_dim: latent width; defaults to ``n_concepts``.
        hidden_dims: encoder head widths.
        decoder_hidden: decoder widths.
        predictor_hidden: predictor widths (the paper uses ReLU MLPs).
        beta: weight of the KL term in the beta-VAE objective.
        bce_weight: weight of the predictor's binary cross-entropy.
    """

    def __init__(
        self,
        *,
        n_channels: int = 6,
        horizon: int = 12,
        n_concepts: int = 4,
        n_classes: int = 2,
        latent_dim: int | None = None,
        hidden_dims: tuple[int, ...] = (64, 64),
        decoder_hidden: tuple[int, ...] = (64, 64),
        predictor_hidden: tuple[int, ...] = (32, 32),
        beta: float = 0.1,
        bce_weight: float = 1.0,
    ) -> None:
        super().__init__(
            n_channels=n_channels,
            horizon=horizon,
            n_concepts=n_concepts,
            n_classes=n_classes,
        )
        if n_classes != 2:
            raise ValueError("BeckerAttribution implements the binary predictor.")
        self.latent_dim_ = latent_dim if latent_dim is not None else n_concepts
        self._latent_dim = self.latent_dim_
        self.beta = beta
        self.kl_weight = beta
        self.bce_weight = bce_weight
        flat = n_channels * horizon
        self.encoder = GaussianHead(flat, self.latent_dim_, hidden_dims=hidden_dims)
        self.decoder = MLP(self.latent_dim_, flat, hidden_dims=decoder_hidden)
        self.predictor = MLP(
            self.latent_dim_,
            1,
            hidden_dims=predictor_hidden,
            activation="relu",
        )
        logger.debug(
            "Initialised BeckerAttribution (D=%d, T=%d, C=%d)",
            n_channels,
            horizon,
            self.latent_dim_,
        )

    # -- encoding helpers ----------------------------------------------------

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.encoder(x.reshape(x.shape[0], -1))

    @staticmethod
    def _pair_kl(mean_a: Tensor, logvar_a: Tensor, mean_b: Tensor, logvar_b: Tensor) -> Tensor:
        """Component-wise KL between two diagonal Gaussian posteriors."""
        var_a = torch.exp(logvar_a)
        var_b = torch.exp(logvar_b)
        return 0.5 * ((var_a + (mean_a - mean_b) ** 2) / var_b - 1.0 + logvar_b - logvar_a)

    @staticmethod
    def _shared_mask(mean_a: Tensor, logvar_a: Tensor, mean_b: Tensor, logvar_b: Tensor) -> Tensor:
        """Ada-GVAE mask: coordinates with KL below the midpoint threshold."""
        delta = BeckerAttribution._pair_kl(mean_a, logvar_a, mean_b, logvar_b)
        tau = 0.5 * (delta.max(dim=-1).values + delta.min(dim=-1).values)
        return delta < tau.unsqueeze(-1)

    def _labels(self, batch: Batch) -> Tensor:
        if batch.outcome is not None:
            return batch.outcome.reshape(-1).float()
        if batch.context is not None and "y" in batch.context:
            return batch.context["y"].reshape(-1).float()
        return torch.zeros(batch.x.shape[0], device=batch.x.device)

    # -- forward / loss ------------------------------------------------------

    def forward(self, batch: Batch) -> dict[str, Tensor]:
        x = batch.x.float()
        mean, logvar = self._encode(x)
        outputs: dict[str, Tensor] = {
            "mu": mean,
            "logvar": logvar,
            "x_recon": self.decoder(reparameterise(mean, logvar)).reshape_as(x),
            "logits": self.predictor(mean).squeeze(-1),
        }
        pair_x = batch.context.get("pair_x") if batch.context is not None else None
        if pair_x is None:
            return outputs

        pair_x = pair_x.float()
        pair_mean, pair_logvar = self._encode(pair_x)
        shared = self._shared_mask(mean, logvar, pair_mean, pair_logvar)
        avg_mean = 0.5 * (mean + pair_mean)
        avg_logvar = 0.5 * (logvar + pair_logvar)
        mean_a = torch.where(shared, avg_mean, mean)
        logvar_a = torch.where(shared, avg_logvar, logvar)
        mean_b = torch.where(shared, avg_mean, pair_mean)
        logvar_b = torch.where(shared, avg_logvar, pair_logvar)
        outputs.update(
            {
                "mu": mean_a,
                "logvar": logvar_a,
                "mu_pair": mean_b,
                "logvar_pair": logvar_b,
                "x_recon_pair": self.decoder(reparameterise(mean_b, logvar_b)).reshape_as(pair_x),
                "logits_pair": self.predictor(mean_b).squeeze(-1),
                "shared": shared.float(),
            }
        )
        return outputs

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        labels = self._labels(batch)
        zeros = torch.zeros_like(outputs["mu"])
        recon = F.mse_loss(outputs["x_recon"], batch.x.float())
        kl = gaussian_kl(outputs["mu"], outputs["logvar"], zeros, zeros).mean()
        bce = F.binary_cross_entropy_with_logits(outputs["logits"], labels)
        if "x_recon_pair" in outputs:
            # Ada-GVAE optimises the sum of the two paired beta-VAE objectives.
            pair = batch.context["pair_x"] if batch.context is not None else None
            if pair is None:
                raise ValueError("Paired outputs require context['pair_x'].")
            zeros_pair = torch.zeros_like(outputs["mu_pair"])
            recon = recon + F.mse_loss(outputs["x_recon_pair"], pair.float())
            kl = (
                kl
                + gaussian_kl(
                    outputs["mu_pair"], outputs["logvar_pair"], zeros_pair, zeros_pair
                ).mean()
            )
        total = recon + self.beta * kl + self.bce_weight * bce
        return {"loss": total, "recon": recon, "kl": kl, "bce": bce}

    # -- explanation ---------------------------------------------------------

    def explain(self, batch: Batch) -> dict[str, Tensor]:
        """Return gradient and mean-ablation attributions of shape ``(B, C)``."""
        labels = self._labels(batch)
        mean, _ = self._encode(batch.x.float())
        mean = mean.detach().requires_grad_(True)
        logits = self.predictor(mean).squeeze(-1)
        (gradient,) = torch.autograd.grad(logits.sum(), mean)
        with torch.no_grad():
            baseline = logits.detach()
            ablation = torch.zeros_like(mean)
            for factor in range(self.latent_dim_):
                edited = mean.detach().clone()
                edited[:, factor] = 0.0
                ablation[:, factor] = baseline - self.predictor(edited).squeeze(-1)
        return {
            "attributions": ablation,
            "attributions_grad": gradient.detach(),
            "labels": labels,
            "mu": mean.detach(),
        }

    @torch.no_grad()
    def manipulate(self, batch: Batch, factor: int, delta: float | Tensor) -> dict[str, Tensor]:
        """Add ``delta`` to latent ``factor``, decode, and re-predict."""
        if not 0 <= factor < self.latent_dim_:
            raise ValueError(f"factor must lie in [0, {self.latent_dim_}), got {factor}.")
        mean, _ = self._encode(batch.x.float())
        edit = torch.as_tensor(delta, dtype=mean.dtype, device=mean.device)
        if edit.ndim == 0:
            edit = edit.expand(mean.shape[0])
        edited = mean.clone()
        edited[:, factor] = edited[:, factor] + edit.reshape(-1)
        logits = self.predictor(edited).squeeze(-1)
        return {
            "x": self.decoder(edited).reshape_as(batch.x),
            "mu": edited,
            "logits": logits,
            "prediction": torch.sigmoid(logits),
        }

    def identifiability_statement(self) -> str:
        return (
            "Becker et al. (2026): the Ada-GVAE step assumes independent causal "
            "mechanisms -- paired samples share most factors and differ in only a "
            "few -- and averages posterior coordinates with small pairwise KL to "
            "encourage axis alignment. No identifiability theorem is claimed. "
            "Attributions are the predictor-logit gradient with respect to the "
            "posterior mean (gradient head) and the logit change under zero-mean "
            "ablation of each coordinate (mean-ablation head); they are "
            "external-validity evidence relative to the learned predictor."
        )

    def causal_level(self) -> str:
        return CausalLevel.INTERVENTION
