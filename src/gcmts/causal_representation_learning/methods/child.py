"""CHiLD: hierarchical temporal causal representation learning.

Reference: Li et al., *Towards Identifiability of Hierarchical Temporal Causal
Representation Learning*, NeurIPS 2025 — BibTeX
``zijianliIdentifiabilityHierarchicalTemporal2025``.

Observations are generated from ``L`` latent layers ``z^1..z^L``; only the
bottom layer mixes to ``x``. Each component has time-delayed and hierarchical
(higher-layer) parents,

    z^l_{t,i} = f^l_i( Pa_d(z^l_{t,i}), Pa_h(z^l_{t,i}), eps^l_{t,i} ),
    Pa_h(z^l_{t,i}) = z^{l+1}_{t,i},

and the top layer has only time-delayed parents. The encoder maps a
context window of ``x`` (past, current and future steps — the three
conditionally independent views) to the hierarchy; each layer's conditional
prior is an invertible transition network evaluated by change of variables.
Identification is block-wise per layer (Theorem 2) and, given conditional
independence within a layer, component-wise (Lemma 1) up to invertible
transforms without permutation.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor, nn

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.core.backbones import (
    MLP,
    GaussianHead,
    RealNVP,
    gaussian_logpdf,
    gaussian_reconstruction_nll,
    reparameterise,
)
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput

logger = logging.getLogger(__name__)

__all__ = ["CHiLD"]


class _ConditionalInvertible(nn.Module):
    """Invertible map ``z -> eps`` conditioned on the causal parents.

    ``log p(z | cond) = log p_eps(eps) + log|det dz/deps|`` with a standard
    Gaussian ``p_eps`` (CHiLD Eqs. 9–10).
    """

    def __init__(self, dim: int, cond_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.flow = RealNVP(dim, n_layers=2, hidden_dims=hidden_dims, cond_dim=cond_dim)

    def log_prob(self, z: Tensor, cond: Tensor) -> Tensor:
        eps, logdet = self.flow.forward(z, cond)
        zeros = torch.zeros_like(eps)
        return gaussian_logpdf(eps, zeros, zeros) + logdet

    def mean(self, cond: Tensor) -> Tensor:
        """Deterministic next-step value (eps = 0) via the exact inverse."""
        zeros = torch.zeros(cond.shape[0], self.flow.dim, device=cond.device)
        return self.flow.inverse(zeros, cond)


class CHiLD(BaseCausalRepresentationLearner):
    """Hierarchical temporal causal representation learner.

    Args:
        observed_dim: dimension ``D`` of ``x``.
        latent_dim: per-layer latent width ``n``.
        n_layers: number ``L`` of latent layers.
        context_window: half-width ``L`` of the encoder receptive field
            (``2L+1`` adjacent views).
        hidden_dims: hidden widths for the encoder/decoder/prior networks.
        kl_weight: scaling of the KL term.
        obs_noise: observation-noise std for the Gaussian reconstruction NLL.
    """

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int = 3,
        n_layers: int = 2,
        context_window: int | None = None,
        hidden_dims: tuple[int, ...] = (128, 128),
        kl_weight: float = 1.0,
        obs_noise: float = 0.1,
    ) -> None:
        if obs_noise <= 0:
            raise ValueError("obs_noise must be positive.")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        super().__init__(
            observed_dim=observed_dim, latent_dim=latent_dim * n_layers, max_lag=1
        )
        self.width = latent_dim
        self.n_layers = n_layers
        self.half_window = context_window if context_window is not None else n_layers
        self.kl_weight = kl_weight
        kernel = 2 * self.half_window + 1
        self.features = nn.Conv1d(
            observed_dim, hidden_dims[0], kernel_size=kernel, padding=self.half_window
        )
        self.encoder_heads = nn.ModuleList(
            GaussianHead(
                hidden_dims[0] if level == 0 else latent_dim,
                latent_dim,
                hidden_dims=hidden_dims,
            )
            for level in range(n_layers)
        )
        self.decoder = MLP(latent_dim, observed_dim, hidden_dims=hidden_dims)
        self.priors = nn.ModuleList(
            _ConditionalInvertible(
                latent_dim,
                cond_dim=2 * latent_dim if layer < n_layers - 1 else latent_dim,
                hidden_dims=hidden_dims,
            )
            for layer in range(n_layers)
        )
        self.register_buffer("obs_logvar", torch.log(torch.tensor(obs_noise**2)).expand(1).clone())
        self.obs_logvar: Tensor
        logger.debug("Initialised CHiLD (D=%d, n=%d, L=%d)", observed_dim, latent_dim, n_layers)

    # -- encoder / decoder ---------------------------------------------------

    def _encode_distribution(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mean, logvar)`` stacked over layers, shape ``(B, T, L*n)``."""
        batch, time, _ = x.shape
        h = self.features(x.transpose(1, 2)).transpose(1, 2)  # (B, T, hidden)
        means, logvars = [], []
        feature = h
        for head in self.encoder_heads:
            mean, logvar = head(feature.reshape(batch * time, -1))
            mean = mean.reshape(batch, time, self.width)
            logvar = logvar.reshape(batch, time, self.width)
            means.append(mean)
            logvars.append(logvar)
            feature = mean  # hierarchical: layer l+1 conditions on layer l
        return torch.cat(means, dim=-1), torch.cat(logvars, dim=-1)

    def _split_layers(self, z: Tensor) -> list[Tensor]:
        return [z[..., i * self.width : (i + 1) * self.width] for i in range(self.n_layers)]

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        mean, _ = self._encode_distribution(x)
        return mean

    def decode(self, z: Tensor) -> Tensor:
        bottom = z[..., : self.width]
        batch, time, _ = bottom.shape
        out = self.decoder(bottom.reshape(batch * time, -1))
        return out.reshape(batch, time, self.observed_dim)

    # -- model ---------------------------------------------------------------

    def _prior_log_prob(self, layers: list[Tensor]) -> Tensor:
        """Sum of hierarchical transition log-densities, shape ``(B,)``."""
        batch, time, _ = layers[0].shape
        device = layers[0].device
        total = torch.zeros(batch, device=device, dtype=layers[0].dtype)
        for layer in range(self.n_layers):
            z = layers[layer]
            total = total + gaussian_logpdf(
                z[:, 0], torch.zeros_like(z[:, 0]), torch.zeros_like(z[:, 0])
            )
            if time > 1:
                lagged = z[:, :-1]
                current = z[:, 1:]
                if self.n_layers > 1 and layer == self.n_layers - 1:
                    cond = lagged
                else:
                    higher = layers[layer + 1]
                    cond = torch.cat([lagged, higher[:, 1:]], dim=-1)
                flat_current = current.reshape(-1, current.shape[-1])
                flat_cond = cond.reshape(-1, cond.shape[-1])
                logp = self.priors[layer].log_prob(flat_current, flat_cond)
                total = total + logp.reshape(batch, time - 1).sum(dim=1)
        return total

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        mean, logvar = self._encode_distribution(batch.x)
        z = reparameterise(mean, logvar)
        layers = self._split_layers(z)
        log_q = gaussian_logpdf(z, mean, logvar)  # (B, T)
        prior = self._prior_log_prob(layers)
        x_recon = self.decode(z)
        return CausalRepresentationOutput(
            z=z,
            x_recon=x_recon,
            log_prob=prior,
            extras={
                "mean": mean,
                "logvar": logvar,
                "log_q": log_q,
                "prior": prior,
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        if outputs.x_recon is None:
            raise ValueError("CHiLD forward output is missing the reconstruction.")
        recon = gaussian_reconstruction_nll(
            outputs.x_recon.reshape_as(batch.x), batch.x, self.obs_logvar
        )
        extras = outputs.extras or {}
        log_q = extras["log_q"]
        prior = extras["prior"].unsqueeze(1)
        kl = (log_q - prior).mean()
        total = recon + self.kl_weight * kl
        return {"loss": total, "recon": recon, "kl": kl}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        """Deterministic one-step forecast by inverting each layer's prior at eps=0."""
        lagged = z_past[:, -1]
        lagged_layers = self._split_layers(lagged)
        outputs: list[Tensor] = []
        higher = torch.zeros_like(lagged_layers[-1])
        for layer in range(self.n_layers - 1, -1, -1):
            if self.n_layers > 1 and layer == self.n_layers - 1:
                cond = lagged_layers[layer]
            else:
                cond = torch.cat([lagged_layers[layer], higher], dim=-1)
            value = self.priors[layer].mean(cond)
            outputs.insert(0, value)
            higher = value
        return torch.cat(outputs, dim=-1)

    def predict_next_latents(
        self, z: Tensor, steps: int, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        history = z
        preds = []
        for _ in range(steps):
            z_next = self.transition(history, context)
            preds.append(z_next)
            history = torch.cat([history, z_next.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    def predict_next_observations(
        self, x: Tensor, steps: int, context: dict[str, Tensor] | None = None
    ) -> Tensor:
        z = self.encode(x, context)
        z_future = self.predict_next_latents(z, steps, context)
        return self.decode(cast(Tensor, z_future))

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "CHiLD: with a context window of 2L+1 adjacent observations and "
            "invertible hierarchical transition networks, each layer z^l_t is "
            "identified up to an invertible transform block-wise (Thm. 2) and, "
            "given within-layer conditional independence, component-wise without "
            "permutation (Lemma 1); temporal structure supplies sufficient "
            "variability."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
