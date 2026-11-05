"""NCTRL, adapted from the official implementation.

Paper: Song et al., *Temporally Disentangled Representation Learning under
Unknown Nonstationarity* (NeurIPS 2023), BibTeX
``songTemporallyDisentangledRepresentation2023``.
Official source: https://github.com/xiangchensong/nctrl (MIT license).
The official ARHMM, nonparametric change prior and VAE are used; this file
adapts the batch/optimizer interface to ``gcmts``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.nctrl.models.simulation import NCTRL

logger = logging.getLogger(__name__)

__all__ = ["NCTRLOfficial"]


def _ct(context: dict[str, Tensor] | None, batch_size: int, n_classes: int, device: torch.device) -> Tensor:
    if context is None or "u" not in context:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    u = context["u"]
    if u.ndim == 3:
        u = u[:, -1]
    if u.ndim == 2 and u.shape[1] not in (1, n_classes):
        # Physics/control suites store one regime index per timestep; the
        # official models need the regime that labels the sequence window.
        u = u[:, -1]
    if u.ndim == 1:
        return u.long().to(device)
    if u.shape[-1] == 1:
        return u[:, 0].long().to(device)
    return u.argmax(dim=-1).long().to(device)


class NCTRLOfficial(BaseCausalRepresentationLearner):
    """Interface around the official NCTRL ARHMM model."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        n_regimes: int = 3,
        hidden_dim: int = 128,
        beta: float = 0.0025,
        gamma: float = 0.0075,
        lr: float = 1e-3,
        hmm_lr: float = 1e-3,
        transition_hidden: int = 128,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 0.01,
        obs_noise: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del transition_hidden, hidden_dims, decoder_hidden, kl_weight, obs_noise
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.trainer_lr = float(lr)
        self.core = NCTRL(
            x_dim=self.encoder_input_dim(observed_dim),
            z_dim=latent_dim,
            lags=1,
            n_class=max(int(n_regimes), 1),
            hidden_dim=int(hidden_dim),
            lr=float(lr),
            beta=float(beta),
            gamma=float(gamma),
            hmm_mode="mle_scaled:Ht",
        )
        self.hmm_optimizer = torch.optim.Adam(self.core.hmm.parameters(), lr=float(hmm_lr))
        self.core.log = lambda *args, **kwargs: None

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        obs = self._obs(x)
        _, mean, _, _ = self.core.net(obs)
        return mean.reshape(obs.shape[0], obs.shape[1], self.latent_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        rec = self.core.net._decode(flat)
        return rec.reshape(z.shape[0], z.shape[1], -1) if self.image_shape is not None else rec.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        obs = self._obs(batch.x)
        x_recon, mean, logvar, z = self.core.net(obs)
        logvar = torch.clamp(logvar, -10.0, 10.0)
        mean = torch.clamp(mean, -10.0, 10.0)
        z = torch.clamp(z, -10.0, 10.0)
        return CausalRepresentationOutput(
            z=z,
            x_recon=x_recon,
            extras={"mean": mean, "logvar": logvar},
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        obs = self._obs(batch.x)
        x_recon, mus, logvars, z_est = self.core.net(obs)
        logvars = torch.clamp(logvars, -10.0, 10.0)
        mus = torch.clamp(mus, -10.0, 10.0)
        z_est = torch.clamp(z_est, -10.0, 10.0)
        E_logp_x, c_est = self.core.hmm(z_est.detach())
        hmm_loss = -E_logp_x.mean()
        if torch.is_grad_enabled():
            self.hmm_optimizer.zero_grad()
            hmm_loss.backward(retain_graph=True)
            self.hmm_optimizer.step()

        embeddings = self.core.c_embeddings(c_est)
        recon = self.core.reconstruction_loss(obs[:, :1], x_recon[:, :1]) + (
            self.core.reconstruction_loss(obs[:, 1:], x_recon[:, 1:]) / max(1, obs.shape[1] - 1)
        )
        q_dist = torch.distributions.Normal(mus, torch.exp(logvars / 2).clamp(min=1e-6, max=1e6))
        log_qz = q_dist.log_prob(z_est)
        p_dist = torch.distributions.Normal(
            torch.zeros_like(mus[:, :1]), torch.ones_like(logvars[:, :1])
        )
        kld_normal = (
            torch.sum(p_dist.log_prob(z_est[:, :1]), dim=-1).sum(-1)
            - torch.sum(log_qz[:, :1], dim=-1).sum(-1)
        ).mean()
        residuals, logabsdet = self.core.transition_prior(z_est, embeddings)
        log_pz = torch.sum(self.core.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
            torch.sum(log_qz[:, 1:], dim=-1).sum(-1) - log_pz
        ).div(max(1, obs.shape[1] - 1)).mean()
        total = recon + self.core.beta * kld_normal + self.core.gamma * kld_laplace
        return {"loss": total, "recon": recon}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "NCTRL: under a Markov assumption on time-delayed causally related "
            "processes and nonstationary regimes with independent components, "
            "the latent factors are identifiable up to permutation and "
            "component-wise transforms without observed auxiliary variables. "
            "The official ARHMM/flow implementation is used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
