"""LEAP, adapted from the official implementation.

Paper: Yao et al., *Learning Temporally Causal Latent Processes from General
Temporal Data* (ICLR 2022), BibTeX ``weiranyaoLearningTemporallyCausal2021``.
Official source: https://github.com/weirayao/leap (MIT license).
The official recurrent inference network, spline noise model and TC
discriminator are used; this file adapts the batch/optimizer interface to
``gcmts``.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.leap.modules.srnn_ns import SRNNSyntheticNS
from gcmts.vendor.leap.modules.components.mlp import NLayerLeakyMLP

logger = logging.getLogger(__name__)

__all__ = ["LEAPOfficial"]


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


class LEAPOfficial(BaseCausalRepresentationLearner):
    """Interface around the official LEAP nonstationary temporal model."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        max_lag: int = 1,
        u_dim: int = 0,
        length: int = 1,
        hidden_dim: int = 128,
        beta: float = 2e-3,
        gamma: float = 2e-2,
        lr: float = 1e-3,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (128, 128),
        kl_weight: float = 0.01,
        obs_noise: float = 1.0,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del hidden_dims, decoder_hidden, kl_weight, obs_noise
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=max_lag,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.n_classes = max(int(u_dim), 1)
        self.trainer_lr = float(lr)
        self.core = SRNNSyntheticNS(
            input_dim=self.encoder_input_dim(observed_dim),
            length=max(1, int(length)),
            z_dim=latent_dim,
            lag=max_lag,
            nclass=self.n_classes,
            hidden_dim=int(hidden_dim),
            trans_prior="PNL",
            infer_mode="R",
            bound=5,
            count_bins=8,
            order="linear",
            lr=float(lr),
            beta=float(beta),
            gamma=float(gamma),
            sigma=1e-6,
            bias=False,
            decoder_dist="gaussian",
        )
        self.core.log = lambda *args, **kwargs: None
        # The official training script warm-starts the component-wise spline
        # flow on a standard normal/Laplace prior before fitting the VAE.
        # Without this, the spline can produce non-finite log-Jacobians early.
        for spline in self.core.spline_list:
            optimizer = torch.optim.Adam(spline.parameters(), lr=1e-3)
            for _ in range(300):
                y = torch.randn(64, latent_dim, device=next(spline.parameters()).device)
                z, logdet = spline(y)
                loss = -(spline.base_dist.log_prob(z) + logdet).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # The official synthetic config uses input_dim == z_dim.  Our suites do
        # not, so keep the encoder/decoder shapes but adapt their widths.
        self.core.enc = NLayerLeakyMLP(
            in_features=self.encoder_input_dim(observed_dim),
            out_features=latent_dim,
            num_layers=3,
            hidden_dim=128,
        )
        self.core.dec = NLayerLeakyMLP(
            in_features=latent_dim,
            out_features=self.encoder_input_dim(observed_dim),
            num_layers=2,
            hidden_dim=128,
        )
        self.disc_optimizer = torch.optim.SGD(self.core.discriminator.parameters(), lr=1e-3)

    def _obs(self, x: Tensor) -> Tensor:
        if self.image_shape is None:
            return x
        features, (batch, time) = self.observation_features(x)
        return features.reshape(batch, time, -1)

    def _dict(self, batch: Batch) -> dict[str, dict[str, Tensor]]:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        ct = _ct(batch.context, obs.shape[0], self.n_classes, obs.device)
        one = {"xt": obs, "yt": obs, "ct": ct}
        return {"s1": one, "s2": one}

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        obs = self._obs(x)
        self.core.length = int(obs.shape[1])
        flat = obs.reshape(-1, obs.shape[-1])
        features = self.core.enc(flat).reshape(obs.shape[0], obs.shape[1], -1)
        z, _, _ = self.core.inference(features, random_sampling=False)
        return z

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        rec = self.core.dec(flat)
        return rec.reshape(z.shape[0], z.shape[1], -1) if self.image_shape is not None else rec.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        z = self.encode(batch.x, batch.context)
        recon = self.decode(z)
        return CausalRepresentationOutput(z=z, x_recon=recon, extras={"obs": obs})

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        packed = self._dict(batch)
        obs = packed["s1"]["xt"]
        if torch.is_grad_enabled():
            # Discriminator update on a detached residual stream, then the
            # generator update with the discriminator frozen.  This mirrors the
            # official two-optimizer schedule inside the single gcmts trainer.
            params = list(self.core.parameters())
            for p in params:
                p.requires_grad_(False)
            for p in self.core.discriminator.parameters():
                p.requires_grad_(True)
            self.disc_optimizer.zero_grad()
            d_loss = self.core.training_step(packed, 0, 1)
            d_loss.backward()
            self.disc_optimizer.step()
            for p in params:
                p.requires_grad_(True)
            for p in self.core.discriminator.parameters():
                p.requires_grad_(False)
            g_loss = self.core.training_step(packed, 0, 0)
            for p in self.core.discriminator.parameters():
                p.requires_grad_(True)
        else:
            g_loss = self.core.training_step(packed, 0, 0)
        return {"loss": g_loss, "recon": torch.zeros((), device=g_loss.device)}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "LEAP: under nonstationary, regime-dependent conditionals with "
            "linearly independent log-density gradients and an invertible mixing "
            "map, the latents are identifiable up to permutation and "
            "component-wise invertible maps. The official model is used."
        )

    def ambiguity_class(self) -> AmbiguityClass:
        return AmbiguityClass.PERM_COMPONENTWISE
