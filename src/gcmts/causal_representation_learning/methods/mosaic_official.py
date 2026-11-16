"""MOSAIC, adapted from the official implementation.

Paper: Fan et al., *MOSAIC: Module Discovery via Sparse Additive Identifiable
Causal Learning for Scientific Time Series* (2026), BibTeX
``shichengfanMOSAICModuleDiscovery2026``.
Official source: https://github.com/shichengf/mosaic (MIT license).
The official additive decoder, regime-conditioned temporal prior, sparsity
schedule and two-stage freeze schedule are used; this file adapts the batch and
epoch interfaces to ``gcmts``.

Input scaling.  The vendored model is built for unit-scale variables: the
latent prior is a fixed unit Gaussian, the additive decoder's sub-networks are
Kaiming-initialised for unit-variance scalar inputs, and the reconstruction term
is an *unweighted* MSE over the observation units.  The released loader
(``mosaic/data/dataset.py``) therefore reads a *prepared* ``data.npz`` whose
preparation step lives outside the released code; that preparation is the only
place the model's unit-scale assumptions can come from.  A benchmark that hands
the model physical units silently (i) reweights the reconstruction term across
observed variables by their variances and (ii) initialises ``sum_j f_j(z_j)``
orders of magnitude away from the data.  :class:`MOSAICOfficial` therefore
reproduces the missing per-variable standardisation with running statistics
(:attr:`obs_mean`/:attr:`obs_var`); outputs returned by :meth:`decode` /
:meth:`forward` are mapped back to the input units so the public interface is
unchanged.  Pass ``standardise=False`` to train the model on raw inputs, as an
ablation.
"""

from __future__ import annotations

import logging
from typing import cast

import torch
from torch import Tensor

from gcmts.causal_representation_learning.base import BaseCausalRepresentationLearner
from gcmts.typing import AmbiguityClass, Batch, CausalRepresentationOutput
from gcmts.vendor.mosaic.mosaic.models.mosaic import TimeVaryingProcess_v2

logger = logging.getLogger(__name__)

__all__ = ["MOSAICOfficial"]

_INTEGER_DTYPES = (torch.int64, torch.int32, torch.int16, torch.bool)


def _ct(
    context: dict[str, Tensor] | None, batch_size: int, n_classes: int, device: torch.device
) -> Tensor:
    """Return the regime label ``(B,)`` that conditions the temporal prior.

    The official model is conditioned on one regime per window.  Benchmark
    generators use two encodings: a per-window one-hot/soft indicator
    ``(B, n_classes)`` (CartPole, latent-SCM, NP/VAR) or one integer regime
    label per timestep ``(B, T)`` (physics, control and linear-process suites).
    Integer labels are reduced to the regime of the window's final step; a
    float matrix of width ``n_classes`` is read as an indicator.  Dtype decides
    between the two when a label series happens to have length ``n_classes``,
    matching :func:`gcmts.causal_representation_learning.methods.ivae_official
    ._aux`.
    """
    if context is None or "u" not in context or n_classes <= 1:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    u = context["u"]
    if u.ndim == 3:
        # Per-timestep labels or indicators: use the last step of the window.
        u = u[:, -1]
    if u.ndim == 1:
        return u.long().clamp(0, n_classes - 1).to(device)
    if u.dtype in _INTEGER_DTYPES:
        # One integer label per window, or one per timestep.
        labels = u[:, -1] if u.shape[1] > 1 else u[:, 0]
        return labels.long().clamp(0, n_classes - 1).to(device)
    if u.shape[1] == 1:
        return u[:, 0].long().clamp(0, n_classes - 1).to(device)
    if u.shape[1] != n_classes:
        raise ValueError(
            f"Context 'u' has width {u.shape[1]} but the model was built for "
            f"{n_classes} regimes. Pass u_dim={u.shape[1]} (one-hot context) or "
            "an integer regime-label tensor."
        )
    return u.argmax(dim=-1).long().clamp(0, n_classes - 1).to(device)


class MOSAICOfficial(BaseCausalRepresentationLearner):
    """Interface around the official MOSAIC additive temporal VAE."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_dim: int,
        u_dim: int = 0,
        hidden_dim: int = 128,
        embedding_dim: int = 8,
        hidden_per_z: int = 8,
        beta: float = 0.0025,
        gamma: float = 0.0075,
        lambda_sparse: float = 1e-3,
        warmup_epochs: int = 2,
        rampup_epochs: int = 3,
        freeze_epochs: int = 3,
        lr: float = 1e-3,
        hidden_dims: tuple[int, ...] = (128, 128),
        decoder_hidden: tuple[int, ...] = (64,),
        sparsity: float = 0.01,
        kl_weight: float = 1.0,
        obs_noise: float = 1.0,
        standardise: bool = True,
        image_shape: tuple[int, int, int] | None = None,
        image_feature_dim: int = 64,
    ) -> None:
        del hidden_dims, decoder_hidden, sparsity, kl_weight, obs_noise
        super().__init__(
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            max_lag=1,
            image_shape=image_shape,
            image_feature_dim=image_feature_dim,
        )
        self.n_classes = max(int(u_dim), 1)
        self.trainer_lr = float(lr)
        # Rendered-image inputs are already mapped to CNN features, which are not
        # the "prepared variables" the released preprocessing standardises.
        self.standardise = bool(standardise) and image_shape is None
        if self.standardise:
            width = self.encoder_input_dim(observed_dim)
            self.register_buffer("obs_mean", torch.zeros(1, width))
            self.register_buffer("obs_var", torch.ones(1, width))
            self.register_buffer("obs_count", torch.zeros((), dtype=torch.long))
            self.obs_mean: Tensor
            self.obs_var: Tensor
            self.obs_count: Tensor
        self.core = TimeVaryingProcess_v2(  # type: ignore[no-untyped-call]
            input_dim=self.encoder_input_dim(observed_dim),
            # The window length is re-derived from every batch (see `_batch`);
            # this constructor value is only the vendor model's placeholder.
            length=1,
            z_dim=latent_dim,
            lag=1,
            nclass=self.n_classes,
            hidden_dim=int(hidden_dim),
            embedding_dim=int(embedding_dim),
            hidden_per_z=int(hidden_per_z),
            decoder_type="additive",
            trans_prior="NP",
            lr=float(lr),
            infer_mode="F",
            beta=float(beta),
            gamma=float(gamma),
            lambda_sparse=float(lambda_sparse),
            warmup_epochs=int(warmup_epochs),
            rampup_epochs=int(rampup_epochs),
            freeze_epochs=int(freeze_epochs),
        )
        self.core.log = lambda *args, **kwargs: None  # type: ignore[method-assign]

    def _scale(self) -> Tensor:
        """Per-variable standard deviation used for the input transform."""
        return torch.clamp(self.obs_var, min=1e-8).sqrt()

    def _obs(self, x: Tensor) -> Tensor:
        """Encoder-space observations (standardised when enabled)."""
        if self.image_shape is None:
            obs = x
        else:
            features, (batch, time) = self.observation_features(x)
            obs = features.reshape(batch, time, -1)
        if not self.standardise:
            return obs
        return (obs - self.obs_mean) / self._scale()

    def _denormalise(self, obs: Tensor) -> Tensor:
        if not self.standardise:
            return obs
        return obs * self._scale() + self.obs_mean

    def _update_observation_stats(self, x: Tensor) -> None:
        """Fold the current training batch into the running per-variable statistics.

        Chan's parallel update of mean/variance; only training batches are
        counted (validation runs under ``eval()``), so the statistics describe
        the same data the released pipeline standardises once, offline.
        """
        if not (self.standardise and self.training):
            return
        flat = x.reshape(-1, x.shape[-1]).detach()
        n = flat.shape[0]
        if n == 0:
            return
        batch_mean = flat.mean(dim=0, keepdim=True)
        batch_var = flat.var(dim=0, unbiased=False, keepdim=True)
        count = int(self.obs_count.item())
        if count == 0:
            self.obs_mean.copy_(batch_mean)
            self.obs_var.copy_(batch_var)
        else:
            total = count + n
            delta = batch_mean - self.obs_mean
            self.obs_mean.add_(delta * (n / total))
            self.obs_var.mul_(count / total).add_(
                batch_var * (n / total) + delta.pow(2) * (count * n / total**2)
            )
        self.obs_count.fill_(count + n)

    def _batch(self, batch: Batch) -> dict[str, Tensor]:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        ct = _ct(batch.context, obs.shape[0], self.n_classes, obs.device)
        return {"xt": obs, "yt": obs, "ct": ct}

    def on_epoch_start(self, epoch: int) -> None:
        self.core.manual_epoch = int(epoch)
        self.core.on_train_epoch_start()  # type: ignore[no-untyped-call]

    def encode(self, x: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        obs = self._obs(x)
        self.core.length = int(obs.shape[1])
        flat = obs.reshape(-1, obs.shape[-1])
        encoded = cast(Tensor, self.core.net._encode(flat))  # type: ignore[no-untyped-call]
        z_dim = self.latent_dim
        assert z_dim is not None
        return encoded[:, :z_dim].reshape(obs.shape[0], obs.shape[1], z_dim)

    def decode(self, z: Tensor) -> Tensor:
        flat = z.reshape(-1, z.shape[-1])
        rec = self._denormalise(self.core.net._decode(flat))  # type: ignore[no-untyped-call]
        return rec.reshape(z.shape[0], z.shape[1], self.observed_dim)

    def forward(self, batch: Batch) -> CausalRepresentationOutput:
        obs = self._obs(batch.x)
        self.core.length = int(obs.shape[1])
        flat = obs.reshape(-1, obs.shape[-1])
        x_recon, mean, logvar, z = self.core.net(flat)
        x_recon = self._denormalise(x_recon)
        return CausalRepresentationOutput(
            z=z.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
            x_recon=x_recon.reshape_as(obs),
            extras={
                "mean": mean.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
                "logvar": logvar.reshape(obs.shape[0], obs.shape[1], self.latent_dim),
            },
        )

    def loss(self, outputs: CausalRepresentationOutput, batch: Batch) -> dict[str, Tensor]:
        # Update the standardisation statistics once per training step, before
        # the ELBO is evaluated on this batch.
        self._update_observation_stats(batch.x)
        loss = self.core.training_step(self._batch(batch), 0)  # type: ignore[no-untyped-call]
        return {"loss": loss, "recon": torch.zeros((), device=loss.device)}

    def transition(self, z_past: Tensor, context: dict[str, Tensor] | None = None) -> Tensor:
        return z_past[:, -1]

    def get_latent_adjacency(self) -> Tensor | None:
        return None

    def identifiability_statement(self) -> str:
        return (
            "MOSAIC: a regime-conditioned temporal prior identifies the latent "
            "factors up to permutation and component-wise transforms; the "
            "additive decoder then exposes the observation support of each "
            "latent. The official two-stage/additive model is used."
        )

    def ambiguity_class(self) -> str:
        return AmbiguityClass.PERM_COMPONENTWISE
