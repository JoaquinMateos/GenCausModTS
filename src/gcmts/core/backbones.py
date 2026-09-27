"""Reusable neural backbones shared by the CRL methods.

Contents:
* :class:`MLP` — configurable feed-forward network.
* :class:`GaussianHead` — MLP emitting ``(mean, logvar)``.
* :class:`ExpFamilyPrior` — factorised Gaussian prior ``p(z|u)``.
* :class:`RealNVP` — invertible affine-coupling normalizing flow.
* Context helpers used by conditional methods.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from gcmts.typing import TensorDict

__all__ = [
    "MLP",
    "GaussianHead",
    "ExpFamilyPrior",
    "AffineCoupling",
    "RealNVP",
    "GradientReversal",
    "flatten_sequence",
    "unflatten_sequence",
    "prepare_context",
    "gaussian_kl",
    "gaussian_logpdf",
    "gaussian_reconstruction_nll",
    "reparameterise",
]

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_logpdf(z: Tensor, mean: Tensor, logvar: Tensor) -> Tensor:
    """Elementwise Gaussian log-density summed over the last dimension."""
    return -0.5 * (
        ((z - mean) ** 2) * torch.exp(-logvar) + logvar + _LOG_2PI
    ).sum(dim=-1)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}

LOGVAR_MIN = -12.0
LOGVAR_MAX = 8.0


class MLP(nn.Module):
    """Feed-forward network with optional batch norm and final activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "silu",
        batchnorm: bool = False,
        final_activation: bool = False,
    ) -> None:
        super().__init__()
        act = _ACTIVATIONS[activation]
        dims = [in_dim, *hidden_dims, out_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act())
            elif final_activation:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class GaussianHead(nn.Module):
    """MLP whose output is split into a mean and a clamped log-variance."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
        activation: str = "silu",
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.net = MLP(
            in_dim,
            2 * out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            batchnorm=batchnorm,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean, logvar = self.net(x).chunk(2, dim=-1)
        return mean, logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)


class ExpFamilyPrior(nn.Module):
    r"""Factorised exponential-family (Gaussian) prior ``p(z|u)``.

    ``log p(z|u) = sum_i log N(z_i; mu_i(u), diag_sigma_i(u))``.

    When ``u_dim == 0`` the prior is a fixed standard normal and no parameters
    are used, which is the unconditional VAE case.
    """

    def __init__(
        self,
        u_dim: int,
        latent_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
        learn_logvar: bool = True,
    ) -> None:
        super().__init__()
        self.u_dim = u_dim
        self.latent_dim = latent_dim
        if u_dim == 0:
            self.register_buffer("_zero_mean", torch.zeros(latent_dim))
            self.register_buffer("_zero_logvar", torch.zeros(latent_dim))
            self._zero_mean: Tensor
            self._zero_logvar: Tensor
        else:
            self.net = GaussianHead(u_dim, latent_dim, hidden_dims=hidden_dims)

    def forward(self, u: Tensor | None) -> tuple[Tensor, Tensor]:
        if self.u_dim == 0:
            n = 1 if u is None else u.shape[0]
            mean = self._zero_mean.expand(n, -1)
            logvar = self._zero_logvar.expand(n, -1)
            return mean, logvar
        if u is None:
            raise ValueError("This prior is conditional but no context was provided.")
        if u.shape[-1] != self.u_dim:
            raise ValueError(
                f"Expected context of dim {self.u_dim}, got {u.shape[-1]}."
            )
        return cast(tuple[Tensor, Tensor], self.net(u))


class _GradientReversal(torch.autograd.Function):
    """Identity forward, negated (scaled) gradient backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.scale * grad_output, None


class GradientReversal(nn.Module):
    """Domain-adversarial gradient reversal layer (Ganin & Lempitsky, 2015)."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, _GradientReversal.apply(x, self.scale))


class AffineCoupling(nn.Module):
    """One affine-coupling layer with a configurable binary mask."""

    def __init__(
        self,
        dim: int,
        *,
        cond_dim: int = 0,
        hidden_dims: tuple[int, ...] = (128, 128),
        mask: Tensor | None = None,
        volume_preserving: bool = False,
        scale_clamp: float = 2.0,
    ) -> None:
        super().__init__()
        if mask is None:
            mask = (torch.arange(dim) % 2 == 0).float()
        if mask.shape != (dim,):
            raise ValueError(f"Mask must have shape ({dim},).")
        self.register_buffer("mask", mask.float())
        self.mask: Tensor
        self.volume_preserving = volume_preserving
        self.scale_clamp = scale_clamp
        self.cond_dim = cond_dim
        self.net = MLP(
            dim + cond_dim,
            2 * dim,
            hidden_dims=hidden_dims,
            activation="silu",
        )

    def _scale_shift(self, x: Tensor, cond: Tensor | None) -> tuple[Tensor, Tensor]:
        h = x * self.mask
        if self.cond_dim:
            if cond is None:
                raise ValueError("Conditioning vector required.")
            h = torch.cat([h, cond], dim=-1)
        log_s, t = self.net(h).chunk(2, dim=-1)
        log_s = self.scale_clamp * torch.tanh(log_s)
        active = 1.0 - self.mask
        log_s = log_s * active
        t = t * active
        if self.volume_preserving:
            denom = active.sum().clamp(min=1.0)
            log_s = log_s - (log_s.sum(dim=-1, keepdim=True) / denom) * active
        return log_s, t

    def forward(
        self, x: Tensor, cond: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Map ``x -> z``; returns ``(z, log|det J|)``."""
        log_s, t = self._scale_shift(x, cond)
        z = x * self.mask + (1.0 - self.mask) * (x * torch.exp(log_s) + t)
        return z, log_s.sum(dim=-1)

    def inverse(self, z: Tensor, cond: Tensor | None = None) -> Tensor:
        """Map ``z -> x`` with the exact inverse of :meth:`forward`."""
        x_masked = z * self.mask
        log_s, t = self._scale_shift(x_masked, cond)
        x = x_masked + (1.0 - self.mask) * (z - t) * torch.exp(-log_s)
        return x


def _alternating_masks(dim: int, n_layers: int) -> list[Tensor]:
    masks = []
    for layer in range(n_layers):
        offset = layer % 2
        masks.append((torch.arange(dim) % 2 == offset).float())
    return masks


class RealNVP(nn.Module):
    """Stacked affine-coupling flow with optional volume-preserving layers.

    ``forward`` maps data ``x -> z`` (encoding); ``inverse`` maps ``z -> x``
    (decoding). ``log_prob`` evaluates the exact density under an isotropic
    Gaussian base distribution.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_layers: int = 6,
        hidden_dims: tuple[int, ...] = (128, 128),
        cond_dim: int = 0,
        volume_preserving: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.volume_preserving = volume_preserving
        masks = _alternating_masks(dim, n_layers)
        self.layers = nn.ModuleList(
            AffineCoupling(
                dim,
                cond_dim=cond_dim,
                hidden_dims=hidden_dims,
                mask=masks[i],
                volume_preserving=volume_preserving,
            )
            for i in range(n_layers)
        )

    def forward(
        self, x: Tensor, cond: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        layers: list[AffineCoupling] = [
            layer for layer in self.layers if isinstance(layer, AffineCoupling)
        ]
        z = x
        total_logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in layers:
            z, logdet = layer(z, cond)
            total_logdet = total_logdet + logdet
        return z, total_logdet

    def inverse(self, z: Tensor, cond: Tensor | None = None) -> Tensor:
        layers: list[AffineCoupling] = [
            layer for layer in self.layers if isinstance(layer, AffineCoupling)
        ]
        x = z
        for layer in reversed(layers):
            x = layer.inverse(x, cond)
        return x

    def log_prob(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        z, logdet = self.forward(x, cond)
        base = -0.5 * (z**2 + torch.log(torch.tensor(2 * torch.pi))).sum(dim=-1)
        return base + logdet


# ---------------------------------------------------------------------------
# Context / tensor helpers
# ---------------------------------------------------------------------------

def flatten_sequence(x: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """Flatten ``(B, T, D) -> (B*T, D)``; pass through ``(B, D)``."""
    if x.ndim == 3:
        b, t, d = x.shape
        return x.reshape(b * t, d), (b, t)
    if x.ndim == 2:
        return x, (x.shape[0], 1)
    raise ValueError("Expected a 2D or 3D tensor.")


def unflatten_sequence(x: Tensor, batch: int, time: int) -> Tensor:
    """Reshape ``(B*T, d) -> (B, T, d)``."""
    return x.reshape(batch, time, x.shape[-1])


def _align_rows(value: Tensor, n: int) -> Tensor:
    """Align leading dimension ``B`` of a 2D tensor to ``n = B * T`` rows."""
    if value.shape[0] == n:
        return value
    if value.shape[0] == 1:
        return value.expand(n, -1)
    if n % value.shape[0] == 0:
        return value.repeat_interleave(n // value.shape[0], dim=0)
    raise ValueError(
        f"Cannot align context of leading dim {value.shape[0]} to {n} rows."
    )


def prepare_context(
    context: TensorDict | None,
    n: int,
    *,
    key: str = "u",
) -> Tensor | None:
    """Return a float context vector of shape ``(n, U)`` or ``None``.

    Integer/bool context entries are one-hot encoded; float entries are used
    as-is. Per-sample context ``(B, U)`` is repeated over ``T`` time steps to
    match a flattened sequence of ``n = B * T`` rows.
    """
    if context is None or key not in context:
        return None
    value = context[key]
    if value.dtype in (torch.int64, torch.int32, torch.bool):
        flat = value.reshape(-1)
        n_classes = int(flat.max().item()) + 1 if flat.numel() else 1
        one_hot = torch.nn.functional.one_hot(flat.long(), n_classes).float()
        return _align_rows(one_hot, n)
    value = value.float()
    if value.ndim == 3:
        value = value.reshape(-1, value.shape[-1])
    elif value.ndim == 1:
        value = value.reshape(-1, 1)
    return _align_rows(value, n)


def gaussian_reconstruction_nll(
    x_recon: Tensor,
    x: Tensor,
    logvar: Tensor,
) -> Tensor:
    r"""Gaussian reconstruction NLL with a learned observation log-variance.

    ``-log N(x; x_recon, exp(logvar))`` averaged over all elements:

    .. math::
        \tfrac{1}{2}\big((x-\hat x)^2 e^{-\log\sigma^2} + \log\sigma^2
        + \log 2\pi\big).

    Using a learned ``logvar`` makes the reconstruction/KL balance data-driven
    instead of fixing an implicit unit variance, which matters when the true
    observation noise is much smaller than the data scale.
    """
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    squared = (x_recon - x) ** 2
    return (0.5 * (squared * torch.exp(-logvar) + logvar + _LOG_2PI)).mean()


def gaussian_kl(
    q_mean: Tensor,
    q_logvar: Tensor,
    p_mean: Tensor,
    p_logvar: Tensor,
) -> Tensor:
    """KL( N(q_mean, exp(q_logvar)) || N(p_mean, exp(p_logvar)) ) per row."""
    var_q = torch.exp(q_logvar)
    var_p = torch.exp(p_logvar)
    return 0.5 * (
        (var_q + (q_mean - p_mean) ** 2) / var_p
        - 1.0
        + p_logvar
        - q_logvar
    ).sum(dim=-1)


def reparameterise(
    mean: Tensor,
    logvar: Tensor,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``z = mean + eps * std`` with ``eps ~ N(0, I)``."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return mean + eps * std
