"""Placeholder concrete methods for static-dynamic disentanglement."""

from __future__ import annotations

import torch
from torch import Tensor

from gcmts.static_dynamic_disentanglement.base import BaseStaticDynamicDisentangler
from gcmts.typing import Batch, DisentangledFactors

__all__ = ["DummyDisentangler"]


class DummyDisentangler(BaseStaticDynamicDisentangler):
    """Minimal stub that splits the observation dimension into static/dynamic."""

    def __init__(
        self,
        *,
        observed_dim: int,
        latent_static_dim: int,
        latent_dynamic_dim: int,
    ) -> None:
        super().__init__(
            observed_dim=observed_dim,
            latent_static_dim=latent_static_dim,
            latent_dynamic_dim=latent_dynamic_dim,
        )

    def encode_static(self, x: Tensor) -> Tensor:
        return x[:, :, : self.latent_static_dim]

    def encode_dynamic(self, x: Tensor) -> Tensor:
        return x[:, :, self.latent_static_dim : self.latent_static_dim + self.latent_dynamic_dim]

    def decode(self, factors: DisentangledFactors) -> Tensor:
        parts = [factors.z_stc, factors.z_dyc]
        if factors.z_sts is not None:
            parts.append(factors.z_sts)
        if factors.z_dys is not None:
            parts.append(factors.z_dys)
        return torch.cat(parts, dim=-1)[:, :, : self.observed_dim]

    def disentangle(self, x: Tensor) -> DisentangledFactors:
        return DisentangledFactors(
            z_stc=self.encode_static(x),
            z_dyc=self.encode_dynamic(x),
        )

    def predict(self, x: Tensor, target_domain: Tensor | None = None) -> Tensor:
        return x

    def ood_score(self, x: Tensor, target_domain: Tensor | None = None) -> Tensor:
        return torch.zeros(x.shape[0])

    def loss(self, outputs: dict[str, Tensor], batch: Batch) -> dict[str, Tensor]:
        return {"loss": torch.tensor(0.0)}

    def identifiability_statement(self) -> str:
        return "Dummy disentangler: no identifiability guarantees."
