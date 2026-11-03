"""MOSAIC on the rendered-image path: dense decoder, no per-pixel modules."""

from __future__ import annotations

import torch

from gcmts.causal_representation_learning.methods import MOSAIC
from gcmts.typing import Batch


def test_mosaic_image_forward_backward() -> None:
    size, channels = 16, 3
    observed_dim = channels * size * size
    model = MOSAIC(
        observed_dim=observed_dim,
        latent_dim=4,
        image_shape=(channels, size, size),
        image_feature_dim=8,
        hidden_dims=(16,),
        decoder_hidden=(16,),
    )
    x = torch.rand(2, 2, observed_dim)
    out = model(Batch(x=x))
    assert out.x_recon is not None and out.x_recon.shape == x.shape
    losses = model.loss(out, Batch(x=x))
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert model.support is None
    assert model.get_latent_adjacency() is None


def test_mosaic_scalar_path_keeps_additive_decoder() -> None:
    model = MOSAIC(observed_dim=4, latent_dim=3, hidden_dims=(8,), decoder_hidden=(8,))
    x = torch.rand(2, 3, 4)
    out = model(Batch(x=x))
    assert out.x_recon is not None and out.x_recon.shape == x.shape
    assert model.support is not None
    assert model.get_latent_adjacency() is not None
