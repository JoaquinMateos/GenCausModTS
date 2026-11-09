from __future__ import annotations

import torch

from gcmts.causal_representation_learning.methods.citris_image import (
    CITRISImage,
    iCITRISImage,
)
from gcmts.typing import Batch


def test_official_citris_image_adapters_forward_and_loss() -> None:
    batch_size, time, channels, width = 1, 3, 3, 32
    x = torch.rand(batch_size, time, channels * width * width)
    targets = torch.zeros(batch_size, time, 3)
    targets[:, 1:, 0] = 1.0
    batch = Batch(x=x, z=torch.randn(batch_size, time, 3), context={"I": targets})
    for cls in (CITRISImage, iCITRISImage):
        model = cls(
            observed_dim=channels * width * width,
            latent_dim=3,
            n_vars=3,
            image_shape=(channels, width, width),
            c_hid=4,
        )
        output = model(batch)
        assert output.z is not None and output.z.shape == (batch_size, time, 3)
        assert output.x_recon is not None and output.x_recon.shape == batch.x.shape
        losses = model.loss(output, batch)
        assert torch.isfinite(losses["loss"])
