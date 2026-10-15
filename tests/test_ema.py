"""Tests for the exponential moving average of model weights."""

from __future__ import annotations

import torch
from torch import nn

from gcmts.core.ema import EMA


def test_ema_converges_and_apply_restores_weights() -> None:
    model = nn.Linear(3, 2)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(4.0)
    for _ in range(25):
        ema.update(model)
    for value in ema.shadow.values():
        if value.is_floating_point():
            assert torch.allclose(value, torch.full_like(value, 4.0), atol=1e-3)

    original = {key: value.clone() for key, value in model.state_dict().items()}
    with ema.apply(model):
        assert torch.allclose(model.weight, torch.full_like(model.weight, 4.0), atol=1e-3)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_ema_state_dict_roundtrip() -> None:
    model = nn.Linear(3, 2)
    source = EMA(model, decay=0.9)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    source.update(model)

    target = EMA(model, decay=0.5)
    target.load_state_dict(source.state_dict())
    assert set(target.shadow) == set(source.shadow)
    for key, value in source.shadow.items():
        assert torch.equal(target.shadow[key], value)


def test_ema_rejects_invalid_decay() -> None:
    import pytest

    with pytest.raises(ValueError):
        EMA(nn.Linear(2, 2), decay=1.0)
