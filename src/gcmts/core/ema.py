"""Exponential moving average of model weights.

Averaging the iterates of a neural network is a cheap, broadly useful
regulariser: the averaged weights are less sensitive to the noise of the last
optimisation steps than the final iterate, which stabilises validation metrics
and usually improves test performance. It is especially helpful in this
benchmark, where the models are tiny, each run is short, and the reported
numbers are compared across seeds.

The average is maintained over the floating-point entries of ``state_dict``;
non-floating buffers (e.g. ``num_batches_tracked``) are copied verbatim. Use
:meth:`EMA.apply` to evaluate with the averaged weights and restore the live
weights afterwards, or :meth:`EMA.copy_to` to make the average the model's
final weights.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import Tensor, nn

__all__ = ["EMA"]


class EMA:
    """Track an exponential moving average of a module's floating-point state.

    Args:
        model: the module to average. The shadow starts as a copy of its state.
        decay: weight of the running average in ``[0, 1)``; larger is smoother
            (0.999 is a common default for short runs).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1).")
        self.decay = float(decay)
        self.shadow: dict[str, Tensor] = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the model's current state into the average."""
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value)

    def state_dict(self) -> dict[str, Tensor]:
        """Return a detached copy of the averaged state."""
        return {key: value.detach().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.shadow = {key: value.detach().clone() for key, value in state.items()}

    @contextlib.contextmanager
    def apply(self, model: nn.Module) -> Iterator[None]:
        """Temporarily load the averaged weights into ``model`` (restored after)."""
        backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        try:
            yield
        finally:
            model.load_state_dict(backup)

    def copy_to(self, model: nn.Module) -> None:
        """Make the averaged weights the model's permanent state."""
        model.load_state_dict(self.shadow)
