"""A minimal, well-logged PyTorch trainer implementing :class:`BaseTrainer`."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, cast

import torch
from torch import Tensor

from gcmts.core.base import BaseModel, BaseTrainer
from gcmts.typing import Batch

logger = logging.getLogger(__name__)

__all__ = ["SimpleTrainer"]


def move_batch(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor in a :class:`Batch` to ``device`` in place."""
    batch.x = batch.x.to(device)
    if batch.z is not None:
        batch.z = batch.z.to(device)
    if batch.mask is not None:
        batch.mask = batch.mask.to(device)
    if batch.context is not None:
        batch.context = {k: v.to(device) for k, v in batch.context.items()}
    if batch.adjacency is not None:
        batch.adjacency = batch.adjacency.to(device)
    return batch


def _subset_batch(batch: Batch, indices: Tensor) -> Batch:
    context = (
        {k: v[indices] for k, v in batch.context.items()} if batch.context else None
    )
    return Batch(
        x=batch.x[indices],
        z=None if batch.z is None else batch.z[indices],
        context=context,
        mask=None if batch.mask is None else batch.mask[indices],
        adjacency=batch.adjacency,
    )


class SimpleTrainer(BaseTrainer):
    """Plain PyTorch training loop with deterministic batching and logging.

    Args:
        max_epochs: number of passes over the (re-sampled) data.
        lr: learning rate for Adam.
        batch_size: number of trajectories per optimisation step.
        steps_per_epoch: optimisation steps per epoch.
        device: ``"auto"`` selects CUDA/MPS/CPU.
        grad_clip: optional global gradient-norm clip.
        log_every: log every ``log_every`` epochs; ``0`` disables logging.
        seed: optional seed for the internal batch sampler.
        weight_decay: Adam weight decay.
    """

    def __init__(
        self,
        *,
        max_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        steps_per_epoch: int = 20,
        device: str = "auto",
        grad_clip: float | None = None,
        log_every: int = 10,
        seed: int | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(max_epochs=max_epochs, device=device)
        self.lr = lr
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.weight_decay = weight_decay
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )

    def fit(
        self,
        model: BaseModel,
        train_data: Any,
        val_data: Any | None = None,
    ) -> dict[str, list[float]]:
        device = self._device_for(model)
        model.to(device)
        model.train()
        optimiser = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        history: dict[str, list[float]] = defaultdict(list)

        logger.info(
            "Training %s on %s for %d epochs (%d steps/epoch, batch=%d, device=%s)",
            type(model).__name__,
            type(train_data).__name__,
            self.max_epochs,
            self.steps_per_epoch,
            self.batch_size,
            device,
        )

        for epoch in range(self.max_epochs):
            terms: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.steps_per_epoch):
                batch = move_batch(
                    self._draw_batch(train_data), device
                )
                outputs = model.forward(batch)
                losses = model.loss(outputs, batch)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss at epoch %d; skipping step.", epoch)
                    optimiser.zero_grad(set_to_none=True)
                    break
                optimiser.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                optimiser.step()
                for name, value in losses.items():
                    terms[name].append(float(value.detach().cpu()))

            if not terms:
                continue
            for name, values in terms.items():
                history[name].append(sum(values) / len(values))

            if self.log_every and (epoch % self.log_every == 0 or epoch == self.max_epochs - 1):
                summary = ", ".join(f"{k}={v[-1]:.4f}" for k, v in history.items())
                logger.info("epoch %d/%d | %s", epoch + 1, self.max_epochs, summary)

        return dict(history)

    def predict(self, model: BaseModel, data: Any) -> Tensor | dict[str, Tensor]:
        device = self._device_for(model)
        model.to(device)
        model.eval()
        batch = move_batch(self._draw_batch(data), device)
        with torch.no_grad():
            outputs = model.forward(batch)
        if hasattr(outputs, "z"):
            return cast(Tensor, outputs.z)
        if isinstance(outputs, dict):
            return outputs
        return cast(Tensor | dict[str, Tensor], outputs)

    def _draw_batch(self, data: Any) -> Batch:
        if isinstance(data, Batch):
            n = data.x.shape[0]
            size = min(self.batch_size, n)
            indices = torch.randperm(n, generator=self.generator)[:size]
            return _subset_batch(data, indices)
        if hasattr(data, "sample"):
            return cast(Batch, data.sample(self.batch_size))
        batch = next(iter(data))
        if not isinstance(batch, Batch):
            raise TypeError("Data loader must yield gcmts.typing.Batch objects.")
        return batch
