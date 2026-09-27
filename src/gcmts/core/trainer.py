"""A minimal, well-logged PyTorch trainer implementing :class:`BaseTrainer`."""

from __future__ import annotations

import logging
import math
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
    """Plain PyTorch training loop with scheduling, validation and logging.

    Args:
        max_epochs: number of passes over the (re-sampled) data.
        lr: peak learning rate for Adam.
        batch_size: number of trajectories per optimisation step.
        steps_per_epoch: optimisation steps per epoch.
        device: ``"auto"`` selects CUDA/MPS/CPU.
        grad_clip: global gradient-norm clip (``None`` disables; default 1.0).
        log_every: log every ``log_every`` epochs; ``0`` disables logging.
        seed: optional seed for the internal batch sampler.
        weight_decay: Adam weight decay.
        scheduler: ``"cosine"`` (warmup + cosine), ``"plateau"``, ``"step"`` or
            ``"none"``.
        min_lr_ratio: floor of the cosine schedule as a fraction of ``lr``.
        warmup_fraction: fraction of total steps used for linear warmup.
        val_size: size of the held-out validation batch (``0`` disables
            validation). Used for monitoring, plateau scheduling and early
            stopping.
        val_every: evaluate the validation batch every ``val_every`` epochs.
        early_stopping_patience: stop if validation loss does not improve for
            this many epochs (``None`` disables).
    """

    def __init__(
        self,
        *,
        max_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        steps_per_epoch: int = 20,
        device: str = "auto",
        grad_clip: float | None = 1.0,
        log_every: int = 10,
        seed: int | None = None,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        min_lr_ratio: float = 0.05,
        warmup_fraction: float = 0.05,
        val_size: int = 0,
        val_every: int = 1,
        early_stopping_patience: int | None = None,
    ) -> None:
        super().__init__(max_epochs=max_epochs, device=device)
        if scheduler not in {"cosine", "plateau", "step", "none"}:
            raise ValueError(f"Unknown scheduler '{scheduler}'.")
        self.lr = lr
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.min_lr_ratio = min_lr_ratio
        self.warmup_fraction = warmup_fraction
        self.val_size = val_size
        self.val_every = max(1, val_every)
        self.early_stopping_patience = early_stopping_patience
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )

    def _build_scheduler(
        self, optimiser: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        total_steps = max(self.max_epochs * self.steps_per_epoch, 1)
        warmup_steps = int(self.warmup_fraction * total_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(warmup_steps, 1))
            progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
        if self.scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, mode="min", factor=0.5, patience=5
            )
        if self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimiser, step_size=max(self.max_epochs // 3, 1), gamma=0.5
            )
        return None

    def _make_val_batch(self, data: Any, device: torch.device) -> Batch | None:
        if self.val_size <= 0:
            return None
        if isinstance(data, Batch):
            n = data.x.shape[0]
            size = min(self.val_size, n)
            indices = torch.randperm(n, generator=self.generator)[:size]
            return move_batch(_subset_batch(data, indices), device)
        if hasattr(data, "sample"):
            return move_batch(cast(Batch, data.sample(self.val_size)), device)
        return move_batch(next(iter(data)), device)

    def _evaluate(
        self, model: BaseModel, batch: Batch
    ) -> dict[str, float]:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            outputs = model.forward(batch)
            losses = model.loss(outputs, batch)
        if was_training:
            model.train()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

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
        scheduler = self._build_scheduler(optimiser)
        per_step_scheduler = isinstance(
            scheduler,
            (torch.optim.lr_scheduler.LambdaLR, torch.optim.lr_scheduler.StepLR),
        )
        val_batch = self._make_val_batch(
            train_data if val_data is None else val_data, device
        )
        history: dict[str, list[float]] = defaultdict(list)
        best_val = float("inf")
        stale = 0

        logger.info(
            "Training %s on %s for %d epochs (%d steps/epoch, batch=%d, "
            "scheduler=%s, val=%s, device=%s)",
            type(model).__name__,
            type(train_data).__name__,
            self.max_epochs,
            self.steps_per_epoch,
            self.batch_size,
            self.scheduler_name,
            val_batch is not None,
            device,
        )

        for epoch in range(self.max_epochs):
            terms: dict[str, list[float]] = defaultdict(list)
            for _ in range(self.steps_per_epoch):
                batch = move_batch(self._draw_batch(train_data), device)
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
                if per_step_scheduler and scheduler is not None:
                    scheduler.step()
                for name, value in losses.items():
                    terms[name].append(float(value.detach().cpu()))

            for name, values in terms.items():
                history[name].append(sum(values) / len(values))

            val_loss: float | None = None
            if val_batch is not None:
                if epoch % self.val_every == 0 or epoch == self.max_epochs - 1:
                    val_loss = self._evaluate(model, val_batch)["loss"]
                history["val_loss"].append(
                    float("nan") if val_loss is None else val_loss
                )
            history["lr"].append(optimiser.param_groups[0]["lr"])

            if scheduler is not None and not per_step_scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss is not None else history["loss"][-1])
                else:
                    scheduler.step()

            if self.log_every and (epoch % self.log_every == 0 or epoch == self.max_epochs - 1):
                summary = ", ".join(f"{k}={v[-1]:.4f}" for k, v in history.items() if v)
                logger.info("epoch %d/%d | %s", epoch + 1, self.max_epochs, summary)

            if self.early_stopping_patience is not None and val_loss is not None:
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.early_stopping_patience:
                        logger.info("Early stopping at epoch %d.", epoch + 1)
                        break

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
