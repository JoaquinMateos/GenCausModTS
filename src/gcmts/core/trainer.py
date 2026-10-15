"""A minimal, well-logged PyTorch trainer implementing :class:`BaseTrainer`."""

from __future__ import annotations

import contextlib
import logging
import math
import os
import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from torch import Tensor

from gcmts.core.base import BaseModel, BaseTrainer
from gcmts.core.ema import EMA
from gcmts.core.profiling import PhaseProfiler
from gcmts.typing import Batch

logger = logging.getLogger(__name__)

__all__ = ["AdaptiveScheduler", "EMA", "SimpleTrainer", "move_batch"]

#: Keys the training losses actually read (all CRL and effect methods use only
#: ``x`` and ``context``); everything else is evaluation-only.
TRAIN_BATCH_KEYS = frozenset({"x", "context"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() not in {"", "0", "false", "no"}


def _env_int(name: str) -> int | None:
    """Positive integer from the environment, or ``None`` when unset/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _env_float(name: str) -> float | None:
    """Float from the environment, or ``None`` when unset/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _null_phase(name: str) -> contextlib.AbstractContextManager[None]:
    return contextlib.nullcontext()


def move_batch(
    batch: Batch, device: torch.device, *, keys: frozenset[str] | None = None
) -> Batch:
    """Move every tensor in a :class:`Batch` to ``device`` in place.

    ``keys`` restricts the move to a subset of fields (e.g.
    :data:`TRAIN_BATCH_KEYS`), so evaluation-only tensors (``z``, ``adjacency``,
    ``outcome``, ...) need not be copied during training.
    """

    def want(name: str) -> bool:
        return keys is None or name in keys

    if want("x"):
        batch.x = batch.x.to(device)
    if batch.z is not None and want("z"):
        batch.z = batch.z.to(device)
    if batch.mask is not None and want("mask"):
        batch.mask = batch.mask.to(device)
    if batch.context is not None and want("context"):
        batch.context = {k: v.to(device) for k, v in batch.context.items()}
    if batch.adjacency is not None and want("adjacency"):
        batch.adjacency = batch.adjacency.to(device)
    if batch.counterfactual is not None and want("counterfactual"):
        batch.counterfactual = batch.counterfactual.to(device)
    if batch.outcome is not None and want("outcome"):
        batch.outcome = batch.outcome.to(device)
    if batch.domain is not None and want("domain"):
        batch.domain = batch.domain.to(device)
    return batch


class _BatchPrefetcher:
    """Generate training batches on a background thread.

    Data generation is CPU-bound and dominates the step for most suites, while
    the model step is small; producing batch ``i+1`` while step ``i`` runs hides
    the compute behind generation. The worker is the only caller of the sampler,
    so the RNG sequence (and therefore the training data) is unchanged.
    """

    def __init__(self, draw: Callable[[], Batch], *, depth: int = 2) -> None:
        self._draw = draw
        self._queue: queue.Queue[Batch | None] = queue.Queue(maxsize=max(depth, 1))
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="gcmts-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                batch = self._draw()
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:  # noqa: BLE001 - propagated via next()
            self._error = error
            with contextlib.suppress(queue.Full):
                self._queue.put(None, timeout=0.2)

    def next(self) -> Batch:
        batch = self._queue.get()
        if batch is None:
            raise self._error or RuntimeError("batch prefetcher stopped unexpectedly")
        return batch

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                self._queue.get_nowait()
        self._thread.join(timeout=2.0)



class AdaptiveScheduler:
    """Linear warm-up followed by plateau-driven decay.

    Warms the learning rate up over ``warmup_epochs`` epochs, then delegates to
    :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`: the rate is multiplied
    by ``factor`` when the monitored metric stops improving by ``threshold`` for
    ``patience`` epochs, and floored at ``min_lr_ratio`` times the base rate.
    Unlike a fixed cosine over a pre-set epoch budget it reacts to the observed
    progress, so it needs no known number of epochs — which is what makes it a
    better fit for the benchmark's over-sized budgets, where convergence
    happens long before the schedule ends.

    It is stepped once per epoch with the monitored metric (validation loss when
    available, otherwise the training loss).
    """

    def __init__(
        self,
        optimiser: torch.optim.Optimizer,
        *,
        warmup_epochs: int = 5,
        factor: float = 0.5,
        patience: int = 5,
        threshold: float = 1e-3,
        threshold_mode: Literal["rel", "abs"] = "rel",
        min_lr_ratio: float = 0.05,
    ) -> None:
        self.optimiser = optimiser
        self.base_lrs = [float(group["lr"]) for group in optimiser.param_groups]
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self._epoch = 0
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            min_lr=[lr * min_lr_ratio for lr in self.base_lrs],
        )

    def _set_base_lrs(self, scale: float) -> None:
        for group, base in zip(self.optimiser.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * scale

    def step(self, metric: float | None = None) -> None:
        """Advance one epoch.

        ``metric`` (lower is better) drives the decay; ``None`` advances only
        the warm-up, so callers can skip the plateau update on epochs where no
        fresh validation estimate is available.
        """
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            self._set_base_lrs(self._epoch / self.warmup_epochs)
            return
        if self._epoch == self.warmup_epochs + 1:
            # Restore the nominal rate for the first post-warm-up epoch.
            self._set_base_lrs(1.0)
        if metric is not None:
            self._plateau.step(metric)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimiser.param_groups]


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
        counterfactual=None
        if batch.counterfactual is None
        else batch.counterfactual[indices],
        outcome=None if batch.outcome is None else batch.outcome[indices],
        domain=None if batch.domain is None else batch.domain[indices],
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
        prefetch: generate the next training batch on a background thread while
            the current step runs (results-neutral; defaults to ``GCMTS_PREFETCH``).
        light_batch: copy only ``x`` and ``context`` to the device during
            training, skipping evaluation-only tensors (defaults to
            ``GCMTS_LIGHT_BATCH``).
        profile: record per-phase step timings (defaults to ``GCMTS_PROFILE``);
            intended for short diagnostic runs only.
        early_stopping_min_delta: absolute improvement in validation loss
            required to reset the patience counter.
        early_stopping_min_delta_ratio: relative improvement (fraction of the
            best value) additionally required; use a tiny value for a very
            permissive stop that only fires on a true plateau.
        restore_best_weights: restore the validation-best weights when training
            stops (or finishes). Off by default so the returned model is the
            final one, preserving the historical behaviour.
        adaptive_warmup_epochs: linear warm-up epochs for the ``adaptive``
            scheduler (per-epoch, unlike the per-step warm-up of ``cosine``).
        adaptive_patience: epochs without improvement before the ``adaptive``
            scheduler reduces the learning rate.
        adaptive_factor: multiplicative learning-rate decay for ``adaptive``.
        adaptive_threshold: relative improvement threshold for ``adaptive``.
        amp: train under ``torch.autocast`` with a ``GradScaler`` (CUDA only;
            defaults to ``GCMTS_AMP``). opt-in because it slightly changes the
            numerics of a run.
        ema_decay: if set, keep an exponential moving average of the weights and
            evaluate with it (defaults to ``GCMTS_EMA_DECAY``); ``None``
            disables it.
        kl_warmup_epochs: if positive, linearly ramp any ``model.kl_weight``
            attribute from 0 to its configured value over this many epochs
            (``beta`` warm-up for the VAE-based CRL methods).
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
        early_stopping_min_delta: float = 0.0,
        early_stopping_min_delta_ratio: float = 0.0,
        restore_best_weights: bool = False,
        adaptive_warmup_epochs: int = 5,
        adaptive_patience: int = 5,
        adaptive_factor: float = 0.5,
        adaptive_threshold: float = 1e-3,
        amp: bool | None = None,
        ema_decay: float | None = None,
        kl_warmup_epochs: int = 0,
        prefetch: bool | None = None,
        light_batch: bool | None = None,
        profile: bool = False,
    ) -> None:
        super().__init__(max_epochs=max_epochs, device=device)
        if scheduler not in {"cosine", "plateau", "step", "adaptive", "none"}:
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
        self.early_stopping_patience = (
            early_stopping_patience
            if early_stopping_patience is not None
            else _env_int("GCMTS_EARLY_STOPPING_PATIENCE")
        )
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_min_delta_ratio = early_stopping_min_delta_ratio
        self.restore_best_weights = restore_best_weights
        self.adaptive_warmup_epochs = max(int(adaptive_warmup_epochs), 0)
        self.adaptive_patience = max(int(adaptive_patience), 1)
        self.adaptive_factor = adaptive_factor
        self.adaptive_threshold = adaptive_threshold
        self.amp = (_env_flag("GCMTS_AMP") if amp is None else amp)
        self.ema_decay = (
            ema_decay if ema_decay is not None else _env_float("GCMTS_EMA_DECAY")
        )
        self.kl_warmup_epochs = max(int(kl_warmup_epochs), 0)
        self.prefetch = _env_flag("GCMTS_PREFETCH") if prefetch is None else prefetch
        self.light_batch = _env_flag("GCMTS_LIGHT_BATCH") if light_batch is None else light_batch
        self.profile = profile or _env_flag("GCMTS_PROFILE")
        self.generator = (
            torch.Generator().manual_seed(seed) if seed is not None else None
        )
        self.seed = seed
        self.profiler: PhaseProfiler | None = None
        self._iterators: dict[int, Any] = {}
        self.best_val: float | None = None
        self.best_epoch: int | None = None
        self.stopped_epoch: int | None = None
        self._ema: EMA | None = None
        self._use_amp = False
        self._amp_device = "cpu"

    def _build_scheduler(
        self, optimiser: torch.optim.Optimizer
    ) -> (
        torch.optim.lr_scheduler.LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
        | AdaptiveScheduler
        | None
    ):
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
        if self.scheduler_name == "adaptive":
            return AdaptiveScheduler(
                optimiser,
                warmup_epochs=self.adaptive_warmup_epochs,
                factor=self.adaptive_factor,
                patience=self.adaptive_patience,
                threshold=self.adaptive_threshold,
                min_lr_ratio=self.min_lr_ratio,
            )
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

    def _evaluate(self, model: BaseModel, batch: Batch) -> dict[str, float]:
        was_training = model.training
        model.eval()
        ema_context = (
            self._ema.apply(model) if self._ema is not None else contextlib.nullcontext()
        )
        with (
            ema_context,
            torch.no_grad(),
            torch.autocast(device_type=self._amp_device, enabled=self._use_amp),
        ):
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
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self._amp_device = device.type
        self._use_amp = bool(self.amp) and device.type == "cuda"
        self._ema = EMA(model, self.ema_decay) if self.ema_decay is not None else None
        kl_target: float | None = None
        if self.kl_warmup_epochs > 0 and model.kl_weight is not None:
            kl_target = float(model.kl_weight)
            model.kl_weight = 0.0
        optimiser = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scaler = torch.amp.GradScaler(device.type, enabled=self._use_amp)
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
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        stopped_epoch: int | None = None

        logger.info(
            "Training %s on %s for %d epochs (%d steps/epoch, batch=%d, "
            "scheduler=%s, val=%s, device=%s, prefetch=%s, light_batch=%s, "
            "amp=%s, ema=%s, kl_warmup=%d)",
            type(model).__name__,
            type(train_data).__name__,
            self.max_epochs,
            self.steps_per_epoch,
            self.batch_size,
            self.scheduler_name,
            val_batch is not None,
            device,
            self.prefetch,
            self.light_batch,
            self._use_amp,
            self.ema_decay,
            self.kl_warmup_epochs,
        )

        profiler = PhaseProfiler(enabled=self.profile, device=device)
        self.profiler = profiler
        phase = profiler.phase if self.profile else _null_phase
        move_keys = TRAIN_BATCH_KEYS if self.light_batch else None

        self._iterators.pop(id(train_data), None)
        prefetcher: _BatchPrefetcher | None = None
        if self.prefetch:
            prefetcher = _BatchPrefetcher(lambda: self._draw_batch(train_data))

        def draw() -> Batch:
            return prefetcher.next() if prefetcher is not None else self._draw_batch(train_data)

        try:
            for epoch in range(self.max_epochs):
                if kl_target is not None:
                    # beta warm-up: ramp the KL weight linearly to its target.
                    fraction = min(1.0, (epoch + 1) / self.kl_warmup_epochs)
                    model.kl_weight = kl_target * fraction
                terms: dict[str, list[float]] = defaultdict(list)
                for _ in range(self.steps_per_epoch):
                    with phase("data"):
                        batch = move_batch(draw(), device, keys=move_keys)
                    with (
                        phase("forward"),
                        torch.autocast(device_type=device.type, enabled=self._use_amp),
                    ):
                        outputs = model.forward(batch)
                    with phase("loss"):
                        losses = model.loss(outputs, batch)
                        loss = losses["loss"]
                    if not torch.isfinite(loss):
                        logger.warning("Non-finite loss at epoch %d; skipping step.", epoch)
                        optimiser.zero_grad(set_to_none=True)
                        profiler.end_step()
                        continue
                    with phase("backward"):
                        optimiser.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                        if self.grad_clip is not None:
                            scaler.unscale_(optimiser)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    with phase("optim"):
                        scaler.step(optimiser)
                        scaler.update()
                        if isinstance(
                            scheduler,
                            (
                                torch.optim.lr_scheduler.LambdaLR,
                                torch.optim.lr_scheduler.StepLR,
                            ),
                        ):
                            scheduler.step()
                    if self._ema is not None:
                        self._ema.update(model)
                    with phase("sync"):
                        names = list(losses)
                        try:
                            values = torch.stack(
                                [losses[name].detach() for name in names]
                            ).cpu().tolist()
                        except (RuntimeError, TypeError):
                            values = [float(losses[name].detach().cpu()) for name in names]
                        for name, value in zip(names, values, strict=True):
                            terms[name].append(float(value))
                    profiler.end_step()

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
                    if isinstance(scheduler, AdaptiveScheduler):
                        # Warm up every epoch; decay only on fresh validation
                        # (or, when there is no validation, on the train loss).
                        scheduler.step(
                            val_loss
                            if val_loss is not None
                            else (None if val_batch is not None else history["loss"][-1])
                        )
                    elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(
                            val_loss if val_loss is not None else history["loss"][-1]
                        )
                    else:
                        scheduler.step()

                if self.log_every and (
                    epoch % self.log_every == 0 or epoch == self.max_epochs - 1
                ):
                    summary = ", ".join(f"{k}={v[-1]:.4f}" for k, v in history.items() if v)
                    logger.info("epoch %d/%d | %s", epoch + 1, self.max_epochs, summary)

                if val_loss is not None:
                    # "Very permissive" by default: only a clear plateau fires,
                    # and never before ``early_stopping_patience`` epochs without
                    # a real (absolute + relative) improvement over the best.
                    threshold = self.early_stopping_min_delta + (
                        self.early_stopping_min_delta_ratio * abs(best_val)
                        if math.isfinite(best_val)
                        else 0.0
                    )
                    if (best_val - val_loss) > threshold:
                        best_val = val_loss
                        best_epoch = epoch
                        if self.restore_best_weights:
                            best_state = {
                                key: value.detach().clone()
                                for key, value in model.state_dict().items()
                            }
                    elif (
                        self.early_stopping_patience is not None
                        and epoch - best_epoch >= self.early_stopping_patience
                    ):
                        stopped_epoch = epoch + 1
                        logger.info(
                            "Early stopping at epoch %d (best val_loss %.6f, no "
                            "improvement for %d epochs).",
                            stopped_epoch,
                            best_val,
                            self.early_stopping_patience,
                        )
                        break
        finally:
            if prefetcher is not None:
                prefetcher.close()

        if self.restore_best_weights and best_state is not None:
            model.load_state_dict(best_state)
        if self._ema is not None:
            # The averaged weights are the model that was evaluated, so make them
            # the final state (this takes precedence over restore_best_weights).
            self._ema.copy_to(model)
        if kl_target is not None:
            model.kl_weight = kl_target
        self.best_val = best_val if math.isfinite(best_val) else None
        self.best_epoch = best_epoch
        self.stopped_epoch = stopped_epoch
        if self.profile:
            profiler.report(title=f"{type(model).__name__} step profile")
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
        iterator = self._iterators.get(id(data))
        if iterator is None:
            iterator = iter(data)
            self._iterators[id(data)] = iterator
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data)
            self._iterators[id(data)] = iterator
            batch = next(iterator)
        if not isinstance(batch, Batch):
            raise TypeError("Data loader must yield gcmts.typing.Batch objects.")
        return batch
