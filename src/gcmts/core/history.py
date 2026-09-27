"""Training-history persistence and curve plotting utilities.

The trainer returns a ``dict[str, list[float]]`` of per-epoch values. These
helpers write it to disk (JSON + CSV) and render the standard training-curve
figure used by the benchmarks.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["save_history", "plot_training_curves", "save_training_report"]

_INFO_KEYS = {"lr"}


def save_history(
    history: dict[str, list[float]],
    out_dir: str | Path,
    *,
    stem: str = "history",
) -> Path:
    """Write ``history`` as ``<stem>.json`` and ``<stem>.csv``; return the CSV path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    keys = list(history.keys())
    length = max((len(v) for v in history.values()), default=0)
    csv_path = out / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", *keys])
        for i in range(length):
            row: list[Any] = [i + 1]
            row.extend(history[k][i] if i < len(history[k]) else "" for k in keys)
            writer.writerow(row)
    return csv_path


def _plot_series(axis: Any, x: list[int], y: list[float], label: str, *, log: bool) -> None:
    finite = [
        (xi, yi)
        for xi, yi in zip(x[: len(y)], y, strict=False)
        if yi == yi
    ]
    if not finite:
        return
    xs, ys = zip(*finite, strict=False)
    use_log = log and all(value > 0 for value in ys)
    axis.plot(xs, ys, label=label, linewidth=1.5)
    if use_log:
        axis.set_yscale("log")


def plot_training_curves(
    history: dict[str, list[float]],
    out_dir: str | Path,
    *,
    title: str = "training",
    stem: str = "training_curves",
) -> Path:
    """Render loss, terms and learning rate curves; return the PNG path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    length = max((len(v) for v in history.values()), default=0)
    epochs = list(range(1, length + 1))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    loss_axis = axes[0]
    if "loss" in history:
        _plot_series(loss_axis, epochs, history["loss"], "loss", log=True)
    if "val_loss" in history:
        _plot_series(loss_axis, epochs, history["val_loss"], "val_loss", log=True)
    loss_axis.set_title("Objective")
    loss_axis.set_xlabel("epoch")

    term_axis = axes[1]
    for key, values in history.items():
        if key in _INFO_KEYS or key in {"loss", "val_loss"}:
            continue
        _plot_series(term_axis, epochs, values, key, log=True)
    term_axis.set_title("Loss terms")
    term_axis.set_xlabel("epoch")

    lr_axis = axes[2]
    if "lr" in history:
        _plot_series(lr_axis, epochs, history["lr"], "learning rate", log=True)
    lr_axis.set_title("Learning rate")
    lr_axis.set_xlabel("epoch")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()

    png_path = out / f"{stem}.png"
    figure.savefig(png_path, dpi=150)
    figure.savefig(out / f"{stem}.pdf")
    plt.close(figure)
    return png_path


def save_training_report(
    history: dict[str, list[float]],
    out_dir: str | Path,
    *,
    title: str = "training",
) -> dict[str, Path]:
    """Persist history (JSON+CSV) and the training-curve figure (PNG+PDF)."""
    return {
        "history_csv": save_history(history, out_dir),
        "curves_png": plot_training_curves(history, out_dir, title=title),
    }
