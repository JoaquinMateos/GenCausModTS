"""Real-world time-series datasets for external-validity evaluation.

Real series have no ground-truth latent variables, so they cannot certify
identifiability (rule R1). They are used instead to check *external validity*:
whether a representation learner trained without supervision supports
forecasting, and whether it remains accurate on a later (out-of-distribution)
time period. The loaders window a multivariate series into fixed-length
sequences and standardise each channel with **training-split** statistics only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from gcmts.data.base import BaseDataGenerator
from gcmts.typing import Batch

__all__ = ["RealTimeSeries", "RealTSGenerator"]


class RealTimeSeries:
    """Windowed real multivariate series with a chronological train/val/test split.

    Args:
        path: CSV file; the first column is a timestamp and is dropped.
        channels: channel names to keep (default: all numeric columns).
        horizon: window length.
        stride: step between consecutive windows.
        subsample: keep every ``subsample``-th row (e.g. to make a 10-min series hourly).
        standardize: z-score each channel with training-split mean/std.
        train_frac / val_frac: chronological split fractions.
        max_windows: cap the number of windows per split (uniform subsampling).
        seed: RNG for window subsampling and ``sample`` draws.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        channels: list[str] | None = None,
        horizon: int = 24,
        stride: int = 1,
        subsample: int = 1,
        standardize: bool = True,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        max_windows: int | None = 20000,
        seed: int = 0,
    ) -> None:
        frame = pd.read_csv(path)
        frame = frame.drop(columns=[frame.columns[0]])
        frame = frame.select_dtypes(include=[np.number])
        if channels is not None:
            frame = frame[channels]
        values = frame.to_numpy(dtype=np.float32)
        if subsample > 1:
            values = values[::subsample]

        n = values.shape[0]
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        self.channels = list(frame.columns)
        self.observed_dim = values.shape[1]
        self.horizon = horizon
        self.rng = torch.Generator().manual_seed(seed)

        if standardize:
            mean = values[:train_end].mean(axis=0, keepdims=True)
            std = values[:train_end].std(axis=0, keepdims=True) + 1e-6
            values = (values - mean) / std

        self.windows: dict[str, Tensor] = {}
        for name, (start, end) in {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, n),
        }.items():
            segment = values[start:end]
            if segment.shape[0] <= horizon:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(
                segment, horizon, axis=0
            )  # (n-h+1, D, horizon)
            windows = np.transpose(windows, (0, 2, 1))[::stride]  # (N, horizon, D)
            if max_windows is not None and windows.shape[0] > max_windows:
                idx = np.linspace(0, windows.shape[0] - 1, max_windows).astype(int)
                windows = windows[idx]
            self.windows[name] = torch.from_numpy(np.ascontiguousarray(windows))

    def sample(self, n_samples: int, split: str = "train", **_: Any) -> Batch:
        """Draw ``n_samples`` random windows from a split."""
        pool = self.windows[split]
        indices = torch.randint(0, pool.shape[0], (n_samples,), generator=self.rng)
        return Batch(x=pool[indices], context={"split": torch.full((n_samples,), 0.0)})

    def __len__(self) -> int:
        return int(self.windows["train"].shape[0])


class RealTSGenerator(BaseDataGenerator):
    """Adapter exposing a :class:`RealTimeSeries` split as a trainer generator."""

    def __init__(self, series: RealTimeSeries, *, split: str = "train") -> None:
        super().__init__(observed_dim=series.observed_dim, horizon=series.horizon)
        self.series = series
        self.split = split

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: Any) -> Batch:
        return self.series.sample(n_samples, split=self.split)
