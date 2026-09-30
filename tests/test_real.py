"""Tests for the real-time-series loaders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcmts.data.real import RealTimeSeries, RealTSGenerator


def _write_csv(path):
    n = 500
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "a": np.sin(np.linspace(0, 20, n)) + np.random.default_rng(0).normal(0, 0.01, n),
            "b": np.cos(np.linspace(0, 20, n)),
        }
    )
    frame.to_csv(path, index=False)


def test_real_time_series_windowing_and_split(tmp_path):
    path = tmp_path / "series.csv"
    _write_csv(path)
    series = RealTimeSeries(
        path, horizon=16, train_frac=0.6, val_frac=0.2, max_windows=None, seed=0
    )
    assert series.observed_dim == 2
    train = series.sample(8, "train")
    assert train.x.shape == (8, 16, 2)
    assert train.z is None
    assert train.x.std().item() > 0.5  # standardised, not degenerate

    generator = RealTSGenerator(series)
    batch = generator.sample(4)
    assert batch.x.shape == (4, 16, 2)
