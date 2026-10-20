"""Regression tests for the defects found in the ten-seed benchmark run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from gcmts.core.graph_ops import aligned_shd, shd
from gcmts.core.trainer import TRAIN_BATCH_KEYS
from gcmts.core.utils import hungarian_match, mcc
from gcmts.data.physics import PhysicsSCMGenerator
from gcmts.data.real_panels import load_nhefs
from gcmts.evaluation.metrics import (
    DCIDisentanglementMetric,
    R2DiagMetric,
    R2SepMetric,
)
from gcmts.typing import Batch

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_tdrl_blocks_sum_to_latent_dim() -> None:
    sys.path.insert(0, str(_SCRIPTS))
    from run_crl_shared import _tdrl_blocks

    for latent_dim in range(2, 10):
        fix, dyn, obs = _tdrl_blocks(latent_dim)
        assert min(fix, dyn, obs) >= 0
        assert fix + dyn + obs == latent_dim
        assert fix >= 1 and dyn >= 1
    assert _tdrl_blocks(2) == (1, 1, 0)


def test_representation_metrics_tolerate_dimension_mismatch() -> None:
    torch.manual_seed(0)
    for pred_dim, tgt_dim in ((3, 2), (2, 3), (4, 2)):
        pred = torch.randn(80, pred_dim)
        tgt = torch.randn(80, tgt_dim)
        for metric in (R2DiagMetric(), R2SepMetric(), DCIDisentanglementMetric()):
            value = metric(pred, tgt).value
            assert np.isfinite(value) or np.isnan(value)


def test_dci_returns_nan_for_non_finite_latents() -> None:
    from gcmts.evaluation.metrics import DCICompletenessMetric, DCIInformativenessMetric

    pred = torch.full((50, 2), float("nan"))
    tgt = torch.randn(50, 2)
    for metric in (
        DCIDisentanglementMetric(),
        DCICompletenessMetric(),
        DCIInformativenessMetric(),
    ):
        assert np.isnan(metric(pred, tgt).value)


def test_mcc_and_assignment_reject_non_finite() -> None:
    tgt = torch.randn(20, 2)
    assert np.isnan(mcc(tgt, torch.full((20, 3), float("nan"))))
    with pytest.raises(ValueError):
        hungarian_match(torch.full((3, 3), float("nan")))


def test_shd_and_aligned_shd_align_devices() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cpu = torch.eye(3)
    gpu = torch.eye(3, device="cuda")
    assert shd(cpu, gpu) == 0
    assert shd(gpu, cpu) == 0
    assert aligned_shd(cpu, gpu) == pytest.approx(0.0)


def test_lotka_volterra_generator_is_finite_and_bounded() -> None:
    generator = PhysicsSCMGenerator(
        system="lotka_volterra", observed_dim=8, horizon=16, n_regimes=3, seed=0
    )
    batch = generator.sample(1024)
    assert torch.isfinite(batch.x).all()
    assert torch.isfinite(batch.z).all()
    # The competitive dynamics must not saturate the safety clamp.
    assert batch.z.abs().max() < 100.0


def test_nhefs_loader_drops_incomplete_rows(tmp_path: Path) -> None:
    columns = [
        "sex", "age", "race", "education", "smokeintensity", "smokeyrs",
        "exercise", "active", "wt71", "qsmk", "wt82_71",
    ]
    rows = [
        [1, 40, 1, 12, 10, 20, 1, 2, 70.0, 1, 2.5],
        [0, 50, 0, 10, 20, 30, 0, 1, 80.0, 0, np.nan],       # missing outcome
        [1, 60, 1, 14, 5, 25, 2, 3, np.nan, 1, -1.0],        # missing covariate
        [0, 30, 0, 9, 30, 15, 1, 1, 65.0, 0, 4.0],
    ]
    path = tmp_path / "NHEFS.csv"
    import pandas as pd

    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    cohort = load_nhefs(path)
    assert cohort.x.shape[0] == 2
    assert np.isfinite(cohort.x).all()
    assert np.isfinite(cohort.y).all()
    assert set(np.unique(cohort.t)) == {0, 1}


def test_light_batch_keys_cover_static_dynamic_losses() -> None:
    assert {"outcome", "domain"} <= set(TRAIN_BATCH_KEYS)


def test_pool_finiteness_gate_rejects_nan() -> None:
    sys.path.insert(0, str(_SCRIPTS))
    from generate_synthetic_data import _assert_finite

    class _Suite:
        name = "unit"

    good = Batch(x=torch.zeros(4, 3, 2))
    _assert_finite(_Suite(), 0, "train", good)  # no error
    bad = Batch(x=torch.full((4, 3, 2), float("nan")))
    with pytest.raises(RuntimeError, match="Non-finite x"):
        _assert_finite(_Suite(), 0, "train", bad)
