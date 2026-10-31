"""Regression tests for the defects found in the ten-seed benchmark run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from gcmts.core import SimpleTrainer
from gcmts.core.base import BaseModel
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


class _ConstantSource:
    """Serves constant batches (the minimal ``sample`` contract)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self, n_samples: int, horizon: int | None = None, **kwargs: object) -> Batch:
        return Batch(x=torch.full((n_samples, 1, 1), self.value))


class _NaNLossModel(BaseModel):
    """Every training loss is NaN, as for a pool that contains non-finite values."""

    def __init__(self) -> None:
        super().__init__(observed_dim=1)
        self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.weight

    def loss(self, outputs: torch.Tensor, batch: Batch) -> dict[str, torch.Tensor]:
        return {"loss": (outputs * float("nan")).mean()}

    def identifiability_statement(self) -> str:
        return "nan"


def test_trainer_raises_after_sustained_nonfinite_losses() -> None:
    from gcmts.core.trainer import TrainingDivergedError

    trainer = SimpleTrainer(
        max_epochs=3, lr=0.1, batch_size=4, steps_per_epoch=4, val_size=0,
        scheduler="none", log_every=0, seed=0,
    )
    trainer.max_consecutive_nonfinite = 5
    source = _ConstantSource(1.0)
    with pytest.raises(TrainingDivergedError, match="consecutive non-finite"):
        trainer.fit(_NaNLossModel(), source)


def test_trainer_raises_when_validation_is_never_finite() -> None:
    from gcmts.core.trainer import TrainingDivergedError

    class _FiniteTrainModel(_NaNLossModel):
        def loss(self, outputs: torch.Tensor, batch: Batch) -> dict[str, torch.Tensor]:
            return {"loss": ((outputs - batch.x.mean()) ** 2).mean()}

    trainer = SimpleTrainer(
        max_epochs=4, lr=0.01, batch_size=4, steps_per_epoch=2, val_size=4, val_every=1,
        scheduler="none", log_every=0, seed=0,
    )
    with pytest.raises(TrainingDivergedError, match="validation loss was never finite"):
        trainer.fit(_FiniteTrainModel(), _ConstantSource(1.0), _ConstantSource(float("nan")))


def test_non_finite_fields_flags_only_bad_tensors() -> None:
    from gcmts.data.synthetic_store import non_finite_fields

    batch = Batch(x=torch.full((2, 3, 2), float("nan")), z=torch.zeros(2, 3, 1))
    assert non_finite_fields(batch) == ["x"]
    assert non_finite_fields(Batch(x=torch.zeros(2, 3, 2), z=torch.zeros(2, 3, 1))) == []


def test_pool_loader_rejects_stored_nonfinite_pool(tmp_path: Path) -> None:
    from gcmts.data.synthetic_store import save_dataset

    sys.path.insert(0, str(_SCRIPTS))
    import run_crl_shared

    good = Batch(x=torch.zeros(4, 3, 2), z=torch.zeros(4, 3, 1))
    bad = Batch(x=torch.full((4, 3, 2), float("nan")), z=torch.zeros(4, 3, 1))
    save_dataset(
        tmp_path / "unit" / "seed_00.pt", good, bad, suite="unit", seed=0,
        max_lag=1, generator="test",
    )
    with pytest.raises(ValueError, match="non-finite"):
        run_crl_shared._load_pool(tmp_path, "unit", 0)


def test_run_failure_rule_requires_one_finite_metric() -> None:
    from types import SimpleNamespace

    sys.path.insert(0, str(_SCRIPTS))
    import run_crl_shared

    nan = float("nan")
    diverged = SimpleNamespace(mcc=nan, r2_diag=nan, r2_sep=nan, forecast1=nan,
                               forecast4=nan, w2=None)
    cegen_like = SimpleNamespace(mcc=nan, r2_diag=nan, r2_sep=nan, forecast1=1.3,
                                 forecast4=2.4, w2=0.001)
    assert not run_crl_shared._has_finite_metric(diverged)
    assert run_crl_shared._has_finite_metric(cegen_like)


def test_news_replication_count_is_explicit(tmp_path: Path) -> None:
    from gcmts.data.canonical_static import news_replications_available

    for seed in (1, 2, 3):
        for suffix in ("x", "y"):
            (tmp_path / f"topic_doc_mean_n5000_k3477_seed_{seed}.csv.{suffix}").write_text("")
    # A gap at seed 4 ends the contiguous count, so seed 5 is not counted.
    (tmp_path / "topic_doc_mean_n5000_k3477_seed_5.csv.x").write_text("")
    assert news_replications_available(tmp_path) == 3


def test_progress_crash_is_recorded_as_error(tmp_path: Path) -> None:
    import json

    from gcmts.evaluation.progress import ProgressTracker

    tracker = ProgressTracker(tmp_path / "unit_progress.json", "unit", 3).start()
    tracker.advance()
    tracker.crash()
    data = json.loads((tmp_path / "unit_progress.json").read_text())
    assert data["errors"] == 1
    assert data["finished"] is False
    assert data["done"] == 1


def test_crl_summary_is_nan_safe_and_names_pending_block(tmp_path: Path) -> None:
    import json

    sys.path.insert(0, str(_SCRIPTS))
    from make_paper_tables import write_crl_summary

    shared = tmp_path / "shared"
    shared.mkdir()
    rows = [
        {"case": m, "method": m, "dataset": "lotka_volterra", "seed": s,
         "mcc": float("nan"), "forecast1": float("nan")}
        for m in ("iVAE", "LEAP", "TDRL", "NCTRL", "MOSAIC") for s in range(2)
    ]
    (shared / "crl_shared_lotka_volterra.json").write_text(json.dumps(rows))
    latex = tmp_path / "latex"
    (latex / "pic" / "experiments").mkdir(parents=True)
    write_crl_summary(shared, latex, seeds=2)
    text = (latex / "pic" / "experiments" / "crl_summary.tex").read_text()
    assert "Pending blocks" in text and "A.9" in text


def test_generator_regenerates_stored_nonfinite_pool(tmp_path: Path) -> None:
    from gcmts.data.synthetic_store import save_dataset

    sys.path.insert(0, str(_SCRIPTS))
    from generate_synthetic_data import _existing_is_usable

    good = Batch(x=torch.zeros(4, 3, 2), z=torch.zeros(4, 3, 1))
    bad = Batch(x=torch.full((4, 3, 2), float("nan")), z=torch.zeros(4, 3, 1))
    save_dataset(tmp_path / "ok" / "seed_00.pt", good, good, suite="ok", seed=0,
                 max_lag=1, generator="test")
    save_dataset(tmp_path / "bad" / "seed_00.pt", good, bad, suite="bad", seed=0,
                 max_lag=1, generator="test")
    assert _existing_is_usable(tmp_path, "ok", 0)
    assert not _existing_is_usable(tmp_path, "bad", 0)
