"""Tests for resuming interrupted benchmark runs."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from gcmts.evaluation.report import guard_resume, load_rows, save_rows


def test_load_rows_missing_and_roundtrip(tmp_path: Path) -> None:
    assert load_rows(tmp_path, "missing") == []
    rows = [{"case": "a", "seed": 0, "mcc": 0.5}]
    save_rows(rows, tmp_path, stem="s")
    assert load_rows(tmp_path, "s") == rows


def test_load_rows_malformed(tmp_path: Path) -> None:
    (tmp_path / "s.json").write_text("{not json", encoding="utf-8")
    assert load_rows(tmp_path, "s") == []


def test_guard_resume_same_config(tmp_path: Path) -> None:
    ok, reason = guard_resume(tmp_path, "s", {"a": 1, "b": "x"})
    assert ok and reason == ""
    ok, reason = guard_resume(tmp_path, "s", {"a": 1, "b": "x"})
    assert ok and reason == ""


def test_guard_resume_refuses_on_config_change(tmp_path: Path) -> None:
    guard_resume(tmp_path, "s", {"scheduler": "cosine"})
    ok, reason = guard_resume(tmp_path, "s", {"scheduler": "adaptive"})
    assert not ok and "different" in reason
    ok, _ = guard_resume(tmp_path, "s", {"scheduler": "adaptive"}, force=True)
    assert ok


def test_guard_resume_env_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    guard_resume(tmp_path, "s", {"a": 1})
    monkeypatch.setenv("GCMTS_FORCE", "1")
    ok, _ = guard_resume(tmp_path, "s", {"a": 2})
    assert ok


def test_guard_resume_refuses_unknown_provenance(tmp_path: Path) -> None:
    # Rows exist but no config file (written before the guard existed).
    save_rows([{"case": "a", "seed": 0}], tmp_path, stem="s")
    ok, reason = guard_resume(tmp_path, "s", {"a": 1})
    assert not ok and "before the resume guard" in reason
    ok, _ = guard_resume(tmp_path, "s", {"a": 1}, force=True)
    assert ok


def _load_run_crl_shared():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("run_crl_shared", scripts / "run_crl_shared.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_crl_shared"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _suite_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        suite="npvar", seeds=1, epochs_scale=1.0, device="cpu", batch_size=16,
        out_dir=tmp_path, baselines=False, data_source="fresh", pool_size=0,
        dataset_dir=tmp_path, scheduler="adaptive", early_stopping_patience=300,
        restore_best_weights=False, amp=False, ema_decay=0.0, kl_warmup_epochs=0,
        resume=True, force=False,
    )


def test_run_suite_skips_completed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_run_crl_shared()
    calls: list[tuple[str, int]] = []

    def fake_run(suite, method, factory, epochs, seed, device, batch_size, out_dir,
                 save_curves, **kwargs):  # noqa: ANN001, ANN003
        calls.append((method, seed))
        return module.Row(
            case=method, dataset=suite.name, suite=suite.name, method=method, seed=seed,
            mcc=0.1, r2_diag=0.1, r2_sep=0.1, dci=0.1, shd=None, triplet=None,
            forecast1=0.1, forecast4=0.1, seconds=0.1,
        )

    monkeypatch.setattr(module, "_run", fake_run)
    args = _suite_args(tmp_path)

    module.run_suite(args)
    assert len(calls) == len(module._methods_for(module.SUITES["npvar"], 1.0))
    first = len(calls)

    calls.clear()
    module.run_suite(args)
    assert calls == []  # everything already present -> nothing re-run
    assert first > 0
