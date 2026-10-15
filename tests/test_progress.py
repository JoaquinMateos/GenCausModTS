"""Tests for run-progress reporting and the optional tracking helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from gcmts.evaluation.progress import (
    ProgressTracker,
    format_duration,
    read_progress,
)
from gcmts.evaluation.tracking import StepTracker, clear_project, tracking_available


def test_format_duration() -> None:
    assert format_duration(None) == "--"
    assert format_duration(-5) == "0s"
    assert format_duration(9) == "9s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3600) == "1h 00m"
    assert format_duration(90000) == "1d 01h"


def test_progress_tracker_lifecycle(tmp_path: Path) -> None:
    tracker = ProgressTracker(tmp_path / "demo_progress.json", "Demo", 4, log_every=0).start()
    start = read_progress(tracker.path)
    assert start is not None
    assert (start.total, start.done, start.finished) == (4, 0, False)
    assert start.fraction == 0.0

    tracker.advance(2)
    tracker.advance()
    midpoint = read_progress(tracker.path)
    assert midpoint is not None
    assert (midpoint.done, midpoint.errors) == (3, 0)
    assert 0.0 < midpoint.fraction < 1.0

    tracker.fail()
    final = read_progress(tracker.path)
    assert final is not None
    assert (final.done, final.errors) == (4, 1)

    tracker.finish()
    done = tracker.snapshot()
    assert done.finished is True
    assert done.fraction == 1.0
    assert done.eta_seconds is None


def test_read_progress_missing_and_malformed(tmp_path: Path) -> None:
    assert read_progress(tmp_path / "absent_progress.json") is None
    bad = tmp_path / "bad_progress.json"
    bad.write_text("{not json", encoding="utf-8")
    assert read_progress(bad) is None
    wrong = tmp_path / "wrong_progress.json"
    wrong.write_text("[1, 2, 3]", encoding="utf-8")
    assert read_progress(wrong) is None


def test_eta_is_none_before_first_run(tmp_path: Path) -> None:
    tracker = ProgressTracker(tmp_path / "e_progress.json", "E", 10, log_every=0).start()
    snapshot = tracker.snapshot()
    assert snapshot.rate is None
    assert snapshot.eta_seconds is None


def test_step_tracker_is_noop_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force-disable regardless of whether the optional `clearml` extra is installed.
    monkeypatch.setenv("GCMTS_CLEARML", "0")
    tracker = ProgressTracker(tmp_path / "p_progress.json", "P", 2, log_every=0).start()
    step = StepTracker.start(name="noop", params={"a": 1}, progress=tracker)
    assert step.enabled is False
    # Every call must be a safe no-op.
    step.log_run(method="m", metrics={"mcc": 0.5}, iteration=0)
    step.log_curves("m", {"loss": [1.0, 0.5]})
    step.log_single("ratio", float("nan"))
    step.close(status="completed")


def test_project_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCMTS_CLEARML_PROJECT", "Custom")
    assert clear_project() == "Custom"
    assert isinstance(tracking_available(), bool)
