"""Smoke test for the six new branch representatives runner."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_new_representatives.py"
METRICS = ("improvement", "recourse_cost", "budget_violation", "fairness_gap", "cf_mae")


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_new_representatives", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_representatives_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GCMTS_PREFETCH", "0")
    out = tmp_path / "new_representatives.json"
    runner = _load_runner()
    assert (
        runner.main(
            [
                "--device",
                "cpu",
                "--seeds",
                "1",
                "--epochs",
                "1",
                "--methods",
                "SCARF",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    assert row["case"] == "SCARF"
    assert row["dataset"] == "sequential_recourse"
    assert row["seed"] == 0
    assert math.isfinite(row["seconds"])
    for key in METRICS:
        assert key in row, key
        assert math.isfinite(float(row[key])), (key, row[key])
    assert (tmp_path / "new_representatives_agg.csv").exists()
