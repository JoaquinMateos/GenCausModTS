"""Tests for graceful interrupt handling in the full-benchmark orchestrator."""

from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path


def _load_orchestrator():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "run_full_benchmark", scripts / "run_full_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_full_benchmark"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_terminate_children_kills_process_group() -> None:
    module = _load_orchestrator()
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    module._register_proc(proc)
    try:
        killed = module._terminate_children(signal.SIGKILL)
        assert killed == 1
        assert proc.wait(timeout=5) is not None
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        module._unregister_proc(proc)


def test_run_step_skips_when_stopped(tmp_path: Path) -> None:
    module = _load_orchestrator()
    module._STOP.set()
    try:
        result = module._run_step(
            module.Step("x", [sys.executable, "-c", "print(1)"]), 0, [], 0, {}, tmp_path
        )
    finally:
        module._STOP.clear()
    assert result["exit"] == -2


def test_run_step_captures_output(tmp_path: Path) -> None:
    module = _load_orchestrator()
    result = module._run_step(
        module.Step("ok", [sys.executable, "-c", "print('hello')"]), 0, [], 0, {}, tmp_path
    )
    assert result["exit"] == 0
    log = next(tmp_path.glob("*.log"))
    assert "hello" in log.read_text(encoding="utf-8")
