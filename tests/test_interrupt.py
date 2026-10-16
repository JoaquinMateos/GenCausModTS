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


def test_error_excerpt_traceback_and_error_lines(tmp_path: Path) -> None:
    module = _load_orchestrator()
    traced = tmp_path / "a.log"
    traced.write_text(
        "INFO starting\nWARNING something\nTraceback (most recent call last):\n"
        "  File x, line 1\nRuntimeError: boom\n",
        encoding="utf-8",
    )
    excerpt = module._error_excerpt(traced)
    assert excerpt[0].startswith("Traceback")
    assert "RuntimeError: boom" in excerpt[-1]

    noisy = tmp_path / "b.log"
    noisy.write_text("INFO ok\nERROR failed to load\nINFO done\n", encoding="utf-8")
    assert module._error_excerpt(noisy) == ["ERROR failed to load"]


def test_write_error_report_only_for_failures(tmp_path: Path) -> None:
    module = _load_orchestrator()
    step = module.Step("s", [sys.executable, "-c", "x"])
    ok_log = tmp_path / "ok.log"
    ok_log.write_text("INFO all good\n", encoding="utf-8")
    assert module._write_error_report(tmp_path, 0, step, 0, ok_log, interrupted=False) is None
    # SIGTERM during a user interrupt is not a fault
    assert module._write_error_report(tmp_path, 0, step, -15, ok_log, interrupted=True) is None

    fail_log = tmp_path / "fail.log"
    fail_log.write_text("INFO start\nTraceback (most recent call last):\nBoom\n", encoding="utf-8")
    path = module._write_error_report(tmp_path, 3, step, 1, fail_log, interrupted=False)
    assert path is not None and path.name == "03_s.error"
    text = path.read_text(encoding="utf-8")
    assert "exit:    1" in text and "Traceback" in text
