"""Optional ClearML experiment tracking (soft dependency).

Install the extra and point the code at a self-hosted server via the standard
ClearML environment variables (or a ``~/clearml.conf``)::

    uv sync --extra tracking
    export CLEARML_API_HOST=http://clearml.example.com:8008
    export CLEARML_WEB_HOST=http://clearml.example.com:8080
    export CLEARML_FILES_HOST=http://clearml.example.com:8081
    export CLEARML_API_ACCESS_KEY=...
    export CLEARML_API_SECRET_KEY=...

Every call is a no-op when ClearML is not installed, not configured, explicitly
disabled (``GCMTS_CLEARML=0``), or the server is unreachable, so a benchmark run
never depends on it. ``GCMTS_CLEARML_OFFLINE=1`` writes offline tasks for later
upload (``clearml-task``/``clearml-agent``).

One ClearML task per benchmark *step* (e.g. "Benchmark A.1: ..." or "CTEE static
IHDP"). Per-run metrics are logged as scalars with the method as the series and
the seed/replication as the iteration, so the web UI plots one line per method;
training curves are logged from the same histories the tables use.
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import re
import socket
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from gcmts.evaluation.progress import ProgressTracker

__all__ = ["StepTracker", "tracking_available", "clear_project"]

logger = logging.getLogger("gcmts.tracking")

DEFAULT_PROJECT = "GenCausModTS"
_MAX_CURVE_POINTS = 300


def clear_project() -> str:
    """ClearML project name (override with ``GCMTS_CLEARML_PROJECT``)."""
    return os.environ.get("GCMTS_CLEARML_PROJECT", DEFAULT_PROJECT)


def tracking_available() -> bool:
    """True when the ``clearml`` package can be imported."""
    return importlib.util.find_spec("clearml") is not None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _configured() -> bool:
    if os.environ.get("GCMTS_CLEARML", "1") == "0":
        return False
    if not tracking_available():
        return False
    if os.environ.get("CLEARML_API_HOST"):
        return True
    return _clearml_conf().exists()


def _clearml_conf() -> Path:
    return Path(os.environ.get("CLEARML_CONFIG_FILE", str(Path.home() / "clearml.conf")))


def _api_server_from_conf() -> str | None:
    """Best-effort read of ``api_server`` from ``clearml.conf`` (for probing)."""
    try:
        text = _clearml_conf().read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"^\s*api_server\s*[:=]\s*(\S+)", text, re.MULTILINE)
    return match.group(1).strip().rstrip(",") if match else None


def _reachable(timeout: float = 2.0) -> bool:
    """Cheap TCP probe of the API host so init never hangs silently."""
    host = os.environ.get("CLEARML_API_HOST") or _api_server_from_conf()
    if not host:
        return True
    parsed = urlparse(host if "://" in host else f"http://{host}")
    if not parsed.hostname:
        return True
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError as error:
        logger.warning(
            "ClearML server %s:%s unreachable (%s); tracking disabled.",
            parsed.hostname,
            port,
            error,
        )
        return False


class StepTracker:
    """ClearML task for one benchmark step; a no-op when tracking is off."""

    def __init__(self) -> None:
        self.enabled = False
        self.progress: ProgressTracker | None = None
        self._task: Any = None
        self._logger: Any = None
        self._degraded = False

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def start(
        cls,
        *,
        name: str,
        params: dict[str, Any] | None = None,
        tags: Iterable[str] = (),
        progress: ProgressTracker | None = None,
        project: str | None = None,
    ) -> StepTracker:
        tracker = cls()
        tracker.progress = progress
        if not _configured() or not _reachable():
            return tracker
        try:
            from clearml import Task

            if os.environ.get("GCMTS_CLEARML_OFFLINE") == "1":
                Task.set_offline(True)
            tracker._task = Task.init(
                project_name=project or clear_project(),
                task_name=name,
                tags=list(tags),
                output_uri=False,
                reuse_last_task_id=False,
            )
            if params:
                tracker._task.connect(dict(params))
            tracker._logger = tracker._task.get_logger()
            tracker.enabled = True
            logger.info("ClearML tracking enabled for '%s' (project %s).", name, clear_project())
        except Exception as error:  # noqa: BLE001 - never break the benchmark
            logger.warning("ClearML init failed; continuing without tracking: %s", error)
            tracker._task = None
        return tracker

    def log_run(
        self,
        *,
        method: str,
        metrics: dict[str, Any],
        iteration: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Log one run: one scalar per metric (series = method, iteration = seed)."""
        if not self.enabled:
            return
        try:
            if params:
                self._task.connect(dict(params))
            for key, value in metrics.items():
                number = _finite(value)
                if number is not None:
                    self._logger.report_scalar(
                        title=str(key), series=str(method), value=number, iteration=iteration
                    )
            self._sync_progress()
        except Exception as error:  # noqa: BLE001
            self._degrade(error)

    def log_curves(self, method: str, history: dict[str, list[float]]) -> None:
        """Log down-sampled training curves (per-epoch loss/val_loss/...)."""
        if not self.enabled:
            return
        try:
            for key, values in history.items():
                if not values:
                    continue
                stride = max(1, len(values) // _MAX_CURVE_POINTS)
                for epoch in range(0, len(values), stride):
                    number = _finite(values[epoch])
                    if number is not None:
                        self._logger.report_scalar(
                            title=str(key), series=str(method), value=number, iteration=epoch
                        )
        except Exception as error:  # noqa: BLE001
            self._degrade(error)

    def log_single(self, name: str, value: Any) -> None:
        if not self.enabled:
            return
        number = _finite(value)
        if number is None:
            return
        try:
            self._logger.report_single_value(name=str(name), value=number)
        except Exception as error:  # noqa: BLE001
            self._degrade(error)

    def close(self, *, status: str = "completed") -> None:
        if self.enabled and self._task is not None:
            with contextlib.suppress(Exception):
                self._sync_progress()
                self._task.set_status(status if _has_status(status) else "completed")
        self.enabled = False

    # -- internals ---------------------------------------------------------
    def _sync_progress(self) -> None:
        if not self.enabled or self._task is None or self.progress is None:
            return
        if self.progress.total <= 0:
            return
        percent = 100.0 * self.progress.done / self.progress.total
        with contextlib.suppress(Exception):  # older versions lack set_progress
            self._task.set_progress(percent)

    def _degrade(self, error: Exception) -> None:
        if not self._degraded:
            self._degraded = True
            logger.warning("ClearML logging failed; disabling tracking: %s", error)
        self.enabled = False


def _has_status(status: str) -> bool:
    try:
        from clearml import Task

        return status in Task.TaskStatusEnum.get_values()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False
