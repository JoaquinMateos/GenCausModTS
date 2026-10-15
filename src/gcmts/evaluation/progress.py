"""Run-progress reporting for the benchmark scripts and the orchestrator.

Every benchmark process owns exactly one progress file
(``<out_dir>/<stem>_progress.json``) recording how many runs are done out of how
many are planned. The full-benchmark orchestrator polls these files to render a
live dashboard and estimate the remaining time, and the optional ClearML
integration mirrors the same numbers to the web UI.

The file is written atomically (temp file + :func:`os.replace`) so a poller never
observes a half-written file. This module uses the standard library only;
``rich`` is used by the orchestrator, not here.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ProgressSnapshot", "ProgressTracker", "read_progress", "format_duration"]

logger = logging.getLogger("gcmts.progress")


def format_duration(seconds: float | None) -> str:
    """Human-readable ``1d 02h`` / ``03h 12m`` / ``12m 05s`` / ``45s``."""
    if seconds is None:
        return "--"
    seconds = int(max(seconds, 0))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@dataclass(frozen=True)
class ProgressSnapshot:
    """Immutable view of one ``*_progress.json`` file."""

    stem: str
    label: str
    total: int
    done: int
    errors: int
    started_at: float
    updated_at: float
    finished: bool

    @property
    def fraction(self) -> float:
        return 1.0 if self.total <= 0 else min(self.done / self.total, 1.0)

    @property
    def elapsed(self) -> float:
        return max(self.updated_at - self.started_at, 0.0)

    @property
    def rate(self) -> float | None:
        """Completed runs per second, or ``None`` before the first run."""
        if self.done <= 0 or self.elapsed <= 0:
            return None
        return self.done / self.elapsed

    @property
    def eta_seconds(self) -> float | None:
        if self.finished or self.total <= 0 or self.done <= 0:
            return None
        rate = self.rate
        return None if rate is None else max(self.total - self.done, 0) / rate


def read_progress(path: str | Path) -> ProgressSnapshot | None:
    """Read a progress file, returning ``None`` if missing or malformed."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return ProgressSnapshot(
            stem=str(raw.get("stem", "unknown")),
            label=str(raw.get("label", "")),
            total=int(raw.get("total", 0)),
            done=int(raw.get("done", 0)),
            errors=int(raw.get("errors", 0)),
            started_at=float(raw.get("started_at", 0.0)),
            updated_at=float(raw.get("updated_at", 0.0)),
            finished=bool(raw.get("finished", False)),
        )
    except (TypeError, ValueError):
        return None


class ProgressTracker:
    """Write-through progress counter backed by a JSON file.

    Args:
        path: where to write ``*_progress.json``.
        label: human-readable step name (logged / shown in the dashboard).
        total: number of runs (method x seed) this process will execute.
        log_every: emit one log line every this many runs; ``None`` auto-selects
            about ten lines for the whole run, ``0`` disables logging.
    """

    def __init__(
        self,
        path: str | Path,
        label: str,
        total: int,
        *,
        log_every: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.label = label
        self.total = max(int(total), 0)
        if log_every is None:
            log_every = max(1, self.total // 10) if self.total else 0
        self._log_every = max(int(log_every), 0)
        self.stem = self.path.name.removesuffix("_progress.json")
        self.done = 0
        self.errors = 0
        self.finished = False
        self.started_at = time.time()
        self.updated_at = self.started_at
        self._last_logged = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> ProgressTracker:
        self.started_at = self.updated_at = time.time()
        self._write()
        logger.info("[progress] %s: 0/%d runs", self.label, self.total)
        return self

    def advance(self, n: int = 1) -> ProgressTracker:
        self.done += n
        self._touch()
        return self

    def fail(self, n: int = 1) -> ProgressTracker:
        """Count a run that finished unsuccessfully (still counts as done)."""
        self.done += n
        self.errors += n
        self._touch()
        return self

    def finish(self, *, status: str = "finished") -> ProgressTracker:
        self.finished = status == "finished"
        self.updated_at = time.time()
        self._write()
        rate = self.rate
        if rate is not None:
            logger.info(
                "[progress] %s: %d/%d runs in %s (%s%s)",
                self.label,
                self.done,
                self.total,
                format_duration(self.elapsed),
                f"{self.errors} failed, " if self.errors else "",
                f"{rate:.2f} runs/s",
            )
        return self

    # -- intro ------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return max(self.updated_at - self.started_at, 0.0)

    @property
    def rate(self) -> float | None:
        if self.done <= 0 or self.elapsed <= 0:
            return None
        return self.done / self.elapsed

    @property
    def eta_seconds(self) -> float | None:
        rate = self.rate
        if rate is None or self.total <= 0:
            return None
        return max(self.total - self.done, 0) / rate

    def snapshot(self) -> ProgressSnapshot:
        return ProgressSnapshot(
            stem=self.stem,
            label=self.label,
            total=self.total,
            done=self.done,
            errors=self.errors,
            started_at=self.started_at,
            updated_at=self.updated_at,
            finished=self.finished,
        )

    # -- internals --------------------------------------------------------
    def _touch(self) -> None:
        self.updated_at = time.time()
        self._write()
        if self._log_every and self.done - self._last_logged >= self._log_every:
            self._last_logged = self.done
            logger.info(
                "[progress] %s: %d/%d runs (%s elapsed, ~%s left)",
                self.label,
                self.done,
                self.total,
                format_duration(self.elapsed),
                format_duration(self.eta_seconds),
            )

    def _write(self) -> None:
        payload: dict[str, Any] = {
            "stem": self.stem,
            "label": self.label,
            "total": self.total,
            "done": self.done,
            "errors": self.errors,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished": self.finished,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as error:  # pragma: no cover - disk issues only
            logger.debug("Could not write progress file %s: %s", self.path, error)
