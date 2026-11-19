"""Aggregation and persistence helpers for benchmark result rows."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
import time as _time
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_rows",
    "atomic_write_text",
    "code_revision",
    "distinct_seeds",
    "exact_seed_status",
    "guard_resume",
    "load_rows",
    "paired_wilcoxon",
    "save_rows",
    "training_code_digest",
    "wall_seconds",
]

_NON_METRIC = {"epochs", "n_parameters", "seconds", "seed"}

#: Bumped whenever the resume fingerprint changes meaning.  A stored config
#: without this version is treated as unverified (see ``guard_resume``).
FINGERPRINT_VERSION = 2


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    backup: bool = False,
    encoding: str = "utf-8",
) -> Path:
    """Write ``text`` to ``path`` atomically (temp file + :func:`os.replace`).

    A crash or a full disk therefore leaves either the previous file or the
    new one, never a truncated mixture.  ``backup=True`` first copies the
    existing file to ``<name>.bak`` so a later parse failure can be recovered
    instead of silently restarting the run from zero rows.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if backup and target.exists():
        try:
            shutil.copy2(target, target.with_name(target.name + ".bak"))
        except OSError as error:  # pragma: no cover - disk issues only
            logger.warning("Could not back up %s: %s", target, error)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise
    return target


def code_revision() -> str:
    """Best-effort identifier of the code revision (git commit, else version)."""
    import subprocess

    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import gcmts

        return str(getattr(gcmts, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"


def training_code_digest() -> str:
    """SHA-256 over the *contents* of the training-relevant Python sources.

    The resume guard must react to a change in the code that produced the
    numbers, and to nothing else.  Hashing ``git HEAD`` (the previous
    behaviour) made an unrelated edit to a table script invalidate every resume
    and, worse, did not change when vendored training code was edited without
    a commit.  This digest covers every ``.py`` file under ``src/gcmts``
    (including the vendored reference implementations) by relative path and
    bytes, in sorted order.
    """
    return _training_code_digest()


@lru_cache(maxsize=1)
def _training_code_digest() -> str:
    import hashlib

    root = Path(__file__).resolve().parents[1]  # src/gcmts
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError as error:  # pragma: no cover - unreadable source
            digest.update(f"<unreadable:{error}>".encode())
        digest.update(b"\0")
    return digest.hexdigest()


def load_rows(out_dir: str | Path, stem: str) -> list[dict[str, Any]]:
    """Load previously written raw rows for ``stem``.

    A missing file is an empty result (first run).  A file that exists but
    cannot be parsed is *not* treated as empty: that would silently discard the
    completed runs and start over.  Callers get a :class:`ValueError` telling
    them to restore ``<stem>.json.bak`` or move the file aside.
    """
    path = Path(out_dir) / f"{stem}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unreadable result file {path}: {error}. Refusing to treat a "
            f"corrupt or truncated file as empty; restore it from "
            f"{path}.bak or move it aside before resuming."
        ) from error
    if not isinstance(data, list):
        raise ValueError(
            f"Result file {path} must contain a JSON list of rows, got "
            f"{type(data).__name__}."
        )
    return data


def distinct_seeds(
    rows: list[dict[str, Any]],
    case: str | None = None,
    *,
    seed_key: str = "seed",
) -> set[int]:
    """Distinct integer seeds present in ``rows`` (optionally for one case)."""
    selected = [row for row in rows if case is None or row.get("case") == case]
    return {
        int(row[seed_key])
        for row in selected
        if row.get(seed_key) is not None
    }


def exact_seed_status(
    rows: list[dict[str, Any]],
    case: str,
    expected: int,
    *,
    seed_key: str = "seed",
) -> tuple[bool, str]:
    """Check that ``case`` has exactly the seeds ``0 .. expected-1``, each once.

    Returns ``(ok, reason)``.  Unlike a row count this rejects a duplicated
    ``(case, seed)`` row (which would silently reweight the mean) and an
    unexpected extra seed (which would mix protocols).  Rows without a seed
    field are reported as unverified rather than assumed complete.
    """
    selected = [row for row in rows if row.get("case") == case]
    if not selected:
        return False, f"{case}: no rows"
    missing_seed = sum(1 for row in selected if row.get(seed_key) is None)
    if missing_seed:
        return False, f"{case}: {missing_seed} row(s) have no {seed_key}"
    values = [int(row[seed_key]) for row in selected]
    if len(values) != len(set(values)):
        return False, f"{case}: duplicated ({seed_key}) rows"
    observed = set(values)
    wanted = set(range(int(expected)))
    if observed != wanted:
        missing = sorted(wanted - observed)
        extra = sorted(observed - wanted)
        detail = []
        if missing:
            detail.append(f"missing seeds {missing[:5]}")
        if extra:
            detail.append(f"unexpected seeds {extra[:5]}")
        return False, f"{case}: " + ", ".join(detail)
    return True, ""


def _fingerprint(config: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(
        {
            "fingerprint_version": FINGERPRINT_VERSION,
            "training_code": training_code_digest(),
            "config": config,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def guard_resume(
    out_dir: str | Path,
    stem: str,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """Check that existing rows were produced by the same configuration.

    Writes ``<stem>_config.json`` (fingerprint + configuration + code revision)
    the first time, and on later runs compares against it. Returns
    ``(ok, reason)``; ``ok=False`` means the caller must refuse to append (the
    training configuration or code revision changed) unless ``force`` is set.
    """
    path = Path(out_dir) / f"{stem}_config.json"
    fingerprint = _fingerprint(config)
    current = {
        "fingerprint": fingerprint,
        "fingerprint_version": FINGERPRINT_VERSION,
        "training_code": training_code_digest(),
        "code_revision": code_revision(),
        "config": config,
    }
    if os.environ.get("GCMTS_FORCE", "0").strip().lower() not in {"", "0", "false", "no"}:
        force = True
    rows_path = Path(out_dir) / f"{stem}.json"
    if not path.exists():
        if rows_path.exists() and not force:
            return False, (
                "existing rows were written before the resume guard, so their "
                "configuration and code revision are unknown"
            )
        atomic_write_text(path, json.dumps(current, indent=2, default=str))
        return True, ""
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        # Fail closed: an unreadable guard file means the provenance of the
        # existing rows cannot be verified.  Appending would mix protocols.
        if not force:
            return False, (
                f"the stored resume config {path} is unreadable ({error}); "
                "refusing to append unverified rows. Restore it, delete it "
                "together with the rows, or pass --force."
            )
        logger.warning(
            "Forced resume of %s across an unreadable resume config; "
            "rewriting it. Rows may mix protocols.", stem
        )
        atomic_write_text(path, json.dumps(current, indent=2, default=str))
        return True, ""
    if not isinstance(stored, dict):
        if not force:
            return False, (
                f"the stored resume config {path} is malformed; refusing to "
                "append unverified rows. Delete it and the rows, or pass --force."
            )
        logger.warning(
            "Forced resume of %s across a malformed resume config; rewriting "
            "it. Rows may mix protocols.", stem
        )
        atomic_write_text(path, json.dumps(current, indent=2, default=str))
        return True, ""
    stored_version = stored.get("fingerprint_version")
    if stored_version != FINGERPRINT_VERSION:
        if not force:
            return False, (
                f"the stored resume config {path} predates the content-based "
                f"training-code fingerprint (version {stored_version}); its "
                "provenance cannot be verified. Re-run with --force to accept "
                "the change (rows may mix protocols), or delete the rows."
            )
        logger.warning(
            "Forced resume of %s across a fingerprint-format change; "
            "rewriting the stored config. Rows may mix protocols.", stem
        )
        atomic_write_text(path, json.dumps(current, indent=2, default=str))
        return True, ""
    if stored.get("fingerprint") != fingerprint:
        if not force:
            return False, (
                "existing rows were produced by a different configuration or "
                f"training code (stored code={stored.get('code_revision')}, "
                f"current code={current['code_revision']})"
            )
        # A forced resume accepts the change; record the configuration that is
        # actually being run, otherwise the stored metadata (and every caption
        # derived from it) describes the previous run, not this one.
        logger.warning(
            "Forced resume of %s across a configuration change; rewriting the "
            "stored config. Rows may mix protocols.", stem
        )
        atomic_write_text(path, json.dumps(current, indent=2, default=str))
    return True, ""


def wall_seconds(start: float) -> float:
    """Wall-clock seconds since ``start`` (a ``time.perf_counter()`` reading).

    Returns ``NaN`` when ``GCMTS_TIMING=0``. Use that to suppress the per-run
    ``seconds`` field when many workers run concurrently on one machine: their
    wall-clock times are then inflated by CPU contention and are no longer
    comparable across methods, or to a sequential run on an idle machine.
    """
    if os.environ.get("GCMTS_TIMING", "1") == "0":
        return float("nan")
    return _time.perf_counter() - start


def paired_wilcoxon(
    rows: list[dict[str, Any]],
    case_a: str,
    case_b: str,
    metric: str,
) -> float:
    """Paired Wilcoxon signed-rank p-value for ``case_a`` vs ``case_b`` on ``metric``.

    Rows are matched by their per-seed order of appearance, so both cases must
    have been evaluated under the same seed schedule.
    """
    from scipy.stats import wilcoxon

    values_a = [float(r[metric]) for r in rows if r["case"] == case_a and r.get(metric) is not None]
    values_b = [float(r[metric]) for r in rows if r["case"] == case_b and r.get(metric) is not None]
    n = min(len(values_a), len(values_b))
    if n < 2:
        return float("nan")
    if all(a == b for a, b in zip(values_a[:n], values_b[:n], strict=True)):
        return 1.0
    result = wilcoxon(values_a[:n], values_b[:n])
    return float(result.pvalue)


def aggregate_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-seed result rows into ``case -> metric -> {mean, std}``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)

    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for case, group in grouped.items():
        metrics: dict[str, dict[str, float]] = {}
        keys = [
            key
            for key, value in group[0].items()
            if isinstance(value, (int, float)) and key not in _NON_METRIC
        ]
        for key in keys:
            values = [
                float(row[key])
                for row in group
                if row.get(key) is not None and np.isfinite(float(row[key]))
            ]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    # ddof=1: the table reports the sample standard deviation
                    # over seeds/replications, not the population std of the
                    # realised runs.
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n": float(len(values)),
                }
        aggregated[case] = metrics
    return aggregated


def save_rows(
    rows: list[dict[str, Any]],
    out_dir: str | Path,
    *,
    stem: str,
) -> dict[str, Path]:
    """Write raw rows and their mean/std aggregation as JSON + CSV atomically.

    The previous raw file is copied to ``<stem>.json.bak`` before replacement,
    so a bad write (or an accidental overwrite by a restarted run) can be
    recovered.  Every file is written through
    :func:`atomic_write_text`, so a poller or a concurrent table build never
    observes a truncated JSON document.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{stem}.json"
    atomic_write_text(raw_path, json.dumps(rows, indent=2), backup=True)

    aggregated = aggregate_rows(rows)
    agg_path = out / f"{stem}_agg.json"
    atomic_write_text(agg_path, json.dumps(aggregated, indent=2))

    flat: list[dict[str, Any]] = []
    for case, metrics in aggregated.items():
        row: dict[str, Any] = {"case": case}
        for name, stats in metrics.items():
            row[f"{name}_mean"] = stats["mean"]
            row[f"{name}_std"] = stats["std"]
        flat.append(row)
    csv_path = out / f"{stem}_agg.csv"
    if flat:
        fieldnames = list(flat[0].keys())
        for row in flat:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)
        atomic_write_text(csv_path, buffer.getvalue())
    return {"raw": raw_path, "agg_json": agg_path, "agg_csv": csv_path}
