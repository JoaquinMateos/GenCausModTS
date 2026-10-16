"""One-command full benchmark orchestrator for the review.

Runs every experiment family (CRL shared suites, CRL sanity, CTEE temporal and
static, SDD, adjacent branches, and optionally the 18-problem grid), then
regenerates all LaTeX tables. Designed to be launched once on a CUDA server:

    uv run python scripts/run_full_benchmark.py --device cuda --seeds 10

Soft smoke run (fast, few seeds/epochs, small replications):

    uv run python scripts/run_full_benchmark.py --soft --device cuda

Parallelism
-----------
The individual runs train *small* models whose cost is dominated by CPU-side
data generation, so a single process leaves the GPU almost entirely idle
(typically ~1% utilisation). The benchmark families are independent, so this
orchestrator dispatches them to a bounded pool of worker processes that share
the GPU, keeping it fed instead of serialising everything behind one CPU-bound
process:

    uv run python scripts/run_full_benchmark.py --device cuda --seeds 10 --jobs 8

``--jobs 1`` reproduces the original strictly-sequential behaviour. On hosts
with several GPUs, ``--gpus 0,1`` pins the worker steps round-robin to the
listed devices (via ``CUDA_VISIBLE_DEVICES``). Each step still writes exactly
the same output files as before, so results are bit-for-bit identical to a
sequential run. The 18-problem grid is split into two balanced shards
(``crl_shared_grid_a`` / ``crl_shared_grid_b``, both read by
``make_paper_tables.py``); disable with ``--grid-shards 1``.

A live progress dashboard (Rich, when installed) reads the per-step
``*_progress.json`` files and shows runs done/left plus an approximate ETA, and
ClearML tracking is available with the optional ``tracking`` extra
(``uv sync --extra tracking``); see ``README.md`` for the self-hosted ClearML
environment variables. Disable the dashboard with ``--no-dashboard``.

Interrupting
------------
Each step runs in its own process group. The first ``Ctrl+C`` stops scheduling
new steps and sends ``SIGTERM`` to the running ones (graceful); a second
``Ctrl+C`` sends ``SIGKILL`` and aborts immediately. Either way no child is left
running, and because every runner is resumable you can relaunch the same command
to continue.

Everything is written under ``outputs/`` and ``outputs_shared/`` and the tables
into ``../latex/pic/experiments/``; a JSON summary and a log are written to
``outputs/full_benchmark_summary.json`` and ``outputs/full_benchmark.log`` while
each step's own stdout/stderr is captured under ``outputs/logs/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from gcmts.evaluation.benchmarks import (
    CRL_BENCHMARKS,
    CTEE_REAL_BENCHMARKS,
    CTEE_STATIC_BENCHMARKS,
    crl_label,
)
from gcmts.evaluation.progress import format_duration, read_progress

LOGGER = logging.getLogger("gcmts.full")

CRL_SUITES = list(CRL_BENCHMARKS)
CTEE_DATASETS = ["harmonic", "tumour", "linear_sem"]
REAL_PANELS = {
    "etth1": "ETTh1.csv",
    "etth2": "ETTh2.csv",
    "ettm1": "ETTm1.csv",
    "jena": "jena_climate_2009_2016.csv",
    "covid_policy": "covid_policy.csv",
    "exchange_rate": "exchange_rate.csv",
    "solar_al": "solar_al.csv",
    "electricity": "electricity.csv",
    "traffic": "traffic.csv",
    "beijing_pm25": "beijing_pm25.csv",
    "cesm2": "cesm2.csv",
}

# Stems already understood by ``make_paper_tables.write_stats_section``.
_GRID_SHARD_STEMS = {1: ("crl_shared_grid",), 2: ("crl_shared_grid_a", "crl_shared_grid_b")}

# Lines that mark a problem in a captured step log (used for the *.error files).
_ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|\bERROR\b|\bCRITICAL\b|"
    r"\bException\b|\w*Error\b|CUDA out of memory|"
    r"Segmentation fault|Killed|core dumped)",
    re.MULTILINE,
)
_ERRORS_LOCK = threading.Lock()


@dataclass
class Step:
    name: str
    command: list[str]
    barrier: bool = False  # run only after every non-barrier step has finished
    progress: Path | None = None  # *_progress.json written by the step


def _progress(out_dir: str, stem: str) -> Path:
    return Path(out_dir) / f"{stem}_progress.json"


def build_steps(args: argparse.Namespace) -> list[Step]:
    py = sys.executable
    device = args.device
    seeds = args.seeds
    epochs_scale = args.epochs_scale
    trainer_args = [
        "--scheduler", args.scheduler,
        "--early-stopping-patience", str(args.early_stopping_patience),
    ]
    if args.restore_best_weights:
        trainer_args.append("--restore-best-weights")
    if args.amp:
        trainer_args.append("--amp")
    if args.ema_decay > 0:
        trainer_args += ["--ema-decay", str(args.ema_decay)]
    if args.kl_warmup_epochs > 0:
        trainer_args += ["--kl-warmup-epochs", str(args.kl_warmup_epochs)]
    steps: list[Step] = []
    for suite in CRL_SUITES:
        if args.suites and suite not in args.suites:
            continue
        steps.append(
            Step(
                f"CRL {crl_label(suite)}",
                [py, "scripts/run_crl_shared.py", "--device", device, "--suite", suite,
                 "--seeds", str(seeds), "--epochs-scale", str(epochs_scale),
                 "--data-source", args.data_source, "--dataset-dir", str(args.dataset_dir),
                 *trainer_args,
                 "--out-dir", "outputs_shared"],
                progress=_progress("outputs_shared", f"crl_shared_{suite}"),
            )
        )
    if not args.suites:
        steps.append(
            Step(
                "CRL sanity (matching-benchmark)",
                [py, "scripts/run_crl_benchmarks.py", "--device", device,
                 "--seeds", str(seeds), "--out-dir", "outputs"],
                progress=_progress("outputs", "crl_benchmarks"),
            )
        )
        steps.append(
            Step(
                "CTEE temporal",
                [py, "scripts/run_ctee_benchmarks.py", "--device", device,
                 "--datasets", *CTEE_DATASETS, "--seeds", str(seeds), "--out-dir", "outputs"],
                progress=_progress("outputs", "ctee_benchmarks"),
            )
        )
        for dataset in CTEE_STATIC_BENCHMARKS:
            steps.append(
                Step(
                    f"CTEE static {dataset.upper()}",
                    [py, "scripts/run_ctee_static.py", "--device", device, "--dataset", dataset,
                     "--replications", str(args.replications), "--out-dir", "outputs"],
                    progress=_progress("outputs", f"ctee_static_{dataset}"),
                )
            )
        steps.append(
            Step(
                "SDD invariance",
                [py, "scripts/run_sdd_benchmarks.py", "--device", device,
                 "--seeds", str(seeds), "--out-dir", "outputs"],
                progress=_progress("outputs", "sdd_benchmarks"),
            )
        )
        steps.append(
            Step(
                "Adjacent branches + II.D",
                [py, "scripts/run_adjacent_benchmarks.py", "--device", device,
                 "--seeds", str(seeds), "--out-dir", "outputs"],
                progress=_progress("outputs", "adjacent_benchmarks"),
            )
        )
        for dataset in CTEE_REAL_BENCHMARKS:
            steps.append(
                Step(
                    f"CTEE real {dataset}",
                    [py, "scripts/run_ctee_static.py", "--device", device, "--dataset", dataset,
                     "--replications", "1", "--out-dir", "outputs"],
                    progress=_progress("outputs", f"ctee_static_{dataset}"),
                )
            )
        real_dir = Path("data/real")
        available = [name for name, fname in REAL_PANELS.items() if (real_dir / fname).exists()]
        if available:
            steps.append(
                Step(
                    "Real external validity",
                    [py, "scripts/run_real_benchmarks.py", "--device", device,
                     "--datasets", *available, "--seeds", str(seeds), "--out-dir", "outputs"],
                    progress=_progress("outputs", "real_benchmarks"),
                )
            )
    if args.with_grid:
        stems = _GRID_SHARD_STEMS[args.grid_shards]
        if args.grid_shards == 1:
            steps.append(
                Step(
                    "CRL 18-problem grid",
                    [py, "scripts/run_crl_shared.py", "--device", device, "--grid",
                     "--seeds", str(seeds), "--epochs-scale", str(epochs_scale),
                     "--data-source", args.data_source, "--dataset-dir", str(args.dataset_dir),
                     *trainer_args,
                     "--out-dir", "outputs_shared"],
                    progress=_progress("outputs_shared", "crl_shared_grid"),
                )
            )
        else:
            for index, stem in enumerate(stems):
                steps.append(
                    Step(
                        f"CRL 18-problem grid [{index + 1}/{args.grid_shards}]",
                        [py, "scripts/run_crl_shared.py", "--device", device, "--grid",
                         "--shard-index", str(index), "--shard-total", str(args.grid_shards),
                         "--stem", stem, "--seeds", str(seeds),
                         "--epochs-scale", str(epochs_scale),
                         "--data-source", args.data_source, "--dataset-dir", str(args.dataset_dir),
                         *trainer_args,
                         "--out-dir", "outputs_shared"],
                        progress=_progress("outputs_shared", stem),
                    )
                )
    steps.append(
        Step(
            "Regenerate LaTeX tables",
            [py, "scripts/make_paper_tables.py", "--shared-dir", "outputs_shared",
             "--outputs-dir", "outputs", "--latex-dir", "../latex"],
            barrier=True,
        )
    )
    return steps


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:80] or "step"


def _resolve_schedule(args: argparse.Namespace, n_steps: int) -> tuple[int, list[str]]:
    """Return ``(jobs, gpus)`` for this run."""
    cpu = os.cpu_count() or 4
    # CPU-bound workload: use the cores, leave headroom for the OS/GPU driver.
    auto_jobs = max(1, min(n_steps, cpu - 1 if cpu > 2 else cpu, 8))
    jobs = args.jobs if args.jobs > 0 else auto_jobs
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    return jobs, gpus


# -- graceful interrupt handling --------------------------------------------
# Each step runs in its own process group, so on Ctrl+C we can stop the whole
# tree (the script and any grandchildren). The first Ctrl+C stops scheduling new
# steps and sends SIGTERM (graceful); a second Ctrl+C sends SIGKILL and aborts.
_ACTIVE_PROCS: dict[int, subprocess.Popen[bytes]] = {}
_ACTIVE_LOCK = threading.Lock()
_STOP = threading.Event()
_INTERRUPTS = 0
_ORIGINAL_SIGINT: object = None


def _register_proc(proc: subprocess.Popen[bytes]) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS[proc.pid] = proc


def _unregister_proc(proc: subprocess.Popen[bytes]) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_PROCS.pop(proc.pid, None)


def _active_procs() -> list[subprocess.Popen[bytes]]:
    with _ACTIVE_LOCK:
        return [proc for proc in _ACTIVE_PROCS.values() if proc.poll() is None]


def _terminate_children(sig: int) -> int:
    """Send ``sig`` to every running step's process group; return the count."""
    procs = _active_procs()
    for proc in procs:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                proc.send_signal(sig)
    return len(procs)


def _install_signal_handlers() -> None:
    global _ORIGINAL_SIGINT

    def _handler(signum: int, frame: object) -> None:
        global _INTERRUPTS
        _INTERRUPTS += 1
        _STOP.set()
        active = len(_active_procs())
        if _INTERRUPTS == 1:
            LOGGER.warning(
                "Interrupt: no new steps will start and %d running step(s) are "
                "being stopped. Press Ctrl+C again to force-kill.",
                active,
            )
            _terminate_children(signal.SIGTERM)
        else:
            LOGGER.warning("Second interrupt: force-killing %d running step(s).", active)
            _terminate_children(signal.SIGKILL)
            raise KeyboardInterrupt

    _ORIGINAL_SIGINT = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)


def _restore_signal_handlers() -> None:
    if _ORIGINAL_SIGINT is not None:
        signal.signal(signal.SIGINT, _ORIGINAL_SIGINT)  # type: ignore[arg-type]


# -- error extraction for fast triage ---------------------------------------
def _error_excerpt(log_path: Path, *, limit: int = 80) -> list[str]:
    """Return the error-relevant lines of a captured step log.

    If the log contains a traceback, return from the first ``Traceback`` line to
    the end (an uncaught exception is the last thing a step prints); otherwise
    return the matching ``ERROR`` / ``*Error`` / ``Killed`` lines.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    for position, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            return lines[position:]
    return [line for line in lines if _ERROR_RE.search(line)][-limit:]


def _write_error_report(
    log_dir: Path,
    index: int,
    step: Step,
    returncode: int,
    log_path: Path,
    *,
    interrupted: bool,
) -> Path | None:
    """Write ``<index>_<step>.error`` when a step failed; return its path."""
    if returncode == -2:  # skipped because the run was interrupted
        return None
    if interrupted and returncode < 0:  # SIGTERM/SIGKILL from Ctrl+C, not a fault
        return None
    excerpt = _error_excerpt(log_path)
    has_traceback = bool(excerpt) and "Traceback (most recent call last)" in excerpt[0]
    if returncode == 0 and not has_traceback:
        return None
    body = excerpt or ["(no error lines found in the step log)"]
    header = [
        f"step:    {step.name}",
        f"command: {' '.join(step.command)}",
        f"exit:    {returncode}",
        f"log:     {log_path}",
        "-" * 72,
    ]
    error_path = log_dir / f"{index:02d}_{_slug(step.name)}.error"
    error_path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return error_path


def _record_error(log_dir: Path, step: Step, returncode: int, error_path: Path) -> None:
    """Append a one-line entry to ``<log_dir>/errors.log`` (thread-safe)."""
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}  exit={returncode:>4}  "
        f"{step.name}  ->  {error_path.name}\n"
    )
    with _ERRORS_LOCK, (log_dir / "errors.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def _run_step(
    step: Step,
    index: int,
    gpus: list[str],
    threads: int,
    extra_env: dict[str, str],
    log_dir: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if threads > 0:
        # Opt-in thread cap to limit oversubscription under high ``--jobs``.
        # Left untouched by default so runs stay numerically identical to a
        # sequential run (a different thread count can change FP reduction order).
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[key] = str(threads)
    env.update(extra_env)
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]

    log_path = log_dir / f"{index:02d}_{_slug(step.name)}.log"
    if _STOP.is_set():
        LOGGER.warning("SKIP %s (interrupted)", step.name)
        return {"step": step.name, "seconds": 0.0, "exit": -2, "error": None}
    LOGGER.info("RUN %s -> %s", step.name, log_path)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(step.command) + "\n\n")
        handle.flush()
        # Own process group so an interrupt can stop the step (and any children).
        proc = subprocess.Popen(
            step.command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        _register_proc(proc)
        try:
            returncode = proc.wait()
        finally:
            _unregister_proc(proc)
    elapsed = time.time() - start
    LOGGER.info("DONE %s in %.1fs (exit=%d)", step.name, elapsed, returncode)
    error_path = _write_error_report(
        log_dir, index, step, returncode, log_path, interrupted=_STOP.is_set()
    )
    if error_path is not None:
        _record_error(log_dir, step, returncode, error_path)
        LOGGER.error("Step %s failed (exit=%d); see %s", step.name, returncode, error_path)
    return {
        "step": step.name,
        "seconds": elapsed,
        "exit": returncode,
        "error": None if error_path is None else str(error_path),
    }


# -- live progress dashboard ------------------------------------------------
def _step_progress(steps: list[Step]) -> tuple[list[tuple[Step, object]], int, int]:
    """Return ``(rows, total_done, total_planned)`` from the progress files."""
    rows: list[tuple[Step, object]] = []
    total_done = total_planned = 0
    for step in steps:
        snapshot = read_progress(step.progress) if step.progress is not None else None
        rows.append((step, snapshot))
        if snapshot is not None:
            total_done += snapshot.done
            total_planned += snapshot.total
    return rows, total_done, total_planned


def _render_dashboard(steps: list[Step], started: float) -> object:
    from rich.console import Group
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    table = Table(title="GenCausModTS full benchmark", expand=True, pad_edge=False)
    table.add_column("Step", ratio=2, overflow="fold")
    table.add_column("Progress", ratio=3)
    table.add_column("Runs", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("ETA", justify="right")

    rows, total_done, total_planned = _step_progress(steps)
    for step, snapshot in rows:
        if snapshot is None:
            table.add_row(step.name, Text("queued", style="dim"), "-", "-", "-")
            continue
        bar = ProgressBar(total=max(snapshot.total, 1), completed=snapshot.done, width=None)
        runs = f"{snapshot.done}/{snapshot.total}"
        if snapshot.errors:
            runs += f" ({snapshot.errors} err)"
        eta = "done" if snapshot.finished else format_duration(snapshot.eta_seconds)
        table.add_row(
            step.name, bar, runs, format_duration(snapshot.elapsed), eta
        )

    elapsed = time.time() - started
    if total_planned and total_done:
        left = max(total_planned - total_done, 0)
        eta = elapsed / total_done * left
        footer = (
            f"Overall {total_done}/{total_planned} runs "
            f"({100 * total_done / total_planned:.0f}%) | elapsed {format_duration(elapsed)}"
            f" | ~{format_duration(eta)} left"
        )
    elif total_planned:
        footer = f"Overall 0/{total_planned} runs | elapsed {format_duration(elapsed)}"
    else:
        footer = f"Starting... elapsed {format_duration(elapsed)}"
    return Group(table, Text(footer, style="bold"))


def _plain_summary(steps: list[Step], started: float) -> str:
    rows, total_done, total_planned = _step_progress(steps)
    parts = [
        f"{snapshot.label or step.name}: {snapshot.done}/{snapshot.total}"
        for step, snapshot in rows
        if snapshot is not None
    ]
    head = (
        f"progress {total_done}/{total_planned} runs, "
        f"elapsed {format_duration(time.time() - started)}"
    )
    return head + (" | " + "; ".join(parts) if parts else "")


class _Dashboard:
    """Live ``rich`` dashboard (TTY) or periodic plain-text summary otherwise."""

    def __init__(
        self,
        steps: list[Step],
        log_path: Path,
        *,
        enabled: bool = True,
        interval: float = 2.0,
    ) -> None:
        self.steps = steps
        self.log_path = log_path
        self.enabled = enabled
        self.interval = interval
        self.started = time.time()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rich = False
        self._live: object | None = None
        self._saved_handlers: list[logging.Handler] = []

    def _rich_available(self) -> bool:
        try:
            import rich  # noqa: F401
        except ImportError:
            return False
        return sys.stderr.isatty()

    def start(self) -> None:
        if not self.enabled or not self.steps:
            return
        if self._rich_available():
            from rich.console import Console
            from rich.live import Live
            from rich.logging import RichHandler

            console = Console(stderr=True)
            root = logging.getLogger()
            self._saved_handlers = root.handlers[:]
            root.handlers = [
                RichHandler(console=console, show_path=False, rich_tracebacks=True),
                logging.FileHandler(self.log_path),
            ]
            self._live = Live(
                _render_dashboard(self.steps, self.started),
                console=console,
                refresh_per_second=1,
                transient=False,
            )
            self._live.start()
            self._rich = True
            self._thread = threading.Thread(target=self._rich_loop, daemon=True)
        else:
            self._thread = threading.Thread(target=self._plain_loop, daemon=True)
        self._thread.start()

    def _rich_loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._live.update(_render_dashboard(self.steps, self.started))  # type: ignore[attr-defined]

    def _plain_loop(self) -> None:
        while not self._stop.wait(max(self.interval, 30.0)):
            LOGGER.info("%s", _plain_summary(self.steps, self.started))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval, 30.0) + 1)
        if self._rich and self._live is not None:
            self._live.update(_render_dashboard(self.steps, self.started))  # type: ignore[attr-defined]
            self._live.stop()  # type: ignore[attr-defined]
            logging.getLogger().handlers = self._saved_handlers
        elif self.enabled and self.steps:
            LOGGER.info("%s (finished)", _plain_summary(self.steps, self.started))


@contextmanager
def _dashboard(
    steps: list[Step], log_path: Path, *, enabled: bool, interval: float
) -> Iterator[None]:
    dashboard = _Dashboard(steps, log_path, enabled=enabled, interval=interval)
    dashboard.start()
    try:
        yield
    finally:
        dashboard.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs-scale", type=float, default=1.0)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--soft", action="store_true", help="fast smoke run (few seeds/epochs)")
    parser.add_argument("--with-grid", action="store_true", help="also run the 18-problem grid")
    parser.add_argument("--suites", nargs="*", default=None, help="only these CRL suites")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument(
        "--jobs", type=int, default=0,
        help=(
            "max concurrent benchmark steps sharing the GPU "
            "(0 = auto from CPU count; 1 = sequential)"
        ),
    )
    parser.add_argument(
        "--threads", type=int, default=0,
        help=(
            "cap intra-op threads per worker (0 = leave the environment "
            "untouched, preserving sequential-equivalent numerics)"
        ),
    )
    parser.add_argument(
        "--gpus", default="1",
        help=(
            "comma-separated GPU ids to pin steps to round-robin, "
            "e.g. '0' or '0,1' (default: '1'; pass '' to inherit the environment)"
        ),
    )
    parser.add_argument(
        "--keep-timing", action="store_true",
        help=(
            "record per-run seconds even when steps run concurrently "
            "(the times are inflated by CPU contention; off by default)"
        ),
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="disable the live Rich progress dashboard (periodic plain summary only)",
    )
    parser.add_argument(
        "--dashboard-interval", type=float, default=2.0,
        help="dashboard refresh interval in seconds (default: 2)",
    )
    parser.add_argument(
        "--grid-shards", type=int, choices=(1, 2), default=2,
        help="split the 18-problem grid into 1 or 2 balanced shards (default: 2)",
    )
    parser.add_argument(
        "--data-source", choices=("fresh", "pool"), default="fresh",
        help=(
            "CRL training data: 'fresh' draws new SCM trajectories each step "
            "(standard protocol); 'pool' loads persisted fixed datasets from "
            "--dataset-dir (finite-pool protocol; generate with "
            "scripts/generate_synthetic_data.py)"
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "append to existing results even when the stored configuration or "
            "code revision changed (otherwise such steps refuse to resume)"
        ),
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/synthetic"),
        help="root of the persisted synthetic pools used with --data-source pool",
    )
    parser.add_argument(
        "--scheduler", default="adaptive",
        choices=("cosine", "adaptive", "plateau", "step", "none"),
        help="CRL learning-rate schedule (default: adaptive; 'cosine' = legacy)",
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=300,
        help="CRL epochs without improvement before stopping; 0 disables (default 300)",
    )
    parser.add_argument(
        "--restore-best-weights", action="store_true",
        help="restore the validation-best weights when a CRL run stops early",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="enable CUDA automatic mixed precision for the CRL steps",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.0,
        help="EMA decay for the CRL weights (0 disables; e.g. 0.99)",
    )
    parser.add_argument(
        "--kl-warmup-epochs", type=int, default=0,
        help="beta warm-up epochs for the CRL VAE methods (0 disables)",
    )
    args = parser.parse_args()

    if args.soft:
        args.seeds = 2
        args.epochs_scale = 0.1
        args.replications = 10

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out / "full_benchmark.log"), logging.StreamHandler()],
    )

    steps = build_steps(args)
    if args.dry_run:
        for step in steps:
            print(" ".join(step.command))
        return 0

    data_steps = [step for step in steps if not step.barrier]
    barrier_steps = [step for step in steps if step.barrier]
    jobs, gpus = _resolve_schedule(args, len(data_steps))
    threads = args.threads  # 0 = leave the environment untouched
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "errors.log").write_text(
        f"# step errors - {time.strftime('%Y-%m-%d %H:%M:%S')} - "
        f"{' '.join(sys.argv[1:])}\n",
        encoding="utf-8",
    )

    extra_env: dict[str, str] = {}
    if args.force:
        extra_env["GCMTS_FORCE"] = "1"
    if jobs > 1 and not args.keep_timing:
        # Wall-clock times under CPU contention are not comparable across
        # methods or to a sequential run, so suppress them by default.
        extra_env["GCMTS_TIMING"] = "0"
        LOGGER.warning(
            "Running %d steps concurrently: per-run 'seconds' will be recorded as "
            "unavailable to avoid contention-inflated timings. Use --jobs 1 (or "
            "--keep-timing) if you need the Time column.",
            jobs,
        )

    LOGGER.info(
        "Full benchmark: %d steps + %d barrier step(s) (device=%s, seeds=%d, "
        "epochs_scale=%.2f, replications=%d, jobs=%d, threads/job=%s, gpus=%s)",
        len(data_steps),
        len(barrier_steps),
        args.device,
        args.seeds,
        args.epochs_scale,
        args.replications,
        jobs,
        threads if threads > 0 else "default",
        ",".join(gpus) or "inherit",
    )

    summary_by_index: dict[int, dict[str, object]] = {}
    interrupted = False
    _install_signal_handlers()
    try:
        with _dashboard(
            data_steps,
            out / "full_benchmark.log",
            enabled=not args.no_dashboard,
            interval=args.dashboard_interval,
        ):
            if jobs <= 1 or len(data_steps) <= 1:
                for index, step in enumerate(data_steps):
                    summary_by_index[index] = _run_step(
                        step, index, gpus, threads, extra_env, log_dir
                    )
            else:
                with ThreadPoolExecutor(max_workers=jobs) as pool:
                    futures = {
                        pool.submit(
                            _run_step, step, index, gpus, threads, extra_env, log_dir
                        ): index
                        for index, step in enumerate(data_steps)
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        try:
                            summary_by_index[index] = future.result()
                        except Exception:  # pragma: no cover - surfaced below
                            LOGGER.exception("Step %s crashed", data_steps[index].name)
                            summary_by_index[index] = {
                                "step": data_steps[index].name,
                                "seconds": float("nan"),
                                "exit": -1,
                            }

            # Only regenerate tables when every data step finished normally.
            if not _STOP.is_set():
                for offset, step in enumerate(barrier_steps):
                    summary_by_index[len(data_steps) + offset] = _run_step(
                        step, len(data_steps) + offset, gpus, threads, {}, log_dir
                    )
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Interrupted: stopping remaining steps.")
    finally:
        _terminate_children(signal.SIGKILL)
        _restore_signal_handlers()
    interrupted = interrupted or _STOP.is_set()

    summary = [summary_by_index[index] for index in sorted(summary_by_index)]
    (out / "full_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if interrupted:
        LOGGER.warning(
            "Stopped by user after %d step(s); their partial results are saved. "
            "Re-run the same command to resume where it left off.",
            len(summary),
        )
        return 130
    failures = [row for row in summary if row["exit"] != 0]
    LOGGER.info("Finished: %d steps, %d failed.", len(summary), len(failures))
    for row in failures:
        LOGGER.error("FAILED: %s (exit=%s)", row["step"], row["exit"])
    errored = [row for row in summary if row.get("error")]
    if errored:
        LOGGER.error(
            "%d step(s) reported errors; per-step *.error files are in %s "
            "(one-line summary: %s)",
            len(errored), log_dir, log_dir / "errors.log",
        )
    else:
        LOGGER.info("No step errors; see %s for the (empty) summary.", log_dir / "errors.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
