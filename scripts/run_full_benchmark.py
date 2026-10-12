"""One-command full benchmark orchestrator for the review.

Runs every experiment family (CRL shared suites, CRL sanity, CTEE temporal and
static, SDD, adjacent branches, and optionally the 18-problem grid), then
regenerates all LaTeX tables. Designed to be launched once on a CUDA server:

    uv run python scripts/run_full_benchmark.py --device cuda --seeds 10

Soft smoke run (fast, few seeds/epochs, small replications):

    uv run python scripts/run_full_benchmark.py --soft --device cuda

Everything is written under ``outputs/`` and ``outputs_shared/`` and the tables
into ``../latex/pic/experiments/``; a JSON summary and a log are written to
``outputs/full_benchmark_summary.json`` and ``outputs/full_benchmark.log``.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from gcmts.evaluation.benchmarks import (
    CRL_BENCHMARKS,
    CTEE_REAL_BENCHMARKS,
    CTEE_STATIC_BENCHMARKS,
    crl_label,
)

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


@dataclass
class Step:
    name: str
    command: list[str]


def build_steps(args: argparse.Namespace) -> list[Step]:
    py = sys.executable
    device = args.device
    seeds = args.seeds
    epochs_scale = args.epochs_scale
    steps: list[Step] = []
    for suite in CRL_SUITES:
        if args.suites and suite not in args.suites:
            continue
        steps.append(
            Step(
                f"CRL {crl_label(suite)}",
                [py, "scripts/run_crl_shared.py", "--device", device, "--suite", suite,
                 "--seeds", str(seeds), "--epochs-scale", str(epochs_scale),
                 "--out-dir", "outputs_shared"],
            )
        )
    if not args.suites:
        steps.append(
            Step(
                "CRL sanity (matching-benchmark)",
                [py, "scripts/run_crl_benchmarks.py", "--device", device,
                 "--seeds", str(seeds), "--out-dir", "outputs"],
            )
        )
        steps.append(
            Step(
                "CTEE temporal",
                [py, "scripts/run_ctee_benchmarks.py", "--device", device,
                 "--datasets", *CTEE_DATASETS, "--seeds", str(seeds), "--out-dir", "outputs"],
            )
        )
        for dataset in CTEE_STATIC_BENCHMARKS:
            steps.append(
                Step(
                    f"CTEE static {dataset.upper()}",
                    [py, "scripts/run_ctee_static.py", "--device", device, "--dataset", dataset,
                     "--replications", str(args.replications), "--out-dir", "outputs"],
                )
            )
        steps.append(
            Step(
                "SDD invariance",
                [py, "scripts/run_sdd_benchmarks.py", "--device", device,
                 "--seeds", str(seeds), "--out-dir", "outputs"],
            )
        )
        steps.append(
            Step(
                "Adjacent branches + II.D",
                [py, "scripts/run_adjacent_benchmarks.py", "--device", device,
                 "--seeds", str(seeds), "--out-dir", "outputs"],
            )
        )
        for dataset in CTEE_REAL_BENCHMARKS:
            steps.append(
                Step(
                    f"CTEE real {dataset}",
                    [py, "scripts/run_ctee_static.py", "--device", device, "--dataset", dataset,
                     "--replications", "1", "--out-dir", "outputs"],
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
                )
            )
    if args.with_grid:
        steps.append(
            Step(
                "CRL 18-problem grid",
                [py, "scripts/run_crl_shared.py", "--device", device, "--grid",
                 "--seeds", str(seeds), "--epochs-scale", str(epochs_scale),
                 "--out-dir", "outputs_shared"],
            )
        )
    steps.append(
        Step(
            "Regenerate LaTeX tables",
            [py, "scripts/make_paper_tables.py", "--shared-dir", "outputs_shared",
             "--outputs-dir", "outputs", "--latex-dir", "../latex"],
        )
    )
    return steps


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
    LOGGER.info(
        "Full benchmark: %d steps (device=%s, seeds=%d, epochs_scale=%.2f, replications=%d)",
        len(steps),
        args.device,
        args.seeds,
        args.epochs_scale,
        args.replications,
    )

    summary: list[dict[str, object]] = []
    for step in steps:
        printable = " ".join(step.command)
        if args.dry_run:
            print(printable)
            continue
        LOGGER.info("RUN %s", step.name)
        start = time.time()
        completed = subprocess.run(step.command, check=False)
        elapsed = time.time() - start
        LOGGER.info("DONE %s in %.1fs (exit=%d)", step.name, elapsed, completed.returncode)
        summary.append({"step": step.name, "seconds": elapsed, "exit": completed.returncode})

    if not args.dry_run:
        (out / "full_benchmark_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        failures = [row for row in summary if row["exit"] != 0]
        LOGGER.info("Finished: %d steps, %d failed.", len(summary), len(failures))
        for row in failures:
            LOGGER.error("FAILED: %s (exit=%s)", row["step"], row["exit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
