"""Materialise the CRL synthetic benchmarks as persistent train/eval pools.

The review's protocol fixes the synthetic *process* per seed (latent graph,
mechanisms, mixing map) across methods; the runners normally draw fresh
trajectories every step, which is CPU-bound. This writes, for every
``(suite, seed)``, a fixed train pool and a **disjoint** held-out eval pool
drawn from the same seeded generator, so the benchmark can be generated once on
a CPU box and trained on a GPU box without per-step generation.

Rigor / protocol note
---------------------
* Pools use the same seeding as the runners (``torch.manual_seed(seed)`` before
  ``suite.make_generator(seed)``), so the latent graph/mechanisms/mixing match
  the protocol and are identical across methods.
* The pool is finite: over one run each trajectory is seen about
  ``steps * batch_size / train_size`` times, unlike the standard fresh-sampling
  protocol. Size ``--train-size`` accordingly and report the change (the manifest
  records the sizes for audit).
* ``--clean`` deletes a previous generation before writing.

Usage (from ``gcmts/``)::

    uv run python scripts/generate_synthetic_data.py --clean --seeds 10
    uv run python scripts/generate_synthetic_data.py --seeds 10 --with-grid --train-size 65536
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_crl_shared as shared  # noqa: E402

from gcmts.data.synthetic_store import dataset_path, save_dataset  # noqa: E402
from gcmts.evaluation.progress import ProgressTracker  # noqa: E402

logger = logging.getLogger("gcmts.generate_data")


def _suite_specs(args: argparse.Namespace) -> list[shared.Suite]:
    names = args.suites or sorted(shared.SUITES)
    unknown = [name for name in names if name not in shared.SUITES]
    if unknown:
        raise SystemExit(f"Unknown suites: {', '.join(unknown)}")
    specs = [shared.SUITES[name] for name in names]
    if args.with_grid:
        specs.extend(config.as_suite() for config in shared.GRID_CONFIGS)
    return specs


def generate_one(
    suite: shared.Suite, seed: int, *, train_size: int, eval_size: int, out_dir: Path
) -> str:
    """Build the seeded generator and persist its (train, eval) pools."""
    torch.manual_seed(seed)
    generator = suite.make_generator(seed)
    train = generator.sample(train_size)
    eval_batch = generator.sample(eval_size)
    record = save_dataset(
        dataset_path(out_dir, suite.name, seed),
        train,
        eval_batch,
        suite=suite.name,
        seed=seed,
        max_lag=int(getattr(generator, "max_lag", 1)),
        generator=type(generator).__name__,
        latent_dim=getattr(generator, "latent_dim", None),
    )
    return record.path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument(
        "--suites", nargs="*", default=None, help="subset of CRL suites (default: all)"
    )
    parser.add_argument(
        "--with-grid", action="store_true", help="also materialise the 18-problem grid"
    )
    parser.add_argument(
        "--train-size", type=int, default=65536,
        help="trajectories in the training pool (finite-pool protocol; default: 65536)",
    )
    parser.add_argument(
        "--eval-size", type=int, default=4096,
        help="trajectories in the disjoint held-out eval pool (default: 4096)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="delete the existing output directory before generating",
    )
    parser.add_argument("--dry-run", action="store_true", help="list what would be generated")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    specs = _suite_specs(args)
    total = len(specs) * args.seeds

    if args.dry_run:
        for suite in specs:
            for seed in range(args.seeds):
                print(dataset_path(args.out_dir, suite.name, seed))
        print(f"# {len(specs)} suites x {args.seeds} seeds = {total} datasets")
        return 0

    if args.clean and args.out_dir.exists():
        logger.info("Cleaning previous generation at %s", args.out_dir)
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating %d datasets (%d suites x %d seeds; train=%d, eval=%d) into %s",
        total, len(specs), args.seeds, args.train_size, args.eval_size, args.out_dir,
    )
    progress = ProgressTracker(
        args.out_dir / "generate_progress.json", "Synthetic data generation", total
    ).start()
    for suite in specs:
        for seed in range(args.seeds):
            path = generate_one(
                suite, seed,
                train_size=args.train_size, eval_size=args.eval_size, out_dir=args.out_dir,
            )
            logger.info("wrote %s", path)
            progress.advance()
    progress.finish()
    logger.info(
        "Done: %d datasets under %s (manifest: %s/%s)",
        total, args.out_dir, args.out_dir, "manifest.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
