"""Profile the per-phase cost of one training run.

Shows how a step's wall time splits across data generation, forward, loss,
backward, optimiser and the loss-sync — so it is obvious whether a suite is
CPU-bound (``data`` dominates) or GPU-bound (``forward``/``backward`` dominate).

Usage (from ``gcmts/``)::

    uv run python scripts/profile_training.py --suite npvar --method iVAE
    uv run python scripts/profile_training.py --suite voronoi --method CITRIS --device cuda
    uv run python scripts/profile_training.py --suite npvar --device cuda --json profile.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_crl_shared as shared  # noqa: E402

from gcmts.core import SimpleTrainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="npvar", choices=sorted(shared.SUITES))
    parser.add_argument("--method", default=None, help="method name (default: first of the suite)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", type=Path, default=None, help="also write the profile here")
    parser.add_argument(
        "--data-source", choices=("fresh", "pool"), default="fresh",
        help="profile the fresh-sampling path (default) or the persisted pool path",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/synthetic"),
        help="root of the persisted synthetic pools used with --data-source pool",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    suite = shared.SUITES[args.suite]
    methods = shared._methods_for(suite, 1.0)
    if args.method:
        methods = [m for m in methods if m[0] == args.method]
        if not methods:
            print(
                f"Unknown method '{args.method}' for suite '{args.suite}'. "
                f"Available: {', '.join(name for name, _, _ in shared._methods_for(suite, 1.0))}"
            )
            return 2
    name, factory, _ = methods[0]

    if args.data_source == "pool":
        train_data, _eval_batch, _max_lag = shared._load_pool(
            args.dataset_dir, suite.name, args.seed
        )
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
        train_data = suite.make_generator(args.seed)
    model = factory()
    trainer = SimpleTrainer(
        max_epochs=args.epochs,
        lr=1e-3,
        batch_size=args.batch_size,
        steps_per_epoch=15,
        log_every=0,
        device=args.device,
        seed=args.seed,
        grad_clip=1.0,
        scheduler="cosine",
        val_size=512,
        val_every=20,
        profile=True,
    )
    if isinstance(train_data, shared.PooledGenerator):
        train_data.to(trainer.resolve_device(model))
    print(f"Profiling {suite.name}/{name} for {args.epochs} epochs (batch {args.batch_size})...")
    trainer.fit(model, train_data)
    if trainer.profiler is not None:
        profile = trainer.profiler.as_dict()
        if args.json is not None:
            args.json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
