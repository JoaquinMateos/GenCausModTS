"""A/B test: does the static outcome-masking fix change CATE quality?

Trains each static CTEE method on the same IHDP replications twice, once with the
canonical input (covariates only, outcome masked) and once with the legacy input
(outcome appended to the state), and reports out-of-sample PEHE. The legacy path
is what ``run_ctee_static.py`` used before the fix; it lets the model copy the
factual outcome.

    uv run python scripts/diagnose_static_masking.py --device mps --reps 0 1 2 \
        --epochs-scale 0.5 --methods ganite crn cepae catsg --out diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ctee_static import METHODS, _load_pair, _metrics, _tau_from_model  # noqa: E402

from gcmts.core import SimpleTrainer  # noqa: E402
from gcmts.data.semi_synthetic import IHDPGenerator  # noqa: E402
from gcmts.evaluation.benchmarks import EMA_DECAY  # noqa: E402


def run_one(dataset: str, index: int, method: str, mask: bool, device: str,
            epochs_scale: float, data_dir: Path) -> dict:
    train_rep, test_rep = _load_pair(dataset, index, data_dir)
    train_gen = IHDPGenerator(train_rep, seed=index, mask_outcome=mask)
    test_gen = IHDPGenerator(
        test_rep, seed=index + 10_000, reference=train_gen, mask_outcome=mask
    )
    factory, epochs, _ = METHODS[method]
    epochs = max(2, int(round(epochs * epochs_scale)))
    torch.manual_seed(index)
    model = factory(train_gen)
    trainer = SimpleTrainer(
        max_epochs=epochs, lr=1e-3, batch_size=256, steps_per_epoch=15,
        log_every=0, device=device, seed=index, grad_clip=1.0, scheduler="cosine",
        val_size=512, val_every=20, ema_decay=EMA_DECAY,
    )
    trainer.fit(model, train_gen)
    tau_hat, _ = _tau_from_model(model, test_gen, trainer.resolve_device(model))
    metrics = _metrics(tau_hat, test_rep)
    return {
        "dataset": dataset, "replication": index, "method": method,
        "mask_outcome": mask, "tau_hat_mean": float(np.mean(tau_hat)),
        "tau_hat_sd": float(np.std(tau_hat)), **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ihdp")
    parser.add_argument("--data-dir", type=Path, default=Path("data/semi_synthetic"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--reps", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="*", default=["ganite", "crn", "cepae", "catsg"])
    parser.add_argument("--epochs-scale", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("docs/diagnostics_masking.json"))
    args = parser.parse_args()

    rows = []
    for index in args.reps:
        for method in args.methods:
            for mask in (True, False):
                row = run_one(args.dataset, index, method, mask, args.device,
                              args.epochs_scale, args.data_dir)
                rows.append(row)
                print(
                    f"rep={index} {method:7s} mask={mask!s:5s} "
                    f"pehe={row['pehe']:.3f} ate={row['ate_error']:.3f} "
                    f"tau={row['tau_hat_mean']:.2f}+-{row['tau_hat_sd']:.2f}",
                    flush=True,
                )
                args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
