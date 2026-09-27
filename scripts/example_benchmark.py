"""Minimal end-to-end example: train iVAE and run the CRL benchmark runner."""

from __future__ import annotations

import logging

from gcmts.causal_representation_learning.methods import IVAE
from gcmts.core import SimpleTrainer
from gcmts.data.synthetic import NonlinearICAGenerator
from gcmts.evaluation import BenchmarkTask, CrlBenchmarkRunner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


def main() -> None:
    data = NonlinearICAGenerator(
        observed_dim=6, latent_dim=3, horizon=1, n_regimes=3, seed=0
    )
    model = IVAE(observed_dim=6, latent_dim=3, u_dim=3)

    trainer = SimpleTrainer(
        max_epochs=200, lr=1e-3, batch_size=128, steps_per_epoch=20, seed=0
    )
    history = trainer.fit(model, data)
    print("final loss:", history["loss"][-1])

    task = BenchmarkTask(
        name="example_ivae",
        model=model,
        data=data.sample(512),
        metrics=["mcc", "r2_diag"],
        causal_level="L1_association",
    )
    result = CrlBenchmarkRunner().run(task)
    print(result.to_dict())


if __name__ == "__main__":
    main()
