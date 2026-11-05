# CRL hyperparameter search

The shared CRL methods use Optuna for the method-level hyperparameters only;
the benchmark protocol (data generation, seeds, evaluation metrics and the
common epoch ceiling) remains fixed.

Run a search on the held-out `npvar` seed:

```bash
uv pip install --python .venv/bin/python -e '.[tune]'
.venv/bin/python scripts/tune_crl_methods.py \
  --methods iVAE LEAP TDRL NCTRL MOSAIC CaDRe \
  --trials 6 --epochs 250 --batch-size 32 --device cuda \
  --out configs/crl_hpo.json
```

The selected configuration is stored in `configs/crl_hpo.json` and consumed
with:

```bash
.venv/bin/python scripts/run_crl_shared.py \
  --suite npvar --seeds 5 --epochs-scale 0.01 --batch-size 32 \
  --device cuda --hpo-config configs/crl_hpo.json \
  --early-stopping-patience 50 --restore-best-weights
```

The official iVAE is retained in the fidelity diagnostic. The shared temporal
table uses the stable image/temporal `IVAE` adapter because the official i.i.d.
iVAE is not defined for long flattened sequences and diverges at the full
budget. The rendered-image suite uses the stable image-aware adapters for the
same reason.
