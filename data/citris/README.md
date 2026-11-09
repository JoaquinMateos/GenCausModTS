# CITRIS video benchmarks

The benchmark reads the official Zenodo releases directly; no synthetic
surrogate is used for Causal3DIdent.

```bash
python scripts/download_citris_datasets.py --root data/citris
```

Expected layout:

- `pong/complexpong_time_dep_015_32width_singleint/train.npz`
- `pinball/pinball/train.npz`
- `ball/box_x/train.npz`
- `temporal_causal3d/causal3d_time_dep_all7_conthue_01_coarse/train.npz`

The loaders are in `src/gcmts/data/citris.py`; the official model release is
vendored under `src/gcmts/vendor/citris` with its MIT licence.
