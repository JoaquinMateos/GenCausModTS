# Vendored official CITRIS/iCITRIS models

Source: the authors' released implementation for

- Lippe et al., *CITRIS: Causal Identifiability from Temporal Intervened
  Sequences*, ICML 2022.
- Lippe et al., *Causal Representation Learning for Instantaneous and Temporal
  Effects in Interactive Systems*, ICLR 2023.

The `models/` tree is copied verbatim except for two small compatibility
edits:

1. `torchvision` is imported lazily in `models/shared/visualization.py` (and
   removed from `models/shared/utils.py` where it was unused), so the image
   models can be instantiated in environments without torchvision plotting
   support.
2. `models/citris_vae/lightning_module.py` aligns the latent sample dimension
   with the flattened intervention target in `sample_based_nll`, and computes
   the flow-branch KL directly from the already `(B, T-1)` tensor.

The upstream MIT licence is in `LICENSE`. The benchmark glue is in
`src/gcmts/causal_representation_learning/methods/citris_image.py`.
