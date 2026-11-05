# Vendored reference implementations

Only code needed to adapt the published methods is vendored here. Each copy
retains its upstream license; see the `LICENSE` files.

| Directory | Source | License |
|---|---|---|
| `tdrl/` | https://github.com/weirayao/tdrl | MIT |
| `leap/` | https://github.com/weirayao/leap | MIT |
| `mosaic/` | https://github.com/shichengf/mosaic | MIT (see upstream README) |

Official iVAE code was cloned for reference from https://github.com/ilkhem/iVAE
but is not yet vendored. Vendored files have only import-path/`ipdb` adaptations
and a shape fix in TDRL's nonparametric change prior.
