# Third-party notices

`gcmts` is distributed under the MIT License. The algorithms it implements are
original implementations written from the published papers; no source code was
copied from the authors' repositories. The repositories below are cited for
attribution and were consulted (when available) to confirm the algorithms'
formulations.

## Reference implementations of implemented methods

| Method | Paper (BibTeX key) | Reference repository | License |
| --- | --- | --- | --- |
| iVAE | `khemakhemVariationalAutoencodersNonlinear2020` | — (no public code) | — |
| LEAP | `weiranyaoLearningTemporallyCausal2021` | https://github.com/weirayao/leap | MIT |
| TDRL | `yaoTemporallyDisentangledRepresentation2022` | https://github.com/weirayao/tdrl | MIT |
| NCTRL | `songTemporallyDisentangledRepresentation2023` | https://github.com/xiangchensong/nctrl | MIT |
| Slow Flows | `edouardpineauTimeSeriesSource2020` | — (no public code) | — |
| CITRIS | `lippeCITRISCausalIdentifiability2022` | https://github.com/phlippe/CITRIS | BSD-3-Clause-Clear |
| MOSAIC | `shichengfanMOSAICModuleDiscovery2026` | https://github.com/shichengf/mosaic | No license declared |
| CEGEN | `carlremlingerConditionalLossDeep2021` | — (no public code) | — |

Notes:

- **MIT** (LEAP, TDRL, NCTRL) and **BSD-3-Clause-Clear** (CITRIS) are permissive
  licenses compatible with the MIT license used here, including attribution.
- **MOSAIC**'s repository declares no license; accordingly, only the algorithm
  described in the paper was implemented, and no code or assets were reused.
- The remaining methods have no public reference implementation; they were
  implemented from the equations in the respective papers.

## Direct dependencies

| Package | License |
| --- | --- |
| torch | BSD-3-Clause |
| lightning | Apache-2.0 |
| scikit-learn | BSD-3-Clause |
| networkx | BSD-3-Clause |
| pandas | BSD-3-Clause |
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| pydantic | MIT |

Transitive dependencies are resolved and pinned in `uv.lock`; each retains its
own license. If you redistribute `gcmts`, preserve the copyright notices of the
dependencies as required by their licenses.

## License compatibility

The MIT License is compatible with all licenses above: it imposes no copyleft
obligation and can incorporate and be combined with MIT, BSD-3-Clause,
BSD-3-Clause-Clear, and Apache-2.0 code, provided their attribution and notice
conditions are met. Apache-2.0 additionally grants an explicit patent license,
which is preserved for the corresponding dependencies.
