# Initial reference check

Checked 2026-09-27 against primary publication records. This is a scoped initial pass, not a completed novelty survey.

| Reference | Verified location | Supported use in v0 | Limit |
|---|---|---|---|
| Nie et al., Large Language Diffusion Models (2025) | https://arxiv.org/abs/2502.09992v3, abstract | LLaDA uses masking and reverse masked-token prediction | No pruning result inferred |
| Sun et al., A Simple and Effective Pruning Approach for Large Language Models (ICLR 2024) | https://arxiv.org/abs/2306.11695v3, abstract and publication comment | Wanda ranks weights by weight magnitude and input activation, per output | Our calibration and allocation are project-specific |
| Sieberling et al., EvoPress: Accurate Dynamic Model Compression via Evolutionary Search (ICML 2025) | https://proceedings.mlr.press/v267/sieberling25a.html, abstract and metadata | Nonuniform compression and nonadditivity motivate full-model validation | No claim our heuristic has EvoPress's guarantees |
| Czarnecki et al., Sobolev Training for Neural Networks (2017) | https://arxiv.org/abs/1706.04859v3, abstract | Matching derivatives as well as values has prior precedent | Finite masked-context differences are not identical to input derivatives |

The next novelty pass should revisit the existing project bibliography for DLM compression, SAP, Quant-dLLM, FAIR-Calib, DSA, 2ndMatch, and interaction/relational distillation before making a narrow contribution claim. Earlier notes are discovery aids; citation-specific assertions require checking their primary texts. This v0 makes no first-method or state-of-the-art claim.
