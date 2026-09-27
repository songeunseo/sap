# Reference verification for the revised draft

Checked 2026-09-27 against primary records and the method passages listed below. The first four records retain the checks from the initial draft; four close compression/response precedents were added for this revision. This is a focused comparison, with broader novelty review still open.

| Work | Primary source and passage | Use in the manuscript | Boundary |
|---|---|---|---|
| Large Language Diffusion Models (Nie et al., 2025) | [arXiv v3, abstract](https://arxiv.org/abs/2502.09992v3) | Masked inputs and Transformer reverse prediction motivate calibration across context states. | No pruning result inferred. |
| A Simple and Effective Pruning Approach for Large Language Models (Sun et al., ICLR 2024) | [arXiv v3, abstract and publication record](https://arxiv.org/abs/2306.11695v3) | Wanda uses weight magnitude and input activation for per-output ranking. | The project's sparse-prefix calibration and block allocator are separate choices. |
| EvoPress (Sieberling et al., ICML 2025) | [PMLR proceedings, abstract](https://proceedings.mlr.press/v267/sieberling25a.html) | Compression-profile search and the limits of independent layer-error reasoning. | No transfer of guarantees to the present rank heuristic. |
| Sobolev Training for Neural Networks (Czarnecki et al., 2017) | [arXiv v3, abstract](https://arxiv.org/abs/1706.04859v3) | Function values plus target derivatives have precedent. | Discrete masked-context differences are a different measurement. |
| Sink-Aware Pruning for Diffusion Language Models | [Primary text, Section 3.2, Eqs. 8–14 and Figure 3](https://arxiv.org/html/2602.17664) | Mean soft sink scores on noised inputs reweight activations used by Wanda/SparseGPT. Closest cited direct weight-pruning baseline. | The implemented signal is averaged soft sink reweighting; do not describe it as allocating weights by temporal sink variance. |
| Quant-dLLM (ICLR 2026) | [Official paper, Sections 3.2–3.4, Algorithm 1](https://proceedings.iclr.cc/paper_files/paper/2026/file/805da7ef883245cb35e012cc179a5f6f-Paper-Conference.pdf) | Partially visible calibration and block mixed precision at average two bits. | Quantization differs from the current fixed weight-removal budget. No cross-paper accuracy comparison. |
| FAIR-Calib | [Primary text, Section 3, Figure 2](https://arxiv.org/html/2606.06547) | Teacher frontier/reliability probing supplies weights for hidden-state reconstruction in PTQ. | Generic use of diffusion dynamics or fragile decisions is already prior. Its output-divergence analysis is not a theorem for scalar A+C. |
| 2ndMatch | [Primary text, Section 4, Eq. 9](https://arxiv.org/html/2506.05398) | Finetuning pruned image diffusion models uses projected JᵀJ sensitivity measurements. | Response preservation is established motivation; image-model finetuning differs from static masked-language allocation. |

## What the comparison supports

The current question concerns the same unresolved query across paired partial contexts, with calibration states, ranking, and allocation machinery fixed across controls. This comparison establishes a precise scope to evaluate. It does not establish a first-method claim, practical superiority, or priority over all relational/derivative distillation work.

Existing project notes remain discovery aids for DSA, Layer Collapse, further DLM compression, and relational distillation. Any additional manuscript assertion needs verification against the corresponding primary source. Earlier project baselines must retain their original model, sparsity, calibration, search, and evaluation conditions when discussed.
