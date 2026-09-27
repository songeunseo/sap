# Literature-led allocation review — 2026-09-22

Scope: parent-led primary-source review; no implementation or new experiment. Two subagents separately cover distillation objectives and DLM analysis. The third requested literature axis is handled locally because additional agent creation hit a tool limit. Obsidian Research-State was read successfully before work.

## What the literature changes

Do not make the newly proposed hard endpoint-A floor the presumptive main improvement. Lower calibration fidelity loss is a meaningful objective but is not an empirical safety guarantee for downstream quality. First address what output information the criterion preserves and where its calibration states come from. Keep exact-budget measured exchanges as shared infrastructure, not the claimed novelty.

## Primary-source ledger

| Paper and source | Checked locator / scope | Finding | Design implication and limit |
|---|---|---|---|
| [Shin et al., Rethinking Pruning LLMs, EMNLP2024](https://aclanthology.org/2024.emnlp-main.68.pdf) | §3.1–3.2, Tables1–2, Fig4; unstructured50%, LLaMA7B/OPT125M; updates remaining weights with reconstruction | For Wanda, adding cross-block reconstruction to BR+GP lowers calibration error .51→.38 but raises WikiText PPL6.68→6.79; test reconstruction2.23→2.48. §4 studies generated calibration text. | Supports independent-document quality checks and examining calibration distribution, not a claim that A is useless. Weight reconstruction differs from our frozen-weight allocation, so it does not directly falsify an A-bound candidate. |
| [LLM Surgeon, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/38a1671ab0747b6ffe4d1c6ef117a3a9-Paper-Conference.pdf) | Eq6–10, §3.3–3.5, Algorithm1; Fisher/Kronecker approximation, compensation updates | Final-loss removal costs support dynamic global thresholds, while multi-shot pruning recomputes curvature. Selection costs are independent approximations even though subsequent compensation accounts for correlations. | Borrow measured/current-model sensitivity and distinguish selection from actual joint effect. Do not copy its compensated OBS cost into a frozen-weight problem as if it were the same objective. Curvature and dynamic allocation are existing mechanisms. |
| [GISP, arXiv2510.18030v2](https://arxiv.org/html/2510.18030) | §3.1–3.2, Table2/Fig3, Algorithm1; structured heads/MLP channels, AR LLMs, no intermediate fine-tuning | One-shot global scoring degrades at high pruning ratios; iterative model-level loss gradients and normalization improve the studied setting. | Supports re-evaluating a changing sparse model, not a theorem that a particular trust radius or constraint works. Structured removal, changing sparsity, and task-loss calibration differ from our fixed-budget Wanda-boundary trades. |
| [LiPRA, Neurocomputing2026](https://doi.org/10.1016/j.neucom.2026.134753) | Publisher indexed abstract/introduction only; full methods PDF not obtained | Authors describe budget-preserving global rate perturbations, a continuous local surrogate and KKT-derived rate mapping, with block then sublayer allocation. | Particularly close prior for global sensitivity plus exact/shared budget. Do not claim that approach as ours or invent its KKT formula. No independently verified numerical benchmark or ablation claim is taken from the abstract. |
| [FALCON, AISTATS2024](https://proceedings.mlr.press/v238/meng24a.html) | Primary abstract and paper inspected; general neural-network pruning | Combines fidelity, FLOP and sparsity constraints with a combinatorial/first-order framework. | Confirms constrained pruning is established. Additional FLOP constraints do not answer our conditional-response question; not selected as a new core module. |

## Concrete allocation recommendation derived from these papers

Use one shared functional hard-mask allocator for the objective comparison:

1. Freeze the original nonzero weights, within-row rank order, exact weight budget and finite candidate domain.
2. At the current sparse mask, propose count-preserving removal/restoration changes. A low-cost local predictor may rank proposals but cannot certify their outcome.
3. Measure the complete candidate with the chosen fixed functional objective. Finalists and incumbent use identical calibration states and evaluation precision. Charge rejected proposals and initialization work.
4. Accept measured objective improvements and stop under a predeclared compute cap. Evaluate frozen outputs on separate documents and NELBO/task metrics.

This architecture is borrowed optimization machinery. No claim that new papers establish its performance for A+C or eliminate nonconvexity follows. It should be held constant while comparing scalar gold-vs-rest, full-distribution endpoint matching, and a literature-derived conditional-response readout.

## What not to infer

- A tighter calibration-A bound does not guarantee NELBO or GSM8K preservation.
- Global gradients or curvature do not establish DLM-specific novelty.
- A KKT solution is exact only for its stated surrogate/constraints, not automatically for the hard sparse network.
- Generated calibration data is already prior art and our Reveal-KL study already used dense trajectories. Reusing that concept requires stating the new distinction: state distribution, output information, or paired response—not calling trajectories new.
- The frozen-probe local-linear failure does not refute all local methods; it motivates measured candidate checks and a bounded step.

## Outstanding decision

The strongest route now looks like improving the information content of A+C (target confidence versus competing-token relations), while keeping realistic paired calibration and a known allocator. The distillation and DLM literature reports will determine the actual mathematical proposal; this report does not precommit to another arbitrary weighted sum or hard guard.

