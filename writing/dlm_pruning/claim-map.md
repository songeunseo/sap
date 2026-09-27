# Claim–evidence map

Research question: Does matching conditional responses across partially revealed contexts improve a static DLM pruning allocation beyond matching predictions on the same states?

## Central insight and testable hypothesis

**Insight:** Equal mean squared prediction error can correspond to different errors in the response to context. A constant residual cancels in a response difference; a changing residual may not.

**Algebraic illustration:** Figure 1 compares (1,1,1,1,1,1,1,1) with (1,1,−1,−1,1,1,−1,−1). Both have A=1. Their natural response penalties are 0 and 4/3. These are constructed values from the existing design note, not model observations or expected results. Perfect endpoint matching also matches all responses.

**Empirical hypothesis:** Under the frozen state bank, weight ranking, allocator, and 50% budget, Multi improves GSM8K accuracy over A. The primary Multi−A contrast tests this hypothesis. The two cross contrasts separately test the chosen connections against specific controls.

**Contribution at the current stage:** A concrete calibration criterion and a controlled test of its incremental value. A positive mathematical distinction alone does not establish capability improvement or novelty.

## Claims and evidence

| ID | Claim or question | Status before full completion | Exact evidence needed | Permitted scope |
|---|---|---|---|---|
| C0 | A and C define different preferences over finite residual patterns | Algebraic property | Response identity, Figure 1 exact example, and direct inequality proof | No task-performance guarantee |
| C1 | Multi improves GSM8K over the same-bank A-only allocation | Hypothesis; separate100 has +1/100, Holm p=1 | `report.json#/comparisons/primary_remaining_1119/Multi-A` and all paired checkpoints | One frozen bank/mask/benchmark/protocol |
| C2 | Natural-chain connections have an advantage over the tested cross-chain connections | Hypothesis | Multi–Cross and Multi–CrossMatched, jointly interpreted with C1 | These specific controls; no universal necessity or pure-semantic causal claim |
| C3 | Quality improves over competitive practical alternatives | Unresolved | Matched model/protocol/sparsity baselines and calibration cost | Full four-arm study alone cannot establish this |
| C4 | The effect generalizes | Unresolved | Separately designed calibration seeds, model/benchmark/sparsity checks | No generalization from more GSM8K questions alone |
| C5 | Fresh response fidelity explains downstream differences | Exploratory | Fresh-bank per-span diagnostics plus task comparison | Association and proxy/task mismatch; no causal mediation claim |

Full source: `experiments/dlm_crosschain_control50/output/report.json`. Separate100 source: `experiments/dlm_multi_validation10050/output/report.json`. Generated `closeout.json` records the exact hashes and fresh verification.

## Result-contingent revisions

These are interpretation rules recorded while the full experiment is running. They do not replace or amend its predeclared statistics.

1. Positive Multi–A with adequate evidence: describe an observed incremental benefit under this setup. Check Cross controls before attributing that benefit to natural-chain structure.
2. Positive Multi–A and positive results against both cross controls: describe support for this connection design within the tested bank. Keep generalization, alternative controls, and practical baselines open.
3. Positive Multi–A but inconclusive cross contrasts: retain the incremental-effect result; leave the source of that effect unresolved. A generic regularization explanation remains possible.
4. Inconclusive Multi–A: report its estimate and interval; do not claim equivalence or tune on the primary1119. Reassess whether more precision or a revised method has higher scientific value.
5. Negative Multi–A with adequate evidence: reduce the method claim. Preserve the observed failure of this C/allocator configuration without rejecting all response-preserving designs.
6. Improved diagnostic fidelity without task improvement: report the mismatch. Do not convert calibration improvement into capability evidence.

## Tables and figures

- Figure 1: constructed eight-state residuals; exact values in `figures/response-example.json`, checked against the definitions. No empirical performance claim.

- Table 1: prior development100 and separate100, clearly labelled as historical screens.
- Table 2: all four full1319 scores, exposed200 and primary1119 separately.
- Table 3: the three fixed primary contrasts, paired gain/loss, effect, exact McNemar and Holm, unadjusted intervals.
- Diagnostic table: fresh-bank A, natural C, cross C, query CE and the explicitly defined sign metric.
- Figure candidate after verification: paired primary effects with intervals, generated from closeout.json. No illustrative performance curves.

## Next experiment decision

After closeout, first review C1 and C2 and the uncertainty intervals. Select one unresolved claim with a concrete comparison. Record expected decision value, compute budget and stop condition. Baseline quality/cost and generalization are required before broad method claims; they are not automatically launched by the writer.
