# Role Error Causal Decomposition @ 65%

## Objective

Test whether masked/unmasked local reconstruction errors identify distinct causal
damage to final masked-token predictions, beyond arbitrary token partitions.

## Setup

- Frozen LLaDA-8B-Base revision and 80 allocation-calibration DLM states.
- Fixed 28 projections: all seven repository projection types in layers 3, 11,
  19, and 27.
- One projection at a time receives its persisted Standard Wanda 65% mask.
- Same-path batched interventions: sham, masked-only, unmasked-only, Both,
  energy-matched masked/unmasked, and three cardinality-matched random A/B pairs.
- Sequence-cluster bootstrap with 20,000 resamples.

## Execution gates

- Mixed-role Both exactly reproduced direct sparse output: maximum logit
  difference `0.0`.
- Batch-1 dense and batched suffix sham showed substantial BF16 batch-shape
  drift (maximum raw logit difference `15.875`), so all primary effects use the
  same-path sham row.

## Results

Actual masked/unmasked positions did not create a larger absolute causal
separation than random partitions. Normalized absolute KL contrast was `0.2611`
for the actual split and `0.3012` for cardinality-matched random splits. The
actual-minus-random difference was `-0.0401`, sequence-bootstrap 95% CI
`[-0.0498, -0.0301]`, and was negative for all 8 sequences.

The signed contrast showed a small masked-relative effect beyond random:
actual-minus-random `+0.0164`, 95% CI `[0.00136, 0.03032]`. Its sequence means
were positive for 6/8 sequences, so this is weaker than the absolute-contrast
result.

Energy matching reduced actual absolute KL contrast from `0.2611` to `0.2012`;
77.0% of the original mean contrast remained. This shows that magnitude explains
part, but the present control does not isolate a uniquely semantic role effect.

### Which local statistic predicts the causally worse side?

| Local comparison | Actual agreement | Random agreement | Actual − random 95% CI |
|---|---:|---:|---:|
| Raw summed reconstruction numerator | 72.90% | 87.90% | −15.00 pp `[-16.44, -13.38]` |
| Reconstruction numerator per token | **57.01%** | 49.39% | **+7.62 pp `[+5.82, +9.51]`** |
| Dense-energy-normalized reconstruction | 52.95% | 49.33% | +3.62 pp `[+2.65, +4.51]` |

Raw summed error predicts the causally worse group well because it primarily
tracks how much error was injected; it performs even better for arbitrary
partitions and is not role-specific evidence. Per-token reconstruction has the
largest actual-specific advantage. The current dense-energy normalization keeps
only a smaller advantage.

Current normalized local contrast has weak association with causal contrast:
pooled signed Spearman `0.0535`, projection-mean signed Spearman `0.2304`, and
absolute projection-mean Spearman `-0.1073`. Numerator and dense-energy
normalization selected opposite roles in 1,037/2,240 projection-state cells.

Causal role contrast itself was stable across sequences at the projection level:
leave-one-sequence-out mean Spearman `0.9949` and mean sign agreement `98.66%`.
Random partitions were also stable, although less so (mean Spearman
`0.9603–0.9683`). Stability alone therefore does not establish semantic role
specificity.

Both-role KL was usually subadditive: mean
`KL_Both - KL_M - KL_U = -0.000491`, with 12.10% of cells superadditive. The
median relative logit additivity residual was `0.557`, so the downstream network
does transform the two intervention paths nonlinearly; KL interaction cannot be
attributed only to the softmax.

## Interpretation

The strong hypothesis that the masked/unmasked partition inherently produces
larger causal damage separation than arbitrary equal-cardinality partitions is
not supported. The current `max(E_M,E_U)` score also does not reliably select the
causally worse role.

A narrower signal survives: per-token reconstruction error predicts which real
role causes more final KL better than it predicts an arbitrary partition. This
is statistically stable across the eight sequences, but its absolute accuracy is
57%, so it is an incomplete proxy rather than a finished pruning rule.

## Decision

- Do not use this result to justify the current dense-energy-normalized Max rule.
- Do not start targeted path patching from the rejected large-separation premise.
- Next analyze per-token masked/unmasked marginal reconstruction across all 224
  projections and six sparsities, measuring allocation changes and prediction of
  the already-frozen single-projection KL curves before constructing another
  downstream candidate.
