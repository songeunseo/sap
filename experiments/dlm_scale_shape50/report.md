# Scale versus shape at50%: paired intervention diagnostic

## Objective / hypothesis
Test whether mean Wanda score and positive-log dispersion recommend different useful pruning exchanges. Shape may improve loss under matched budgets; no direction is assumed. This is a diagnostic of selected decisions, not a new full allocation method.

## Actual setup
Frozen LLaDA revision and80-state calibration. Same8 clean inputs added for distributions only. All224 sequential Uniform50 masks exactly reproduced. Twelve feature-selected, disjoint pairs: four early/late same type, four nearby-depth same type, four same-layer same shape. Each direction moves exactly786432 pruned weights between two projections; global budget remains3489660928. Same original row-wise Wanda ranking in both directions; no prefix recalibration per intervention. Sixteen distinct512-token validation article chunks, exact-k MC128, original shared seed/draws, baseline +24 variants. These are previously used development-validation articles, not new final-test documents. Two fixed8-article halves are descriptive replication checks.

## Results
Primary average over all12 pairs: raw-minus-shape NELBO **+0.0000709**, paired article95% CI [-0.0000116, +0.0001514], paired MC SE 0.0000268. Positive favors shape. GroupA -0.0000304, groupB +0.0001721. Raw-minus-baseline -0.0000192; shape-minus-baseline -0.0000901.

| Stratum | Raw minus shape | Article98.33% CI | Raw minus baseline | Shape minus baseline |
|---|---:|---|---:|---:|
| cross_depth_same_type | +0.0005498 | [+0.0003518, +0.0007560] | +0.0001905 | -0.0003593 |
| near_depth_same_type | -0.0000114 | [-0.0001531, +0.0001288] | -0.0000864 | -0.0000750 |
| same_layer_same_shape | -0.0003259 | [-0.0005805, -0.0000612] | -0.0001617 | +0.0001642 |

## Interpretation
Interpret the sign and interval jointly. Cross-depth advantage alone does not show information beyond depth; same-layer comparisons hold depth exactly fixed but vary projection type. Nearby-depth pairs hold projection type fixed and restrict depth distance. The controls do not isolate a single causal activation feature or establish DLM specificity. Log dispersion is related to the existing DSA family, not a novel proxy. Histograms are sampled while recorded weight/score moments are exact for their tensors. Equal transfer counts produce different percentage changes for different matrix sizes.

## Verification
Historical dense activation and mean score controls; all224 mask hashes; baseline per-draw likelihood agreement within2e-6; exact physical weight restoration and baseline sham after every variant; config/source revalidation. See receipts.

## Decision
Finish this fixed diagnostic and retain all signed results. No automatic score inversion, new coefficient, full allocation, test or GSM8K evaluation. A useful next criterion needs one explicit mechanism and a matched ablation; neither a favorable aggregate mean nor a histogram alone establishes it.

## Artifacts
statistics.json, projection_features.csv, distribution_summary.json, selection.json, evaluation/, restoration/, baseline_control.json, results.json, and static PNGs. Large input samples and boundary changes stored under /DATA/tmluser1/dlm-scale-shape50.
