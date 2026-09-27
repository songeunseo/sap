# Completion review — 2026-09-16 22:46 KST

## Status / verification
Complete: baseline +24 interventions, 16 articles ×128 draws each, 400 block records /51,200 forwards. All224 Uniform50 masks reproduced, all24 exact-budget boundary files checked for unique indices and correct baseline-mask membership, all24 physical restoration/sham receipts passed. Input/source hashes and all block digests/mean/variance verified; primary article bootstrap independently reproduced. See verification.json. GPU run completed at22:42:44 KST.

## Observed results
Define delta = NELBO(prune higher mean) - NELBO(prune higher positive-log variance). Positive favors the shape rule.

| Preselected group | Delta | Paired article CI | Pair point directions |
|---|---:|---|---|
| Early vs late, same type | +0.000549826 | 98.33% [+0.000351771,+0.000756003] | Shape favored4/4 |
| Nearby depth, same type | -0.000011358 | 98.33% [-0.000153103,+0.000128797] | Shape favored1/4 |
| Same layer, same shape, different type | -0.000325880 | 98.33% [-0.000580499,-0.000061172] | Shape favored1/4 |
| All12 pairs | +0.000070863 | 95% [-0.000011631,+0.000151363] | Shape favored6/12 |

Overall A/B document halves change sign (-0.000030386/+0.000172111). Cross-depth and same-layer group means retain their respective signs in both halves. Both average direction-to-baseline changes are signed and are recorded in results.json. These averages describe separate intervention models, not a single combined12-pair allocation.

Distribution control: clean/corrupted mean-score rank rho0.998269, positive-log-variance rank rho0.993610. Large corruption-induced ranking changes are not observed in this survey.

## Interpretation
- The early/late selected pairs support protecting later projections relative to the opposite DLP-style mean decision, within this fixed sparse background. They do not prove log variance has information beyond depth.
- The same-layer selected pairs favor the opposite criterion on average; replacing the mean with a universal positive-log-variance rule is not supported.
- Nearby-depth comparisons are inconclusive.
- The experiment locates a dependence on the decision being made (across depth versus across type at fixed depth). It does not establish the causal source of this dependence, a universal piecewise rule, or DLM specificity.
- Effects are small because each variant exchanges only786432 weights. Small effect size is not a failure of measurement; ranking and bootstrap must still be interpreted within the selected finite set.
- This is16 previously used validation articles and a feature-selected pair set; CI conditions on the chosen pairs and proxy estimates. No final-test/GSM8K evidence or independently optimized full policy.

## Decision
Do not promote positive-log variance as a general replacement for magnitude or a novel DLM proxy. Preserve the cross-depth observation as a mechanism clue, with depth/type controls. No further experiment launched in this completion check.

## Memory synchronization
Obsidian MCP was unavailable during the completion check (handshake timeout). Update the status and completion results in Experiments/2026-09-16-Scale-Shape50-Distribution-and-Budget-Exchange.md and Research/DLM-Pruning/Research-State.md when the connection returns; this local record preserves the verified findings meanwhile.
