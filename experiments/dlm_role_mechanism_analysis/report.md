# Masked/Unmasked Allocation Signal Analysis

## Question

Do masked and unmasked tokens provide distinct, stable information for projection-wise sparsity allocation?

## Frozen Setup

- LLaDA-8B, 224 projections, 80 frozen DLM states and six sparsity levels.
- Primary grid 50–75% with target 65%; target50/target75 grids are secondary checks.
- Reconstruction sufficient statistics are pooled within role before normalization.
- The independent resampling unit is the WikiText-2 sequence, not its ten timesteps.
- No curve smoothing, GPU collection, mask search, or downstream selection was performed.

## Results: distinct ranking information

At 65%, masked and unmasked marginal costs remain strongly correlated (mean Spearman **0.9614**), so most projection ordering is shared. However, **6.14%** of comparable projection pairs reverse order between roles and the mean absolute percentile contrast is **4.23%**.

Cardinality-matched random partitions show only **2.31%** rank reversals and **1.68%** mean absolute percentile contrast. Thus the actual DLM role split exposes roughly **2.51×** more rank separation than arbitrary groups of the same size.

## Results: sequence stability

The actual role contrast generalizes across leave-one-sequence-out folds with Spearman **0.9431**, sign agreement **94.01%**, and within-layer Spearman **0.8908**. Random partitions have mean LOO Spearman **0.8205** and sign agreement **90.30%**. The actual split therefore creates a larger difference while preserving it more reliably across sequences.

The sequence-cluster bootstrap 95% CI for the actual mean marginal rank correlation is **[0.9600, 0.9631]**; the rank-reversal CI is **[5.88%, 6.32%]**.

## Results: pooling and allocation

Masked energy weight has median **0.460** and range **[0.079, 0.820]** across projections. Aggregate marginal ranking follows unmasked costs more closely (mean Spearman **0.9931**) than masked costs (**0.9788**).

Current Max and Aggregate differ in **20/224** projections and **0.433%** of prunable weights. Changed projections have mean absolute role contrast **10.26%**, versus **3.64%** elsewhere. Max allocation is also stable: leaving out one sequence changes 0–4 projections relative to its full-data allocation.

Layer and projection-type fixed effects explain **28.4%** of per-projection role contrast; **71.6%** remains within layer/type structure. Changed projections span all four layer quartiles and six projection types.

## Interpretation

Masked/unmasked separation carries real, repeatable allocation information. The effect is localized rather than global: most rankings agree, while a stable minority of comparisons and 20 final allocation decisions differ. This matches the earlier observation that token-level signals can vanish in feature-level weight ranking yet still affect the coarser projection budget decision.

These results do not select Max as the correct aggregation rule. They also do not show that reconstruction is uniquely preferable to another cheap damage proxy or that projection-level allocation is necessary for downstream performance. Those remain method-design questions.

## Decision

- Role separation: **supported as stable allocation information**.
- Max aggregation: **not established**.
- Projection-level representation: **supported statistically; downstream necessity not established**.
- Reconstruction: **useful current proxy; uniqueness not established**.
