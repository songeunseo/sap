# Next diagnostic: does corruption change the redundancy judgment?

Status: planned only; no GPU job or new allocation has been launched.

## Hypothesis
The suspect judgment is the mapping from larger mean Wanda score to more pruning in the frozen DLP public mean path. Hypothesis: corrupted denoising inputs, beyond ordinary depth scaling, alter this statistic in a way that changes which blocks are treated as redundant. The nearly invariant low/high-mask block RMS ordering is contrary evidence, so a positive outcome is not assumed.

## Fixed setup and the single new control
Use the exact existing LLaDA revision/BF16 model and eight 256-token calibration sequences. Reuse their eighty frozen corrupted states. Add **eight fully visible versions of those same sequences** solely as the clean control. This adds one control condition; it does not replace calibration for existing baselines. Preserve seeds, token ids, target224 ordering, final Standard Wanda mask ranking, 65% exact pruning budget, DLP alpha=.15 and exact-row allocator.

Collect in dense batch-one full forwards. Before a new long job, check tmux/processes and register a running Obsidian note. A changed input set or measurement path must be recorded before execution. New downstream/full1319 is outside this diagnostic.

## Small measurement
For each projection store channel sums of squares per sequence/state and columnwise mean abs(weight). Their dot product with sqrt(A) computes the public mean Wanda statistic without materializing a full score matrix. Pool projections by weight count to match the original block mean. Reproduce the saved corrupted-state DLP statistic and exact allocated row counts before interpreting a clean/corrupted comparison.

Compare clean vs corrupted block scores, scale-normalized ordering, leave-one-sequence-out ordering, and assigned exact integer row counts. Also compare each corrupted probability separately for diagnosis, without fitting timestep weights. Identical final Wanda ranking remains fixed when deriving diagnostic allocations; only the criterion input condition changes.

## Decision
If clean and corrupted criteria retain essentially the same ordering/allocation, this route does not explain a DLM-specific transfer error; record it and do not manufacture a correction from mask heterogeneity. If corruption produces a stable, localized judgment change, identify those blocks without looking at new GSM8K outcomes, and physically measure only their frozen65→70 pruning damage using batch-one full forwards before designing one correction.

A raw sign-reversed allocation can at most be an empirical direction control. It is not the proposed DLM method. No automatic large interaction study, optimizer search, role aggregation sweep, or downstream grid is needed.
