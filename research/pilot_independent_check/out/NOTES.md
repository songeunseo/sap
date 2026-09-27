# Independent check: notes

Code: `impl/verify.py`. CPU only, float64 arithmetic after loading the float32 readouts. Runtime about 3 s (5 s wall including interpreter start-up).
Inputs read: `bank.json`, `readouts/*.pt` (not the `readouts/*.json` sidecars), and the uniform `mask_manifest.json`. Nothing else.

## Data checks (asserted in the code)
- The chains are ordered `(seq 100, chain 0), (100, 1), (101, 0), ...`. `node["phase"]` equals the node's list index. State `s = 8*k + phase`.
- All 8 query positions are masked in every node of their chain.
- The masks are nested across phases: the masked set of phase i+1 is a subset of the masked set of phase i. So "masked in both i and j" is the same as "masked in node max(i, j)", and it is never empty.
- Every block has the same pruned count at a given rate: 40: 87,224,320; 45: 98,136,064; 48: 104,685,568; 52: 113,373,184; 55: 119,922,688; 60: 130,834,432.
- Dense vs dense gives A_m = 0.

## Ambiguities and how I resolved them
1. **floor(in * r):** I used exact integer arithmetic, `in*RR//100`. None of the products `in*r` is an integer, so this matches the float floor.
2. **Per-sequence order:** I sorted by `sequence_index` (100..131). This is the same as the chain-list order.
3. **A_m and CE_m aggregation:** I took the mean over masked positions within each state, then an equal-weight mean over the 16 states of the sequence. Positions are not pooled across states.
4. **C_m and C_q aggregation:** each edge is the mean over its positions. Then I took an equal-weight mean over the 4 edges of a scale, then over the 3 scales, per chain, then an equal-weight mean over the 2 chains.
5. **CE_m:** computed as `logaddexp(0, -f_X)`, which is a numerically stable `log(1+exp(-f))`. It uses f_X only (not e), as written.
6. **Cpart_q and Multi_m:** Cpart_q = per-sequence mean of the per-chain (C1+C2+C4)/3, which equals Multi_q - A_q. Multi_m = A_m + C_m per sequence. Their costs come from these per-sequence levels, and the result is identical to differencing the costs.
7. **levels.json:** contains only the 5 metrics the spec lists (A_q, Multi_q, A_m, CE_m, C_m) for 32 blocks x 6 rates. No uniform50 or dense entries are included.
8. **Two-way reliability:** applied literally. If both denominators were 0, rel would be null; this never happened. Output key is `"spans_for_0.8"`.
9. **Half-split:** I used `numpy.random.default_rng(12345)` and 500 permutations. Half 1 is the first 16 entries of the permutation and half 2 is the last 16. For each half I took the mean over its sequences for every block, then the Spearman correlation (scipy, average ranks) between the two 32-vectors. I report the mean over the 500 permutations (nanmean; 0 NaNs occurred).
10. **Depth:** Spearman(0..31, cost[:, s]) per sequence. `depth_positive` counts rho > 0 strictly. `depth_rho_mean` is the plain mean over the 32 sequences. R^2 = 1 - SS_res/SS_tot of the OLS fit (intercept + block index) to the row means.
11. **Depth residual:** for each column I fitted OLS with intercept + block index and subtracted the fit, then applied the same two-way formula without correcting degrees of freedom for the fitted parameters (literal reading). After this step the column means are ~0 by construction. I also report resid_T and resid_W.
12. **Sanity "per-sequence mean of A_m":** read as the mean over the 32 sequences of the per-sequence A_m. uniform50 is the same condition for block 0 and block 31. "Monotone" means non-decreasing across the 6 block-specific rates only (uniform50 excluded, literal reading). As a supplement I also computed the fraction with uniform50 inserted between 48 and 52; it is the same (21/32).
13. **Extra fields beyond the spec (clearly named):** `half_rho_nan_count`, `resid_T`, `resid_W`, `pruned_delta_b00` in stats.json, and the pruned counts in sanity.json.
