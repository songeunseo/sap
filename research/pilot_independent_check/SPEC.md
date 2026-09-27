# Independent re-implementation spec — Probe Reliability Pilot

You are verifying another implementation. **Do not read any `.py` file under
`/home/tmluser1/sap/experiments/dlm_probe_reliability_pilot/` or `/home/tmluser1/sap/experiments/dlm_multiscale_ac50/`,
and do not read `result*.json` in the pilot output folder.** Work only from this spec and the data files below.
Write your own code under `/home/tmluser1/sap/research/pilot_independent_check/impl/` and your outputs under
`/home/tmluser1/sap/research/pilot_independent_check/out/`. CPU only (do not use a GPU).

Python: `PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python /usr/bin/python3` has numpy, scipy, torch.

## Data (read-only)
Root `R = /home/tmluser1/sap/experiments/dlm_probe_reliability_pilot/output`

- `R/bank.json`: `chains` is a list of 64 chains (32 sequences × 2 chains). Each chain has
  `sequence_index` (100..131), `chain_index` (0/1), `query` (8 token positions, always masked), and `nodes`
  (8 nodes = phases 0..7, increasingly more context revealed). Each node has `input_ids` (256 tokens);
  masked positions are exactly those with token id `126336`.
- **State order:** state index `s = 8*k + phase`, where `k` is the chain's position in the `chains` list (0..63).
  There are 512 states.
- `R/readouts/<label>.pt`: a `torch.float32` tensor of shape `[512, 256]`. Entry `[s, t]` is the gold-token
  log-odds `f = log p(gold_t) − log(1 − p(gold_t))` at position `t` for state `s`.
  Labels:
  - `dense`: unpruned teacher
  - `uniform50`: background
  - `bBB_rRR`: block `BB` (00..31) pruned at rate `RR` ∈ {40, 45, 48, 52, 55, 60}; every other block stays at the 50% background
- `/home/tmluser1/sap/experiments/dlm_context_response50/uniform/mask_manifest.json`: `entries` is a list of 224
  projections in block order (7 per block). Each has `shape` = [out, in].
  - **Pruned count** of block b at rate r = Σ over its 7 projections of `floor(in * r) * out`.

## Per-state and per-sequence levels (for one condition X, teacher D = dense)
Let `e[s, t] = f_X[s, t] − f_D[s, t]`. Let `M[s, t]` = position t is masked in state s. Let Q = the chain's 8 query positions.

1. **A_q** (per chain) = mean over 8 nodes × 8 query positions of `e²`. Per sequence = mean over its 2 chains.
2. **C_q scale** (per chain), edges:
   - C1 = (0,1),(2,3),(4,5),(6,7)
   - C2 = (0,2),(1,3),(4,6),(5,7)
   - C4 = (0,4),(1,5),(2,6),(3,7)

   For each edge (i,j) (node indices within the chain), take the mean over the 8 query positions of `(e[j] − e[i])²`.
   C_scale = mean over its 4 edges. **Multi_q** (per chain) = A_q + (C1 + C2 + C4)/3. Per sequence = mean over chains.
   **Cpart_q** = Multi_q − A_q.
3. **A_m** (per state) = mean over masked positions of `e²`. Per sequence = mean over its 16 states.
4. **CE_m** (per state) = mean over masked positions of `log(1 + exp(−f_X))`. Per sequence = mean over its 16 states.
5. **C_m** (per chain): for each edge (i,j), take the mean of `(e[j] − e[i])²` over positions masked in BOTH node i and node j.
   Then mean over edges within a scale, then mean of C1/C2/C4. Per sequence = mean over chains.
   **Cpart_m** = C_m. **Multi_m** = A_m + C_m (per sequence).

## Costs
δ ∈ {2: (48, 52), 5: (45, 55), 10: (40, 60)}. For metric J, block b, sequence s:
`cost[b, s] = (J(b, hi)[s] − J(b, lo)[s]) / (pruned(b, hi) − pruned(b, lo))`. This gives a 32 × 32 matrix (blocks × sequences).

## Statistics per (metric, δ)
1. **Two-way reliability** (exact, deterministic):
   - `resid = c − rowmean − colmean + grandmean`
   - `W = Σ resid² / ((B−1)(S−1))`
   - `O = var(rowmeans, ddof=1)`
   - `T = O − W/S`
   - `rel = max(T, 0) / (max(T, 0) + W/S)`
   - `rel8 = max(T, 0) / (max(T, 0) + W/8)`
   - `spans_for_0.8 = 4W/T` if T > 0, else null
2. **Random half-split:** 500 random permutations of the 32 sequences, split 16/16, Spearman between the two
   halves' block means. Report the mean. Use your own RNG seed; results are compared with tolerance.
3. **Depth:**
   - For each sequence, Spearman(block index 0..31, cost[:, s]). Report the mean and the number positive.
   - Spearman(block index, row means).
   - R² of the least-squares linear fit (intercept + block index) to the row means.
4. **Depth residual:** for each sequence (column), subtract its least-squares linear fit on block index. Apply the
   two-way reliability (item 1) to the residual matrix.

Metrics to cover: A_q, Multi_q, Cpart_q, A_m, CE_m, Multi_m, Cpart_m (C_m) × δ ∈ {2, 5, 10}.

## Also report (sanity)
- For block 0 and block 31: the per-sequence mean of A_m for each of the 6 rates and for `uniform50`.
- The fraction of blocks where A_m (mean over sequences) is monotone non-decreasing across rates 40 < 45 < 48 < 52 < 55 < 60.

## Output
Write `out/levels.json` with `{"<metric>|b<BB>|r<RR>": [32 per-sequence values in sequence_index order]}` for all
metrics A_q, Multi_q, A_m, CE_m, C_m, and all blocks and rates.
Write `out/stats.json` with, for each `"<metric>|d<δ>"`: rel, rel8, T, W, spans_for_0.8, half_rho_mean,
depth_rho_mean, depth_positive, meancost_depth_rho, meancost_depth_r2, resid_rel, resid_rel8.
Write `out/sanity.json` and a short `out/NOTES.md` listing any ambiguity you had to resolve and how you resolved it.
