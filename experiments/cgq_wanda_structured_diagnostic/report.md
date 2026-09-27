# CGQ–Wanda Redundancy at Functional Structured Units

## Scope and integrity

This was an analysis-only diagnostic. No thresholding, pruning mask, weight modification, pruned checkpoint, or downstream evaluation was produced. Step 10 was not run.

- Model: `GSAI-ML/LLaDA-8B-Base`, revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, loaded as bfloat16.
- Frozen calibration: EXP-001 run `20260828T175010-2484545`, 80 states (8 samples × 10 timesteps), state SHA-256 `1e44d30e845aee6a7b9292c93b55e2e2373c5ac9cd7f2077edb311ce041023df`.
- Environment: Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.49.0.
- EXP-004 masked-confidence reproduction max absolute error: `0.0`.
- Model weight SHA-256 before collection, after collection, and after analysis: `ea78a6ff576b2ba4b6961a3fdfe00e138e8a85cdab32b7fdc47e3b25c3e5e13b` in all three checks.
- Persisted sufficient statistics: `sufficient_statistics.pt` (SHA-256 `81e121286f150f5dfad15a7e0c28bbecda6a6a2b2b820c191a6e024b6f528f51`). It contains FP32 aggregate and 10-timestep `A/A_CGQ` vectors for all 224 Linear modules. Overall statistics are means over 80 batch examples; timestep statistics are means over the corresponding 8 examples.

## Artifact-supported facts

### Verified architecture and grouping

The loaded implementation uses 32 `LLaDALlamaBlock`s, `d_model=4096`, MLP hidden size 12,288, 32 Q heads, 32 K heads, 32 V heads, and `head_dim=128`. There is no grouped-query ambiguity. Q/K/V outputs reshape from `[B,T,4096]` to `[B,32,T,128]`; concatenated head outputs feed `attn_out`.

Parameter shapes per block are Q/K/V/attn_out `[4096,4096]`, ff_proj/up_proj `[12288,4096]`, and ff_out `[4096,12288]`. The MLP path is:

`ff_out(silu(ff_proj(ff_norm(x))) * up_proj(ff_norm(x)))`.

The diagnostic score for a weight was `S(i,j)=|W(i,j)| sqrt(A_j)`, separately using uniform and CGQ statistics. The primary group score was the raw additive sum, with no component normalization:

- neuron `k`: `ff_proj[k,:] ∪ up_proj[k,:] ∪ ff_out[:,k]`;
- head `h`: matching 128-row Q/K/V output slices and the matching 128-column `attn_out` input slice.

CGQ used exactly `r=1+sqrt(c)` for masked tokens and `r=0.7+sqrt(c)` for unmasked tokens, so `A_CGQ_j = Σ r_n² X_nj²` under the repository-equivalent accumulator normalization.

### MLP neurons

Across the 32 layers, layer-wise `G_uniform` versus `G_CGQ` had mean Spearman `0.998442` (min `0.997004`), mean Kendall tau `0.972029`, and mean `CV(q_neuron)` `0.007147` (median `0.005953`, max `0.017271`). Mean absolute rank movement was 114.46 of 12,288 neurons, or `0.009315` normalized. The mean of each layer's structured-CV / ff_out-feature-CV ratio was `0.1344`: functional grouping retained only about 13.4% of the already observed ff_out feature-ratio variation.

Component attribution, averaged over layer means:

| Component | Uniform G fraction | CGQ G fraction | ΔG fraction | Component CGQ/uniform | Share of absolute mean-q deviation |
|---|---:|---:|---:|---:|---:|
| ff_proj | 39.95% | 39.90% | 39.82% | 1.61504 | 24.83% |
| up_proj | 40.64% | 40.61% | 40.55% | 1.61558 | 27.14% |
| ff_out | 19.41% | 19.50% | 19.62% | 1.62544 | 48.03% |

Thus total added saliency follows component mass, but the small residual non-uniform group ratio is disproportionately associated with ff_out: about 48% of mean-q deviation despite only about 19% of group saliency. This is not coherent three-component movement.

### Attention heads

Across layers, mean Spearman was `0.995704` (min `0.983871`), mean Kendall tau `0.969002`, and mean `CV(q_head)` `0.003932` (median `0.003077`, max `0.009028`). Mean absolute rank movement was 0.441 of 32 heads, or `0.013794` normalized. The mean structured-CV / attn_out-feature-CV ratio was `0.0771`: head grouping retained about 7.7% of attn_out feature-ratio variation.

Component attribution, averaged over layer means:

| Component | Uniform G fraction | CGQ G fraction | ΔG fraction | Component CGQ/uniform | Share of absolute mean-q deviation |
|---|---:|---:|---:|---:|---:|
| q_proj | 27.20% | 27.23% | 27.27% | 1.61425 | 17.90% |
| k_proj | 27.94% | 27.95% | 27.98% | 1.61361 | 18.76% |
| v_proj | 26.20% | 26.27% | 26.39% | 1.61636 | 20.29% |
| attn_out | 18.65% | 18.55% | 18.36% | 1.59966 | 43.05% |

Again, ΔG is broadly proportional to component mass, while the remaining non-uniformity is overrepresented in attn_out: about 43% of mean-q deviation despite about 19% of saliency. Q/K/V do not show coherent head-specific movement with it.

### All-layer localization and rank-set stability

`nMAD` is mean absolute rank displacement divided by unit count. Set cells are `overlap / Jaccard / crossed-boundary count`; crossing counts include departures plus arrivals. The compact table shows the most selective requested boundary (5% for neurons, 4 heads for attention); all 5/10/20% neuron and 4/8-head top/bottom results, including exact crossing counts, are retained in `artifact_manifest.json`.

| Layer | MLP ρ | MLP CV | MLP nMAD | neuron top-5% | neuron bottom-5% | Head ρ | Head CV | Head nMAD | head top-4 | head bottom-4 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | .99998 | .00355 | .00109 | .989/.977/14 | .992/.984/10 | .99853 | .00592 | .00781 | 1/1/0 | 1/1/0 |
| 1 | .99993 | .00510 | .00183 | .977/.955/28 | .990/.981/12 | .99890 | .00534 | .00586 | 1/1/0 | 1/1/0 |
| 2 | .99983 | .00476 | .00269 | .977/.955/28 | .998/.997/2 | .99927 | .00240 | .00391 | 1/1/0 | 1/1/0 |
| 3 | .99963 | .00551 | .00430 | .976/.952/30 | .993/.987/8 | .99963 | .00366 | .00195 | 1/1/0 | 1/1/0 |
| 4 | .99912 | .00518 | .00668 | .971/.943/36 | .985/.971/18 | .99927 | .00304 | .00391 | 1/1/0 | .75/.60/2 |
| 5 | .99942 | .00412 | .00557 | .967/.937/40 | .985/.971/18 | .99890 | .00215 | .00586 | 1/1/0 | 1/1/0 |
| 6 | .99851 | .00564 | .00911 | .961/.925/48 | .976/.952/30 | .99890 | .00279 | .00586 | 1/1/0 | 1/1/0 |
| 7 | .99862 | .00512 | .00827 | .956/.916/54 | .969/.940/38 | .99743 | .00305 | .00977 | 1/1/0 | 1/1/0 |
| 8 | .99837 | .00523 | .00880 | .966/.934/42 | .972/.946/34 | .99523 | .00354 | .01367 | 1/1/0 | 1/1/0 |
| 9 | .99835 | .00504 | .00935 | .953/.910/58 | .971/.943/36 | .99853 | .00317 | .00586 | 1/1/0 | 1/1/0 |
| 10 | .99823 | .00519 | .00997 | .951/.907/60 | .976/.952/30 | .99597 | .00303 | .01367 | 1/1/0 | .75/.60/2 |
| 11 | .99788 | .00531 | .01105 | .946/.898/66 | .967/.937/40 | .99670 | .00337 | .01367 | 1/1/0 | 1/1/0 |
| 12 | .99826 | .00534 | .01006 | .959/.922/50 | .972/.946/34 | .99487 | .00272 | .01563 | 1/1/0 | 1/1/0 |
| 13 | .99825 | .00545 | .01000 | .956/.916/54 | .971/.943/36 | .99010 | .00310 | .02734 | 1/1/0 | .50/.333/4 |
| 14 | .99816 | .00626 | .01020 | .951/.907/60 | .971/.943/36 | .99633 | .00247 | .01367 | 1/1/0 | .75/.60/2 |
| 15 | .99820 | .00673 | .01049 | .948/.901/64 | .966/.934/42 | .99817 | .00263 | .00781 | 1/1/0 | 1/1/0 |
| 16 | .99824 | .00611 | .00976 | .954/.913/56 | .969/.940/38 | .99670 | .00274 | .01367 | 1/1/0 | 1/1/0 |
| 17 | .99828 | .00590 | .00982 | .959/.922/50 | .971/.943/36 | .99743 | .00207 | .01172 | .75/.60/2 | 1/1/0 |
| 18 | .99832 | .00583 | .00986 | .950/.904/62 | .963/.928/46 | .99853 | .00291 | .00781 | 1/1/0 | 1/1/0 |
| 19 | .99813 | .00600 | .01032 | .967/.937/40 | .967/.937/40 | .99707 | .00282 | .01172 | 1/1/0 | 1/1/0 |
| 20 | .99838 | .00632 | .00994 | .946/.898/66 | .959/.922/50 | .99670 | .00219 | .01563 | 1/1/0 | 1/1/0 |
| 21 | .99829 | .00642 | .00999 | .964/.931/44 | .958/.919/52 | .99743 | .00264 | .00977 | 1/1/0 | 1/1/0 |
| 22 | .99844 | .00672 | .00958 | .943/.892/70 | .950/.904/62 | .99340 | .00366 | .01563 | 1/1/0 | 1/1/0 |
| 23 | .99836 | .00761 | .00962 | .966/.934/42 | .954/.913/56 | .99890 | .00349 | .00586 | 1/1/0 | 1/1/0 |
| 24 | .99839 | .00828 | .00990 | .966/.934/42 | .963/.928/46 | .99633 | .00286 | .01367 | 1/1/0 | 1/1/0 |
| 25 | .99817 | .00907 | .01109 | .956/.916/54 | .967/.937/40 | .99487 | .00544 | .01758 | 1/1/0 | 1/1/0 |
| 26 | .99808 | .00983 | .01180 | .948/.901/64 | .969/.940/38 | .98387 | .00620 | .03516 | 1/1/0 | .75/.60/2 |
| 27 | .99831 | .01081 | .01154 | .958/.919/52 | .961/.925/48 | .98790 | .00711 | .03516 | .75/.60/2 | 1/1/0 |
| 28 | .99796 | .01135 | .01301 | .951/.907/60 | .958/.919/52 | .99487 | .00706 | .02148 | 1/1/0 | 1/1/0 |
| 29 | .99753 | .01226 | .01431 | .946/.898/66 | .959/.922/50 | .98974 | .00542 | .02539 | .75/.60/2 | 1/1/0 |
| 30 | .99700 | .01539 | .01527 | .940/.886/74 | .936/.881/78 | .98570 | .00781 | .03320 | 1/1/0 | 1/1/0 |
| 31 | .99752 | .01727 | .01278 | .958/.919/52 | .945/.895/68 | .99633 | .00903 | .01172 | 1/1/0 | 1/1/0 |

Late-layer localization remains visible, especially MLP layers 30–31 and attention layers 26–31. It is nevertheless much smaller after functional grouping. For example, neuron CV peaks at 0.01727, while mean structured/ff_out feature CV retention is 13.4%; head CV peaks at 0.00903 while mean structured/attn_out feature CV retention is 7.7%. Attention top/bottom four sets are usually unchanged; the largest listed boundary event is four crossings at layer 13 bottom-4. Layers 27 and 30 show limited top-4/top-8 changes, not broad reordering.

## Interpretation

### MLP decision: STRUCTURED WASHOUT

The functional neuron score is more uniform than the already-known ff_out neuron-activation feature statistic. Its ranking remains almost identical, boundary membership is highly stable, and incoming ff_proj/up_proj do not move coherently with the ff_out residual. There is a component-localized ff_out residue, but it is too diluted at the full unit level to classify the result as structured retention.

### Attention decision: STRUCTURED WASHOUT

Grouping 128 channels and all Q/K/V/O slices averages away most of the localized attn_out signal. A few late layers have measurable head rank movement, but attn_out is disproportionately responsible for the residual while Q/K/V do not change coherently. This is weak pseudo-structure inherited from attn_out, not evidence of head-wide specialization.

### Overall conclusion

CGQ information does **not** survive better at functional structured granularity than at ordinary Wanda feature granularity under raw additive Wanda saliency. Both unit families show stronger common-scaling behavior after grouping. The evidence therefore does not empirically motivate a CGQ structured-pruning experiment.

Step 10's optional timestep stability gate was not triggered: Steps 1–9 showed washout rather than materially non-uniform structured retention. The saved timestep statistics remain available without another model pass.

## Speculation

The late-layer residual could reflect specialization confined to output-projection channels rather than whole neurons or heads. This diagnostic cannot establish mechanism or predict downstream pruning quality; it only shows that the residual does not aggregate coherently under the specified functional groups.

## One recommended next step

Stop the CGQ structured-pruning line under this raw-additive Wanda grouping; do not launch a structured pruning or downstream evaluation experiment from these results.
