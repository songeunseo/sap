# Two 50% mini experiments — completion review

Checked: 2026-09-17 13:45 KST. Both pipelines exit0, status complete.
Verification: `research/two_mini50_completion_verification_2026-09-17.json`.
1,447 unique file SHA256 checks, frozen sources/inputs, 224-mask manifests per built model,
prediction hashes, 100 identities/protocol/strict-EM, exact3,489,660,928 pruning verified.
Paired McNemar recomputed; all48 NELBO article rows and bootstrap intervals verified.
Uniform predictions identical between experiments; GPU1 Uniform NELBO per-draw reproduction difference0.

## Experiment 1: Context response

| Method | GSM8K mini100 |
|---|---:|
| Uniform50 | 54 |
| Endpoint A | 55 |
| A+C context response | 61 |

- Primary AC vs A: +6, 9 newly correct /3 newly wrong, exact McNemar p=.14599609375.
- AC vs Uniform: +7, 13 gained /6 lost, raw p=.1670684814453125, Holm2=.334136962890625.
- A vs Uniform: +1, 8 gained /7 lost, p=1.
- No NELBO evaluation was run for A/AC. A/C calibration distortion is not NELBO.
- Interpretation: positive development screen for adding response preservation; statistical superiority not established.
- Gold context reveal also changes mask count, so content-use/progress mechanisms remain confounded.

## Experiment 2: Shared state support coverage

| Method | GSM8K mini100 | NELBO (16 articles) | exp(NELBO) |
|---|---:|---:|---:|
| Uniform50 |54|2.3923108160929445|10.938742181258208|
| Pooled support90% |50|2.392656926280111|10.942528846625857|
| Each-state coverage90% |54|2.391328859939051|10.928006088106518|

- Primary Coverage vs Pooled GSM: +4,8 gained/4 lost, p=.3876953125.
- Coverage vs Uniform: net0,5 gained/5 lost,p=1.
- Coverage minus Pooled NELBO: -.0013280663410597526; descriptive unadjusted article bootstrap95%[-.0022211758027886217,-.00036761595196743745],13/16 articles improve.
- Coverage minus Uniform NELBO: -.000981956153893293;95%[-.0032807855324744396,.0013236964137220697],10/16 articles improve.
- Interpretation: small positive NELBO signal against its exact pooled control; no established Uniform advantage, GSM superiority, or general active-set/static-weight mechanism.
- The16 articles are a previously used development subset, not full WikiText or new independent test. Numbers cannot be directly compared to the551-chunk full-validation table.

## Decision / scope

Both experiments complete; preserve configurations/results. No automatic tuning, full evaluation, or new GPU work.
AC has the stronger GSM point estimate; coverage has a small NELBO improvement over its pooled control.
These different signals do not establish a common ranking of the two methods on NELBO.

## Obsidian synchronization pending

Obsidian MCP is not callable in this turn. Do not claim the vault was updated.
Pending notes to set complete and append these results:
- Experiments/2026-09-16-Context-Response50-GSM8K-Mini100.md
- Experiments/2026-09-17-Shared-State-Support-Coverage50-Mini100.md
- Research/DLM-Pruning/Research-State.md
