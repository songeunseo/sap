# Multi/A sample sensitivity audit

Frozen paired GSM8K answer rows were compared by question ID. Positive difference favors Multi.

| NFE | Sample | n | A | Multi | Difference | Gains/losses | Exact McNemar p | Paired bootstrap 95% CI (pp) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 256 | development100 | 100 | 60 | 63 | +3.00 pp | 6/3 | 0.5078 | [-3.00, +9.00] |
| 256 | separate100 | 100 | 58 | 59 | +1.00 pp | 6/5 | 1 | [-5.00, +7.00] |
| 256 | exposed200 | 200 | 118 | 122 | +2.00 pp | 12/8 | 0.5034 | [-2.50, +6.50] |
| 256 | remaining1119 | 1119 | 646 | 636 | -0.89 pp | 64/74 | 0.4437 | [-2.86, +1.16] |
| 256 | full1319 | 1319 | 764 | 758 | -0.45 pp | 76/82 | 0.6909 | [-2.35, +1.44] |
| 64 | development100 | 100 | 37 | 42 | +5.00 pp | 11/6 | 0.3323 | [-3.00, +13.00] |
| 64 | separate100 | 100 | 38 | 32 | -6.00 pp | 11/17 | 0.3449 | [-16.00, +4.00] |
| 64 | exposed200 | 200 | 75 | 74 | -0.50 pp | 22/23 | 1 | [-7.00, +6.00] |

## Provenance and reproduction

- Exposed200 matches the frozen full-run exposed IDs. Development100 is IDs 0–99; separate100 is exactly the other 100 exposed IDs; remaining1119 is their complement in 0–1318.
- The full-run NFE256 generated strings reproduce all historical development100 and separate100 strings for both A and Multi (100/100 each of four arms/samples). Historical correctness is A 60/58 and Multi 63/59.
- Each paired row has matching question, prompt, target and reference hashes; NFE64 rows also match the corresponding NFE256 questions. Source file SHA-256 digests and all numeric results are in summary.json.
- NFE64 has no rows on the remaining1119; its full-benchmark effect is unmeasured.

## Interpretation

At NFE256, the development100 advantage is +3 questions, separate100 is +1, and remaining1119 is −10. The full1319 difference is −6 (−0.45 percentage points). At NFE64, the same exposed200 splits +5 and −6, yielding −1 overall. Thus the sign change can arise from ordinary question-to-question variation around a small observed effect. The paired intervals are wide and all include zero.

Limiting early screening to 100 questions was reasonable for reducing generation cost, but the result was too uncertain to establish that Multi was better. Reusing and selecting methods on those questions can favor a noisy positive result; these data cannot determine how much selection contributed. The separate100 was already frozen and improves the picture, but it was also previously exposed development data. The remaining1119 at NFE256 is the strongest sample for judging that setting, and it does not support a Multi advantage. NFE64 and NFE256 differ in forward steps, so their rankings answer different questions.

The NFE32 follow-up is exploratory because the step count was selected after the prior Dense/A and Multi64 results. Its result should be read on that basis.
