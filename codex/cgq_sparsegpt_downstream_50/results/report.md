# LLaDA CGQ SparseGPT WinoGrande validation

## Scope

This run evaluates only the preregistered 50% Plain SparseGPT versus frozen
CGQ-SparseGPT comparison on WinoGrande. GSM8K was not run. The existing Dense
WinoGrande result was reused as authorized; the old 68.90% Plain SparseGPT
result was excluded because it used a different calibration protocol.

The new Plain and CGQ models used the pinned LLaDA-8B-Base revision
`0f2787f2d87eac5eed8a087d5ecd24277e6255b2` and the exact 16 cached corrupted
states from the earlier CGQ experiment. Their calibration-state digest was
`b69643776e06bdba6b1928d18108fde5b4b6e1bfb7558ceda195b753c40a364e`.

WinoGrande used all 1,267 validation examples, 5-shot prompting, CFG 0,
128 Monte Carlo samples, batch size 8, random seed 0, and NumPy, Torch, and
few-shot seeds 1234. Document, prompt, and target hashes matched across Dense,
Plain, and CGQ.

## Results

| Model | Sparsity | Correct / total | WinoGrande accuracy |
|---|---:|---:|---:|
| Dense (reused) | 0% | 945 / 1,267 | 74.59% |
| Plain SparseGPT | 50.00015% | 875 / 1,267 | 69.06% |
| CGQ-SparseGPT | 50.00015% | 863 / 1,267 | 68.11% |
| CGQ − Plain | — | −12 | −0.95 pp |

## Pruning verification

| Diagnostic | Value |
|---|---:|
| Plain zero weights | 3,489,671,637 / 6,979,321,856 |
| CGQ zero weights | 3,489,671,694 / 6,979,321,856 |
| Plain–CGQ mask XOR | 562,287,443 / 6,979,321,856 (8.05648%) |
| Cached calibration states | 16 |

The reproduced mask XOR differs from the earlier 8.05648% result only at
floating-point display precision, so the downstream comparison uses the
intended sparse solutions.

## Paired outcomes

| Outcome | Examples |
|---|---:|
| Plain wrong → CGQ correct | 61 |
| Plain correct → CGQ wrong | 73 |
| Both correct | 802 |
| Both wrong | 331 |

## Conclusion

On WinoGrande, the earlier improvement in held-out masked-token KL and
agreement did not transfer to higher downstream accuracy. CGQ changed the
mask materially but scored 0.95 percentage points below the newly calibrated
Plain SparseGPT control, with 12 fewer correct examples. This single-task run
does not establish behavior on GSM8K or support a broader downstream claim.
