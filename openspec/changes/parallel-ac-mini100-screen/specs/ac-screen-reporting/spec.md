# Spec Delta

## Purpose

Produces auditable mini100 development comparisons that separate completed and partial measurements, pair question-level outcomes, expose computational cost, and avoid unsupported claims of mechanism or generalization.

## ADDED Requirements

### Requirement: Honest completeness and identity
Reports SHALL include each arm's actual correct/evaluated counts, completion state, mask identity, bank/readout/allocator and result provenance. A final mini100 accuracy SHALL require all 100 expected validated documents. Partial cross-arm comparisons SHALL use the exact intersection of compatible document IDs and identify that subset explicitly. Row-wise native Uniform54, historical cached Uniform62 and layer-global Uniform SHALL never share an ambiguous label.

#### Scenario: Two arms have only 56 results
- **WHEN** a report is requested before their evaluations complete
- **THEN** it reports their observed results out of 56 and common-ID comparisons, without extrapolating them to mini100 accuracy

### Requirement: Predeclared paired comparisons
The report SHALL predeclare the seven contrasts Multi versus Short, MS-A, Path and All; Square-AC versus Square-A; Vector-AC versus Vector-A; Exchange-AC versus legacy AC. It SHALL show gains/losses, exact paired McNemar p-values, and Holm correction across the fixed seven contrasts once all are complete. Additional legacy comparisons SHALL be labeled exploratory. Holm adjustment SHALL NOT be described as undoing prior adaptive reuse of the development questions or establishing unseen-data confirmation. An incomplete family SHALL not be presented as a completed multiplicity-adjusted finding.

#### Scenario: All arms complete
- **WHEN** all ten arms and required references have valid mini100 results
- **THEN** the report publishes the fixed paired-comparison family and a descriptive score ranking, without converting a point-score lead into established superiority

### Requirement: Cost accounting and noncomparable objectives
Reports SHALL separate calibration, teacher setup, probes/search, diagnostics and generation costs, with actual transformer forwards, completed candidate evaluations, GPU-seconds, wall time and peak memory where measured. Imported historical work SHALL be labeled reused; missing historical timings SHALL remain unknown. Scalar and vector objective magnitudes MUST NOT be ranked against each other. New objective quality SHALL be assessed against matched controls and common task results.

#### Scenario: A vector arm has a smaller raw objective
- **WHEN** vector and scalar objective values use different units
- **THEN** the report displays their definitions separately and does not treat their raw numerical ordering as a quality comparison

### Requirement: Complete development report and bounded interpretation
Reports SHALL keep Hypothesis, Setup, Result, Interpretation, and Decision distinct, include per-question output links, and label the reused mini100 as development. A plan-conformant negative result SHALL count as a valid completed experiment. Exact-match success SHALL not be equated to valid intermediate reasoning or proof of a denoising mechanism. A report SHALL not start a follow-up experiment.

#### Scenario: No new arm improves over legacy AC
- **WHEN** the complete screen has no point-score gain over legacy AC
- **THEN** the report records the negative result and remaining uncertainty without tuning configurations or launching another phase

### Requirement: Theory-linked diagnostics and claim boundaries
Reports SHALL map each hypothesis to its matched comparison using design §0.8 and preserve the source ledger. Multi SHALL report A, C1, C2, C4, C_path and C_all separately; Square SHALL report A/C and averaged squared common/main/interaction residual modes; Vector SHALL report A/C with its vocabulary and per-pair normalization; Exchange SHALL report actual incumbent/candidate losses, predicted gains, accepted moves, covered directions and termination reason. Diagnostic-bank metadata SHALL distinguish Multi's original 128 states from the new Square/Vector recipes. No raw scalar/vector loss ranking or theory-derived GSM8K guarantee is permitted.

#### Scenario: A theoretical sign bound is mentioned
- **WHEN** a report includes the response-sign inequality from design §0.3
- **THEN** it distinguishes its Multi-edge unconditional event from the legacy all-pair conditional flip statistic and does not use the latter to claim the inequality was empirically verified

#### Scenario: A capped exchange produces no gain
- **WHEN** Exchange finishes without an accepted improvement
- **THEN** the report names the old-cost shortlist, warm start, step size and explored candidate count and limits the conclusion to that bounded search

### Requirement: Distinguish observation from causal attribution
Cross-family task comparisons SHALL be labeled end-to-end screens when banks, readout geometry, query exposure or optimization compute differ. The four Multi contrasts SHALL identify what each control tests, including Path's unequal node degrees. A Vector win over scalar AC SHALL NOT by itself establish the value of C; an Exchange win SHALL NOT establish C's necessity under search. Deferred DKD SHALL be reported as untested, not inferior or implemented.

#### Scenario: Two methods have equal state-forward counts
- **WHEN** Square and Vector each use 160 calibration states
- **THEN** the report does not call their query counts, information content or measured compute identical solely from that count
