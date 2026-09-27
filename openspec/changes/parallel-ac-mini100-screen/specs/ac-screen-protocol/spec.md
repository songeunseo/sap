# Spec Delta

## Purpose

Defines a reproducible development comparison of conditional-response pruning candidates, keeping objective changes separate from allocation changes and preserving exact model, data, sparsity, and evaluation identities.

## ADDED Requirements

### Requirement: Frozen experiment matrix
The system SHALL freeze ten arms: MS-A, Short, Path, All, Multi, Square-A, Square-AC, Vector-A, Vector-AC, and Exchange-AC. Legacy native Uniform/A/AC mini100 scores SHALL be labeled cached references, not newly measured task results. Fresh new-bank calibration or diagnostic measurements of those masks SHALL be labeled separately and charged as new work. The manifest SHALL declare each arm's bank, readout, objective version, edge weights, reduction denominators, allocator, seed, source hashes, and matched control. Imported receipts SHALL retain their original schema and receive a separate compatibility mapping. Candidate combinations, coefficient sweeps, and soft-mask training MUST NOT be added implicitly.

#### Scenario: Preparing a screen
- **WHEN** a user prepares the default screen
- **THEN** its manifest contains exactly the ten declared arms and the three separately identified historical references
- **AND** the report identifies MS-A as a new-bank endpoint control, not legacy A

### Requirement: Frozen model and evaluation protocol
The system SHALL use LLaDA-8B-Base revision `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`, original surviving weights, frozen native Wanda row rankings, 224 prunable matrices, and one static mask per model. Every final mask MUST remove exactly 3,489,660,928 of 6,979,321,856 prunable weights. Evaluation SHALL use the existing development IDs 0–99 with exact original prompt/token/target/protocol identities, 5-shot native generation, 256 tokens, 256 steps, temperature zero, low-confidence remasking, no CFG, batch one, and original strict-match grading.

#### Scenario: A mask or prompt is incompatible
- **WHEN** a candidate has a different pruning count or a cached prediction has a different prompt, model, or evaluation identity
- **THEN** the system rejects it before publishing a comparable result and names the mismatching field

### Requirement: Matched objective comparisons
Each A-only/AC pair SHALL share all states, queries, dense references, pruning probes, allocator, and aggregation weights. Square arms SHALL use the four-state endpoint mean and mean error over the four single-group reveal edges. Vector arms SHALL use vocabulary-mean centered-logit squared residuals at endpoints and across the same legacy before/after pairs. Each active AC arm SHALL have response coefficient one; Multi SHALL average C1, C2 and C4 with weights 1/3. There SHALL be no coefficient search, inverse-gap normalization, or hidden transfer of these coefficients to a probability/KL objective. Multi SHALL retain its original bank and exact edge means, as defined in design §0.3.

#### Scenario: Shared measurements produce two objectives
- **WHEN** one sparse candidate's complete readouts are collected for Square or Vector
- **THEN** both its A and AC scores are derived from those same measurements without a second transformer sweep
- **AND** the system rejects revealed queries, nonfinite values, incomplete states, or incompatible teacher readouts

### Requirement: Bounded exact-budget exchange
Exchange-AC SHALL start from the historical AC physical mask and optimize its unchanged scalar objective on the original 80 pairs. It SHALL evaluate at most eight feasible unique proposals per round for at most three rounds, accept only a measured whole-model improvement beyond a frozen numerical tolerance, and stop when no offered candidate improves or the budget is exhausted. Moves SHALL transfer exactly equal weight counts between two blocks, retain frozen per-row rankings, and stay within the declared bounds. No unrelated block adjustment or proxy-only acceptance is permitted.

#### Scenario: No improving exchange
- **WHEN** all proposals in a round are worse, tied within tolerance, or infeasible
- **THEN** the incumbent remains unchanged and search terminates with its reason and spent budget recorded
- **AND** an unchanged physical mask reuses the incumbent's verified mini100 predictions

### Requirement: Separation of selection and evaluation
Calibration SHALL be the only source for proposal ranking, allocation, and exchange acceptance. Diagnostic and GSM8K results MUST NOT choose masks, coefficients, search stopping times, or query groups. New-bank diagnostic measurements SHALL be reported as development diagnostics, not pristine independent confirmation. The screen MUST NOT launch full GSM8K or another 100-question phase automatically.

#### Scenario: A candidate scores poorly on mini100
- **WHEN** its mini100 score is below another arm
- **THEN** its frozen configuration and remaining evaluation schedule are retained, without automatic tuning or early termination based on accuracy

### Requirement: Traceable theoretical scope
The implementation documentation and scientific manifest SHALL distinguish published ingredients, direct mathematical adaptations, untested pruning hypotheses and numerical defaults using design §0.1 and §0.8. Centered-logit Vector and DKD SHALL remain distinct definitions. The active screen SHALL retain ten arms; deferred DKD, adaptive exchange and A-floor variants MUST NOT execute or replace an arm implicitly. A source reference MUST NOT be represented as a guarantee of GSM8K improvement.

#### Scenario: A developer substitutes a cited alternative
- **WHEN** a proposed Vector implementation uses conditional non-target probabilities or a proposed Multi implementation uses stationary remasking
- **THEN** validation rejects it as a different objective or bank even if its method has a valid literature source

### Requirement: Stable scalar and paired residual semantics
Scalar readouts SHALL use the legacy FP32 gold logit minus non-gold logsumexp, followed by FP64 residual reductions. Every edge SHALL compare the same ordered masked queries and corpus gold IDs in its two states. C SHALL measure the difference between sparse and dense responses, not suppress sparse response magnitude itself. Pair, query, chain and span weights SHALL follow the declared family-specific reductions.

#### Scenario: A sparse model exactly matches a sensitive teacher
- **WHEN** dense and sparse readouts are identical at every endpoint but change strongly between endpoints
- **THEN** A and C are both zero and the teacher's sensitivity incurs no response penalty

### Requirement: Frozen monotone Multi sampling
The five Multi-family arms SHALL use the original 128-state bank, eight queries per span, two chains per span, stored eight visibility probabilities and original seeds. The same independent eligible-position uniforms SHALL generate every phase of a chain. Queries SHALL be selected upfront and remain masked. Unchanged states and zero-change edges SHALL retain their original multiplicities. Deduplicating computation MUST NOT change sample weights.

#### Scenario: A chain contains equal adjacent inputs
- **WHEN** no eligible position is revealed between two phases
- **THEN** both phase entries remain in the bank and their edge response is zero; the system does not resample or force a reveal

#### Scenario: Sampling no longer meets the product-coupling assumptions
- **WHEN** a bank enforces fixed mask counts, postselects queries, or uses teacher-dependent reveal decisions
- **THEN** it fails the Multi identity check instead of claiming the cross-bias formula applies

### Requirement: Exact Multi controls and graph normalization
Multi-family scores SHALL use the five losses and explicit edge sets in design §0.3. Short SHALL use four disjoint neighboring pairs, Path all seven adjacent pairs, All all 28 unordered pairs, and Multi the equal mean of the three four-edge matchings. The reducer SHALL preserve equal query/chain/span means. No extra division by phase distance, response magnitude or standard deviation is permitted.

#### Scenario: Short misses a phase-varying residual
- **WHEN** a chain's residuals are [1,1,-1,-1,1,1,-1,-1] for every query
- **THEN** its A is 1, C1 is 0, C2 is 4, C4 is 0 and Multi loss is 7/3
- **AND** All uses population phase variance with factor 16/7, not an additional variance loss

#### Scenario: Node exposure is described
- **WHEN** a report compares Path and Multi
- **THEN** it states that Path has different endpoint degrees even though both edge-mean quadratic forms have trace 2

### Requirement: Square structure and interaction interpretation
Square SHALL use node order [00,10,01,11], the four single-group edges, and endpoint/edge means each divided by four. Sampling SHALL follow the fixed seeds and split-specific recipe in design §3. Queries SHALL be fixed upfront; reveal groups SHALL be disjoint, equally sized and exclude visible/query positions. Degenerate quartets SHALL be retained. The system SHALL expose squared residual mode diagnostics from the same outputs and MUST NOT add an interaction or path-consistency penalty.

#### Scenario: Pure group-interaction residual
- **WHEN** residuals in node order are [1,-1,-1,1]
- **THEN** A is 1, C is 4, AC is 5 and squared mixed residual is 16, with the last quantity reported only as a diagnostic

#### Scenario: Equal graph but different contexts
- **WHEN** Square is compared with a theoretical four-node Multi graph
- **THEN** graph equality does not authorize reusing a bank whose middle contexts have different inclusion relationships

### Requirement: Full-vocabulary Vector geometry and numerical identity
Vector SHALL include all vocabulary coordinates, preserve legacy ordered pairs/queries and weight each pair equally. It SHALL use raw emitted logits stored losslessly as FP32 and whole-vocabulary centering plus score arithmetic in FP64, as in design §0.5. It MUST NOT center each chunk independently, substitute hidden-state MSE, apply temperature/probability weights, or replace per-pair means with a global query mean. Cache identity SHALL include readout and reduction versions.

#### Scenario: Common shifts and vocabulary chunks
- **WHEN** different common logit offsets are added at each endpoint and for each model
- **THEN** full and chunked vector A/C are invariant within declared numerical tolerance
- **AND** CPU tests verify the pairwise-margin identity and reject per-chunk centering

#### Scenario: Pair query counts differ
- **WHEN** one pair has more masked queries than another
- **THEN** each pair contributes one equal-weight pair mean rather than receiving weight proportional to its query count

### Requirement: Reproducible diagnostic banks
Square and Vector diagnostic inputs SHALL be generated and frozen from clean spans 8–15 without reading task outcomes. Square SHALL use the separate seed and five-quartet construction in design §3. Vector SHALL generate 80 diagnostic pairs using ten legacy mask probabilities and the explicit mask/reveal seeds in design §4; the existing 40-state diagnostic input is a source of clean spans only. The diagnostic recipe SHALL not be substituted for the byte-identical legacy calibration pairs.

#### Scenario: Diagnostic source has only five states per span
- **WHEN** the Vector diagnostic source manifest contains 40 noisy states
- **THEN** the implementation extracts the eight clean spans and constructs the declared 80 before/after pairs rather than silently reducing the diagnostic sample or duplicating states

### Requirement: Actual final-model and allocation measurements
Square and Vector SHALL measure native Uniform calibration loss and final physical masks' calibration and diagnostic losses according to design §5. Exchange SHALL retain its measured calibration incumbent and compare anchor/final scalar diagnostics as defined in design §6. Marginal costs SHALL retain their sign; larger positive costs SHALL map to less pruning under the unchanged average-rank rule. The exact-quota DP SHALL be described as budget rounding, not minimization of the functional objective. A final loss MUST NOT be reconstructed from a sum of independent probe losses.

#### Scenario: A final allocation differs from independent predictions
- **WHEN** the assembled mask's measured loss differs from the sum or extrapolation of layer probes
- **THEN** the report retains the actual loss and records the discrepancy without replacing measurements by the surrogate

### Requirement: Exchange acceptance and bounded conclusions
Exchange SHALL follow design §6 with removal quantum 53,248d, d=41, fixed anchor offsets and original scalar pair weighting. Positive predicted gain SHALL mean DeltaN*(g_receiver-g_donor) for a donor that removes more weights. The first finite initial evaluation SHALL fix L0 and epsilon before the reproducibility repeat. Candidate selection SHALL use measured full-bank loss, never mini100 or diagnostics. The fixed old-cost shortlist is a declared limited neighborhood, not a guarantee from EvoPress or SPDY.

#### Scenario: Old costs predict the wrong direction
- **WHEN** a shortlisted exchange has positive predicted gain but no measured gain exceeding epsilon
- **THEN** it is not accepted and a capped stop is labeled no improvement among offered candidates, not local or global convergence
