# Spec Delta

## Purpose

Runs bounded pruning experiments concurrently on explicitly selected GPUs, with durable execution, verifiable reuse of historical artifacts, safe interruption, and progress visible from ordinary terminal commands.

## ADDED Requirements

### Requirement: CPU preparation and explicit launch
Prepare, validate, status, report, and launch dry-run SHALL run without CUDA initialization, model loading, GPU polling, or worker creation. Real launch SHALL require an explicit nonempty GPU list, check running sessions and compute occupancy, and refuse occupied devices without killing their processes. GPU selection SHALL be execution metadata, not a scientific identity that invalidates otherwise compatible checkpoints.

#### Scenario: Dry-run on a CPU host
- **WHEN** a user requests a launch dry-run
- **THEN** the command prints the planned job graph, reuse counts, compute budget and launch command without querying devices or starting tmux

### Requirement: Durable bounded parallel execution
Every long-running model job SHALL run inside a dedicated tmux session. The scheduler SHALL run no more than one model worker on each selected GPU, support any positive number of supplied devices, and dispatch dependency-ready jobs independently across experiment families. It SHALL avoid a cross-family barrier that prevents completed Multi/Short masks from evaluating while other banks collect probes.

#### Scenario: Evaluation overlaps new calibration
- **WHEN** Short's verified mask is ready and a Square probe shard is ready on a second selected GPU
- **THEN** both jobs can execute concurrently without sharing a CUDA device or writing the same artifact

### Requirement: Historical experiment compatibility
The system SHALL preserve frozen legacy source/config files and completed artifacts. Imported probes, masks and predictions MUST retain original fingerprints and provenance; they MUST NOT be rehashed as though produced by new code. The scheduler SHALL exclude concurrent owners of both its own output root and any legacy root it resumes.

#### Scenario: Resume the interrupted multiscale run
- **WHEN** all 64 probes, five allocations and 56 Short/Multi document receipts validate
- **THEN** those computations are reused and only the remaining documents and genuinely missing diagnostic/candidate work are scheduled
- **AND** an active legacy pipeline prevents a second owner from launching

### Requirement: Atomic checkpoint and dependency validation
The system SHALL commit readouts, complete vector-pair metrics, probe receipts, exchange evaluations, accepted incumbents and generated documents atomically. Each artifact SHALL identify scientific config, relevant source files, input/teacher bank, and physical mask/model. Resume SHALL skip only complete compatible items and refuse corrupt or stale artifacts. Identical physical models SHALL be deduplicated only under identical requests and evaluation protocol.

#### Scenario: Worker dies during a write
- **WHEN** a process exits before a checkpoint is atomically committed
- **THEN** resume recomputes the uncommitted unit and preserves all previously validated units
- **AND** a changed bank or source hash is not silently accepted

### Requirement: Safe stop and actionable progress
The system SHALL expose status, stop, and resume behavior with per-arm stage, completed/total work, PID/GPU, elapsed time, measured ETA or unavailable, log path, and terminal failure reason. Stop SHALL terminate only owned process groups and descendants, leave checkpoints intact, and mark interrupted rather than complete. Worker failure SHALL stop owned siblings and retain evidence for a subsequent resume.

#### Scenario: User stops the screen
- **WHEN** the user invokes its stop command
- **THEN** owned workers and descendants exit, unrelated jobs remain untouched, and status reports interrupted with reusable progress

### Requirement: Research lifecycle record
Before the first model job of a real launch, the workflow SHALL record objective, hypothesis, planned setup, time, and related evidence in an Obsidian Experiments note, mark running, and verify the write when MCP is available. Local records SHALL mirror actual setup, progress, results, interpretation, decision and next experiment. MCP absence SHALL be recorded precisely with a pending local record; a missing sync log alone MUST NOT be called disconnected. Planned preparation MUST NOT be recorded as a running experiment.

#### Scenario: Recording a planning-only operation
- **WHEN** only prepare, validation, or dry-run is requested
- **THEN** no note claims that model evaluation has started

### Requirement: Scientific manifest preflight
Before any model work the prepared manifest SHALL expose objective versions, exact sampling and reduction recipes, source links, coefficient/default provenance and the cost accounting in design §8. New Square/Vector banks SHALL be frozen independently of task outcomes. Imported bank bytes and original scientific settings SHALL not be changed to align with a newer theoretical draft. Diagnostic teacher sharing SHALL require matching inputs/readout identity and a single producer. Disk preflight SHALL reserve both Vector teacher banks using actual query counts.

#### Scenario: Preparation reveals unexpected work
- **WHEN** preparation calculates teacher, Uniform reference, probe, final diagnostic and search jobs
- **THEN** the dry-run prints each category separately, including the 32-state smoke cap, without estimating all runtime from mini100 generation counts

### Requirement: Honor the authorized execution scope
Planning edits SHALL not execute models. CPU preparation SHALL not imply a GPU launch. A later user request that explicitly includes both implementation and execution SHALL permit launch after required checks within that same request; the workflow MUST NOT require a redundant second permission solely because a CPU handoff has been reached. A request restricted to implementation SHALL end with the prepared handoff.

#### Scenario: Implementation and launch are requested together
- **WHEN** the user has requested both implementation and execution on specified GPUs and all launch checks pass
- **THEN** the workflow proceeds to the requested tmux launch without an additional artificial approval stage
