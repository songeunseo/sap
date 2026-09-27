# Mechanism review for the next DLM-pruning experiment

**Date:** 2026-09-27  
**Scope:** critical review of three primary sources relevant to early-layer
organization, mask fraction or generalized timestep information, and temporal
oscillation. This memo selects a bounded next step; it does not report a new
model run.

## Decision

Do not make early-layer reallocation or generalized timestep weighting the next
main pruning experiment. The reviewed sources provide useful diagnostic
hypotheses, but none establishes a rule for a fixed, 50% unstructured weight
mask on the project's LLaDA-8B-Base setup. Early-layer allocation and
timestep/mask-aware calibration are also already represented by the project's
reviewed literature and prior branches. A new full run would therefore mix a
known prior with a substantial model and transfer assumption before the
mechanism is isolated.

If one follow-up is approved later, use a **single bounded Dense/A50 ×
NFE interaction screen** with the existing masks: hold the frozen
LLaDA-8B-Base revision, prompts, temperature-0 decoder, and length-256
configuration fixed, and evaluate Dense and the existing A50 mask at NFE 256,
64, and 32. Save committed-token trajectories, confidence, and mask counts;
do not introduce a new allocator or early-versus-late/timestep weighting. A
shared-state reveal control may be included with the same query and the same
number of masked positions, but it controls mask count while visibility
identities and positions can still differ; it does not identify a pure semantic
context effect. Read an accuracy interaction as descriptive evidence: sparse
and dense trajectory occupancy, confidence, and calibration can also explain
it, so a positive interaction is not by itself causal evidence. This is a
proposed screen only; it was not run here.

The recommendation follows the project's evidence boundary: a static mask is a
parameter choice, while independently replayed denoising states are additional
inputs rather than a causal hidden-state transition. Any later allocation
claim must therefore be validated with an exact hard mask on held-out states.

## Project evidence and prior-review boundary

The fresh Obsidian check returned a valid sync response and a valid read of
`Research/DLM-Pruning/Research-State.md`. The current state records a frozen
LLaDA-8B-Base pruning question and a completed cross-chain comparison in which
an incremental C benefit and a natural-connection advantage were not
established. This memo does not convert those non-significant comparisons into
equivalence or universal-failure claims.

The following local reviews were read before assessing novelty:

- `research/literature_dlm_2026-09-22.md`
- `research/dlm_novelty_synthesis_2026-09-18.md`
- `research/dlm_literature_reaudit_2026-09-18.md`
- `research/allocation_novelty_reaudit_2026-09-18.md`
- `research/theory_ac_dynamics_2026-09-22.md`

Those reviews already cover the following adjacent evidence and decisions:

- **Layer Collapse in Diffusion Language Models:** early-is-sparser (EIS) and
  deeper-is-sparser (DIS) are direct DLM allocation precedents. “DLMs should
  prune early layers more” is not an unreviewed novelty claim.
- **Quant-dLLM, Sink-Aware Pruning, and FAIR-Calib:** timestep or mask-aware
  calibration, DLM-specific static compression signals, frontier reliability,
  and post-commit effects already have primary-source precedents. A new scalar
  with a DLM label is not enough by itself.
- **Reveal-KL and trajectory/state reviews:** generic trajectory banks,
  denoising-time weighting, clocks, and temporal reuse were already reviewed.
  A trajectory bank is a measurement bank unless a stateful transition is
  explicitly defined.
- **Induction in Both Directions, Masks Can Be Distracting, Measuring Temporal
  Linguistic Emergence, and Subliminal Clocks:** mask count, progress, and
  state-dependent sensitivity are useful controls or strata. They do not by
  themselves define a static weight-importance objective.

The new papers below are therefore treated as verification and transfer
analysis, not as automatically new mechanisms for this project.

## Primary-source verification

| Source and version | What was checked | Transfer boundary for this project |
|---|---|---|
| [Mechanism Shift During Post-training from Autoregressive to Masked Diffusion LMs, arXiv v5](https://arxiv.org/html/2601.14758v5) ([record](https://arxiv.org/abs/2601.14758)), last revised 2026-08-31 | Four controlled tasks, two paired 7B ARM–MDM families, EAP-IG circuit summaries, and targeted ablations | Task-specific circuit organization in 7B models; no fixed unstructured LLaDA-8B-Base mask study |
| [Induction in Both Directions, arXiv v2](https://arxiv.org/html/2607.15893v2) ([record](https://arxiv.org/abs/2607.15893)), last revised 2026-07-20 | Matched attention-only AR/absorbing-mask DLMs, bidirectional induction, mask-rate probing and patching | A small synthetic circuit and an implicit mask-rate feature; no large-model pruning or general timestep objective |
| [Time Is a Feature, ICLR 2026 abstract](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76931eaba1fcb55b70cde7d0de0161ef-Abstract-Conference.html) and [conference paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/76931eaba1fcb55b70cde7d0de0161ef-Paper-Conference.pdf) | Intermediate-answer oscillation, temporal semantic entropy, weighted voting, and temporal-consistency reinforcement | A decoding and post-training result on LLaDA variants; no static weight-pruning evidence |

### 1. Mechanism Shift During Post-training from Autoregressive to Masked Diffusion LMs

**Direct result.** In Sections 1–3 and Appendix A, Kong, Lee, and Jo compare
Qwen2.5-7B with Dream-Base-7B and LLaMA-2-7B with DiffuLLaMA-7B. The analysis
uses IOI, Greater-Than, Countdown, and Semantic Infilling. For the paired
comparison, the authors use a common-correct pool of 500 examples per model
pair. Their EAP-IG analysis keeps target positions, mask configuration,
denoising schedule, and step aggregation aligned while changing semantic
context; the reported top-1000 edges are an equal-cardinality summary rather
than a complete circuit.

For Countdown, Section 3 and Appendix B report more task-relevant computation
in earlier layers for the MDMs than for the paired ARMs. Targeted output
zeroing of early structures harms Countdown under the tested intervention,
while random and middle/late controls do not show the same drop. This is
evidence for task-specific early-layer organization in the studied 7B model
families. It is not evidence that an early layer should receive a particular
static sparsity rate in LLaDA-8B-Base.

**Limits stated by the source.** The paper studies four structured or synthetic
benchmarks and successful examples from a common-correct subset. The authors
state that open-ended generation and broader task generalization are untested,
that cross-family magnitudes are confounded by backbone, pretraining,
post-training, and diffusion-recipe differences, and that the attribution
pipeline does not exhaust all computation. The targeted ablation is an
intervention on selected heads/MLP dimensions and layers, not permanent
unstructured weight pruning.

**Implication.** This source supports a hypothesis that layer importance can be
task- and state-dependent. It does not support an EIS/DIS rule as a new method
or a universal early-layer law. The model families are Dream and DiffuLLaMA,
not the project's LLaDA-8B-Base checkpoint, so a transfer claim would require
an exact hard-mask comparison on that checkpoint.

### 2. Induction in Both Directions

**Direct result.** In Sections 3–4 and the associated tables/figures, Catruna
and Radoi study matched attention-only AR and absorbing-mask DLMs with two or
three layers and three seeds on a repeated-token induction task. The circuit
uses previous- and next-token context and later induction heads. Forward and
reverse induction are approximately symmetric, and the reported DLM behavior
does not show a one-sided induction advantage over the AR control.

The paper also probes a global mask-rate representation. A layer-0 residual
feature predicts the global mask fraction in the reported small models, and
patching that feature recovers part of the entropy gap between low- and
high-mask conditions. The paper's local-context controls show that preserving
nearby visible context matters strongly, while adding distant masks can still
change entropy.

**Limits stated by the source.** The models are small attention-only
transformers, including folded no-LayerNorm variants; they do not have the
depth, MLPs, and normalization of LLaDA-8B-Base. The task is synthetic and
repeated-token induction, and the masking process is absorbing-mask. The
mask-rate patch demonstrates a functional feature in that toy model; it does
not demonstrate that a parameter group carrying that feature is the right
static pruning target.

**Implication.** This source justifies preserving and measuring mask fraction
as a calibration confound. It does not justify a generalized timestep weight,
an early-layer pruning schedule, or a large-model LLaDA claim. The proposed
same-mask-count diagnostic directly tests the most transferable part of the
result while avoiding a claim that mask-rate encoding is weight importance.

### 3. Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models

**Direct result.** Sections 3–5 and Appendices B and C study LLaDA-8B-Instruct
and LLaDA-1.5 on GSM8K, MATH500, SVAMP, and Countdown. The paper constructs a
fully decoded candidate at each sampling step before remasking and observes
that a correct intermediate answer can later be overwritten. It defines
Temporal Semantic Entropy (TSE) over parsed intermediate answers, then uses
weighted temporal voting at decoding time and negative TSE as a reinforcement
fine-tuning reward. The experiments use semi-autoregressive decoding with
low-confidence remasking, block size 32, and output lengths 128, 256, and 512
with 64, 128, and 256 diffusion steps respectively. The source also reports
ARC-C, Winogrande, and a Sudoku failure analysis in the appendices.

The voting ablation gives more weight to later sampling passes under the
paper's step convention. This is a decoder aggregation rule over complete
intermediate outputs. The reinforcement result changes the model through
post-training; it is not a static weight mask or an allocation rule.

The result is tied to the paper's decoder: semi-autoregressive blocks of 32,
low-confidence remasking, and a complete pre-remask candidate at every step.
It therefore does not directly characterize the project's deployed
temperature-0, length-256, 256-step regime, where the saved decoding trace
uses one commitment per step and copies visible tokens. A provisional argmax
change and a committed-token overwrite are different events; they should not
be merged when transferring the paper's temporal-oscillation language.

**Limits stated by the source.** The method depends on having enough correct or
near-correct intermediate answers. On Sudoku, intermediate correctness stays
below 5%, and temporal voting can reduce accuracy; the authors state that this
limits both voting and the usefulness of the reinforcement signal. Temporal
statistics also discard malformed answer spans and, for the TSE reward, use
only the second half of sampling steps. These choices make TSE a task- and
decoder-dependent signal. LLaDA-8B-Instruct is not the project's
LLaDA-8B-Base model.

**Implication.** The paper supports measuring trajectory stability when studying
decoding, and it warns against turning a useful output-time signal into a
parameter-importance claim. It does not establish that a static mask should
preserve a particular denoising timestep or temporal oscillation pattern. To claim transfer into static pruning, a later screen should hold the decoder
fixed and measure hard-mask effects on held-out states. That evidence is needed
for a mechanism interpretation; it is not a universal prerequisite for running
an explicitly bounded method screen.

## Cross-paper interpretation

The three sources agree on a narrower statement: computation and predictions
can depend on denoising state, visible-context pattern, or sampling history.
They do not agree on, or test, a single static weight objective. Their evidence
has three different units:

| Evidence unit | Verified source result | Safe use in this project | Overreach to avoid |
|---|---|---|---|
| Circuit attribution/intervention | Early task-relevant structures in selected 7B MDM tasks | Stratify or stress-test layer allocation on the exact target model | Calling an attribution subgraph a static sparsity schedule |
| Mask-rate representation | Small DLM encodes global mask fraction and responds to local visibility | Match mask count and report mask fraction as a negative control | Treating a residual probe or patch as proof of weight importance |
| Temporal output trajectory | Intermediate answers may be correct then overwritten | Keep decoder and trajectory metrics separate from parameter pruning | Transferring voting/RFT gains into a static-mask claim |

In particular, “early layer,” “timestep,” and “temporal” refer to different
objects here: a layer within one forward pass, a mask fraction or denoising
state, and a sequence of decoded outputs. Combining them into one proposed
importance score would be an unverified synthesis.

## Why this is not the next main experiment

1. **Model and task mismatch.** The mechanism-shift paper uses Dream/DiffuLLaMA
   paired with AR backbones and four structured tasks. The induction paper uses
   tiny attention-only synthetic models. The temporal-oscillation paper uses
   LLaDA-8B-Instruct/1.5 with a particular semi-autoregressive decoder. None
   evaluates static unstructured pruning on LLaDA-8B-Base.
2. **Prior coverage is already strong.** EIS/DIS, timestep/mask-aware
   compression, frontier-aware calibration, temporal banks, clocks, and
   state-dependent sensitivity are already in the project's source inventory.
   Rediscovering one of these ingredients would not establish novelty.
3. **The present project result does not identify a mechanism.** The current
   state records no established incremental benefit for the added C signal.
   That result should not be reinterpreted as proof that early layers,
   timesteps, or a temporal mechanism caused the observed directions.
4. **The transfer test is missing.** A source-level signal can motivate a
   bounded screen, but a transfer or mechanism claim requires measuring exact
   hard-mask changes on held-out states under the frozen model and decoder. The
   papers reviewed here do not supply that test.

## Reproducibility and non-actions

This review used the current primary versions listed above and the local
literature audits named in the project-boundary section. No GPU/model forward,
mask generation, new experiment, manuscript edit, Obsidian write, or old
artifact mutation was performed for this memo. The only intended write from
this task is this file.

