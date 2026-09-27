# DLM pruning: writing-first workspace

Created 2026-09-27 after the user approved the transition from the current experiment to writing-first research.

## Current scope

Finish the existing A / Multi / Cross / CrossMatched experiment, verify its outputs, and assemble an internal paper v0. No new model experiment or method change is included in this transition.

- [Paper source](paper.template.md): substantive draft text, with explicitly unresolved results.
- [Readable preparation copy](paper-preparation.md): verified historical tables and visible pending full results; this is not paper v0.
- [Claim map and next decisions](claim-map.md): what each comparison can establish.
- [Workflow](workflow.md): evidence, writing, and review rules adapted to this project.
- [Task contract](contract.json): authorized work and boundaries.
- [Reference verification](references.md): checked sources and limits of this initial literature pass.
- `state.json`: machine-readable completion state, written by the continuation process.
- `paper-v0.md`: created only after the full-run verification passes.
- `closeout.json` / `closeout.md`: verified statistics, provenance, historical reproduction, error categories, and costs.

## Commands

Run from this directory:

```bash
bash run.sh check
bash run.sh watch
bash run.sh finalize
```

`check` inspects readiness and verifies the already completed separate100 comparison. It does not publish full-run results. `watch` waits in the dedicated `writing_dlm_closeout` tmux session until the existing controller completes, then calls `finalize` once. `finalize` refuses incomplete or altered inputs. A failed gate leaves a failure record and does not create a paper-v0 file. It never runs a model, changes the experiment, repairs source data, or launches a follow-up experiment.

The existing experiment is `experiments/dlm_crosschain_control50`. Do not call its `validate` command during the run: the implementation audit documented a receipt invalidation problem. Its controller already generates its own report at completion. This workspace only reads those artifacts.

The generated v0 is an internal research draft. Scientific interpretation, the broader novelty review, and final Obsidian synchronization need an agent review after background completion. No automatic submission, publication, or notification is configured.
