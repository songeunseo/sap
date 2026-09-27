# Master decisions: completed full comparison

## Role application

The user explicitly requests three roles using master.md, experiment.md, and writing.md. Master read the original master.md in full and issued separate bounded tasks requiring each specialist to read its own original document. Runtime accepted gpt-5.6-luna/xhigh for both; no fallback model was used. The source files' Depth-AR settings and sprint deadlines are not this project's settings.

## Evidence and interpretation

Master independently reproduced scores, paired gains/losses, exact p-values, Holm correction, and fixed-seed intervals for all three predefined sample groups. See numeric-verification.json for exact values and source hashes. Original v0 inputs remain frozen.

The primary Multi−A estimate is negative and its interval includes zero; neither superiority nor inferiority is established. Neither natural-versus-cross contrast establishes an advantage of the natural connections. Nonsignificance does not establish equivalence. These results constrain the tested bank/readout/allocator/mask family under this protocol; they do not disprove response preservation generally.

## Decisions

- Revise the internal manuscript to report an unestablished incremental benefit, while retaining the algebraic motivation as an explicitly constructed illustration.
- Lead with the measured primary paired-effect figure. Keep all fixed contrasts, samples, adverse diagnostics, and cost qualifications visible.
- Do not promote Multi to an empirically validated improvement or claim natural pairing is necessary.
- Retain primary1119 and excluded200 definitions; do not tune a replacement on these observed primary answers.
- Preserve auto-generated v0 and its audit inputs. Publish the local revision as paper-v1.md and claim-map-v1.md with latest pointers in README/state.
- No additional model experiment is launched. A future experiment requires a separately specified comparison and scope.

## Acceptance gate

Pending: experiment's fresh end-to-end CPU verification and measured figure; writing's direct-source revision and independent numeric check; master's artifact/claim/figure review and Obsidian readback.
