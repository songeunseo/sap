# OWL storage recovery — 2026-09-13

## Observed failure
223/224 masks were saved. Saving block_31.up_proj failed in torch.save with
`PytorchStreamWriter failed writing file data.pkl: file write failed`, followed
by `unexpected pos 64 vs 0`. No GSM8K examples had been generated.
Root filesystem was 100% rounded usage (831/833GB, 1.3GB available at inspection).
Inodes were available; quota command unavailable. Storage pressure is the leading
diagnosis, not a proven ENOSPC errno. /DATA had 933GB available.

## Recovery
Moved only this experiment's masks (1.4GB) to
`/DATA/tmluser1/dlm_owl65_artifacts/masks`; original masks path is now a symlink.
All 224 file hashes matched before/after move (223 complete masks + partial tmp).
Preserved the partial last file separately as
`/DATA/tmluser1/dlm_owl65_artifacts/block_31.up_proj.failed.pt.tmp`.
No historical experiment or model data deleted. Code, calibration, allocation,
ranking, budget, protocol and config unchanged.
Frozen config SHA256:
`b410bac32a6bb4063a4b52c4554fb4ce51f46273096773ea8c26c71156a78912`.

Resume uses the existing code's mask hash checks and regenerates only the missing
final mask file; scoring is recomputed deterministically. Original failure log is
preserved, resumed output is appended to logs/run.log. Obsidian connection failed with
Session not found; this file provides a local recovery record pending sync.

## Resume verification
All frozen source hashes passed. Restarted tmux `owl65` on idle GPU0.
At 22:03:43 KST, masks reached 224/224 and the previously failing final mask was
successfully saved (9.8MB). Remaining build validation/evaluation continues.
