# Alpha mini100 recovery restart

- Status: running, launched 2026-09-14 01:37:12 KST at explicit user request.
- tmux: alpha65_recovery; GPU1.
- Objective: finish the interrupted AlphaPruning mini100 evaluation.
- Hypothesis: unchanged from the original baseline experiment; no new allocation.
- Planned/actual setup: reuse frozen Alpha allocation/mask manifest and exact
  historical100 GSM8K examples, generation settings, seeds and strict EM.
- Storage-only maintenance: recovery.py, per-request response/RNG checkpoints,
  DATA logs/results, process exit receipt. Original scientific config unchanged.
- No previous generation checkpoints exist: evaluate100 from the start.
- No results at launch. No automatic full1319 or retuning.
- Related: STORAGE_RECOVERY.md, original config.json and alpha/build_receipt.json.
- Obsidian connector unavailable at launch; local note used, not synced.
