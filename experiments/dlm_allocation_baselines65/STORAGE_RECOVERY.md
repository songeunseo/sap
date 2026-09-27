# Storage and execution recovery — 2026-09-14

## Findings
- Root reached 20KB available during this investigation (833GB filesystem).
- Alpha log stops mid-character at 57,344 bytes around 00:21:52. No predictions,
  final results or exit code. Disk write failure is suspected, not proven.
  Kernel logs require administrator access. No reboot occurred.
- Accessible du accounted for about112GB, not the full df usage. Permissions and
  inaccessible/deleted-open files prevent a complete root audit.
- Visible major paths: project15GB including .git10GB; other project
  ST-WebAgentBench31GB; /tmp39GB; /var7.9GB. Those unrelated directories,
  .git and system files were not moved/deleted.

## Verified relocation
Destination: /DATA/tmluser1/sap_storage_recovery_20260914/.
- dlm_capacity_predictor/dense_statistics.pt (~1.1GB). Before/after SHA256:
  a9bf47afb8747a59da3d4d864e96a8362d932e6d90cccc772e66b42c96d22cdb
- dlm_capacity_predictor/runtime/ (~3.3GB), including historical pre-fix files.
- All four allocation-baseline result directories and logs.
- Old paths are symlinks. No scientific artifacts discarded. Runtime/results
  trees copied and hashed before original copies were removed. Full manifest:
  /DATA/tmluser1/sap_storage_recovery_20260914/migration.json.
- After migration: root5.2GB free; DATA919GB free. Shared root remains near full.

## Maintenance
Original frozen code/configs unchanged. recovery.py wraps original generation:
- DATA output/logs and per-response checkpoint with Python/NumPy/Torch/CUDA RNG.
- Resume verifies request/provenance/pre-RNG and restores post-RNG. Original
  strict-EM evaluator retained; no change to model/masks/decoding/examples.
- Supervisor saves per-attempt log and return code/signal. Whole-supervisor
  SIGKILL cannot be trapped; PID-aware status flags incomplete runs nonetheless.
- Preflight requires2GiB DATA and512MiB root; future free space is not guaranteed.
- Separate maintenance code hashes frozen; scientific config hashes unchanged.

## Commands
Status: `python3 experiments/dlm_allocation_baselines65/status_safe.py`
Use this instead of the old frozen status.py, which can display stale progress.

When a restart is requested:
`tmux new-session -d -s alpha65_recovery 'bash /home/tmluser1/sap/experiments/dlm_allocation_baselines65/run_recovery.sh 1 alpha'`

This maintenance task does not start a new expensive run. The interrupted90
responses were not saved. First recovery must generate100; masks are reused.
Obsidian tools unavailable this turn; this document is the local recovery record.
