# Multiscale A+C at exact 50% sparsity

Implements the [2026-09-22 experiment plan](../../research/ac_multiscale_experiment_plan_2026-09-22.md).
Preparation and tests run on CPU. A GPU experiment starts only with an explicit `launch` command.
There is no GPU polling daemon, queued launch, full GSM8K evaluation, or automatic confirmation phase.

## Commands

Run from `/home/tmluser1/sap`. The wrapper uses the same installed Python dependencies as the historical A+C experiment, hides CUDA for CPU commands, and uses local Hugging Face caches in offline mode.

```bash
# CPU preparation: banks, fixed prompts/token hashes, existing mask/result hashes.
bash experiments/dlm_multiscale_ac50/run.sh prepare

# CPU validation and status; neither command queries GPU availability.
bash experiments/dlm_multiscale_ac50/run.sh validate
bash experiments/dlm_multiscale_ac50/run.sh status

# Print a future launch command only. Does not query GPUs or start tmux.
bash experiments/dlm_multiscale_ac50/run.sh launch --gpus 0,3 --dry-run
```

**When GPUs are available and execution is requested:**

```bash
# One GPU: --gpus 3. Two GPUs: --gpus 0,3.
bash experiments/dlm_multiscale_ac50/run.sh launch --gpus 0,3

# Monitor from any shell. Closing this shell does not stop the experiment.
watch -n 5 'bash /home/tmluser1/sap/experiments/dlm_multiscale_ac50/run.sh status'
tmux attach -t dlm_multiscale_ac50_development

# Only after reviewing development results and meeting the frozen gate:
bash experiments/dlm_multiscale_ac50/run.sh launch --gpus 0,3 --phase confirmation
```

The launcher checks for existing experiment sessions, an occupied pipeline lock, and active compute processes/memory on the requested GPUs. It does not interrupt another user's process. Workers require tmux and exactly one explicitly visible GPU. A failed worker stops sibling workers; already completed node/document checkpoints remain usable. To resume after fixing an external failure, use the same `launch` command with the same output directory and unchanged inputs.

Alternative output directory (put `--root` before the action):

```bash
bash experiments/dlm_multiscale_ac50/run.sh --root /home/tmluser1/sap/experiments/dlm_multiscale_ac50/another_output prepare
```

Code/input hashes are frozen at preparation. If the implementation or dependencies change, use a new output directory; do not mix old readouts with a new config. Preparation refuses to alter a started experiment. Every command that combines or consumes experimental artifacts validates its inputs.

## What runs

1. Dense reference on two 128-state banks; two repeat checks; reproduce all 224 historical Uniform50 Wanda masks and the complete sparse model hash.
2. Sixty-four 48/52% block probes on the calibration bank, with frozen within-row Wanda ranking. Two GPUs split blocks 0–15 / 16–31. Dense backups are held only for each worker's block shard. The original weights of the probed block are restored to the Uniform50 background after both conditions.
3. Compute A/Short/Path/All/Multi costs from the same scalar readouts; signed rank mapping and historical exact-budget DP. CPU span resampling reports ideal-rate stability before integer rounding.
4. Build each distinct final allocation, evaluate both response banks, and evaluate the fixed development mini-100. Equal row-count allocations share a candidate evaluation; equality of final physical masks with historical models also permits verified cached generations. Historical A+C gets fresh response diagnostics while reusing its verified mini100 generations.
5. Write development/diagnostic reports and stop. A separate explicit confirmation launch evaluates Multi, the selected strongest simple control, and historical A+C on the fixed additional 100 questions, deduplicating physical models.

All five new arms share the node bank, total response coefficient, pruning backend, model revision, and evaluation protocol. No λ/grid/sparsity/solver sweep is included.

## Evaluation identity and restart behavior

The CPU preparation constructs all GSM8K **prompts** in original dataset order so the historical few-shot random stream is preserved, then stores only the selected 200 documents. It does not instantiate a language model or generate any answer. Original document IDs, prompt/doc/target hashes, tokenizer input hashes, task YAML/version, and historical protocol hash are checked. All 300 cached Uniform/A/AC mini predictions are regraded with the official strict-match task on CPU.

Generation calls the repository's native `LLaDAEvalHarness.generate_until` for one selected document at a time: 5-shot, 256 generation tokens, 256 denoising steps, temperature 0, low-confidence remasking, no CFG, batch one. It retains the native stopping-token and special-token cleanup behavior. Scoring uses the installed task filters and `process_results`, not a replacement answer parser.

Readouts are checkpointed after each state. Generated answers are checkpointed after each document using atomic JSON writes; a crash cannot truncate a previously committed checkpoint. Fingerprints bind config, mask/model, bank or request set, split, and confirmation selection. Resume skips only verified completed items. At temperature zero the native generation path does not consume sampling RNG, so skipping earlier requests preserves the decoding procedure.

The five development models generate at most 500 answers; optional confirmation generates at most 300. Each phase has 100 questions. Final reports keep development and confirmation separate. Historical use of the confirmation questions elsewhere in the project is not ruled out.

## Artifacts and progress

The default output is `experiments/dlm_multiscale_ac50/output/`:

- `config.json`, `bank_calibration.json`, `bank_diagnostic.json`, `requests.json`: frozen inputs.
- `readouts/<candidate>/<bank>.json`: resumable scalar gold-vs-rest log odds.
- `probes/blockXX.json`, `allocation.json`, `allocation_stability.json`: probe receipts and exact allocation.
- `candidates/<method>/mask_manifest.json`, `model_identity.json`: masks and model identity.
- `diagnostics/`, `diagnostics_report.md`: A, C1/C2/C4, Path/All, phase error, query CE, response sign changes. Query CE is not NELBO/PPL. Sign changes exclude |dense response|≤1e−6 and report the excluded fraction.
- `gsm8k/<phase>/<method>/examples/`: atomic document checkpoints; `results.json` is written after 100 verified documents.
- `development_report.md`, `confirmation_report.md`: paired results and decisions.
- `progress/`, `logs/`, `worker_environment/`: worker stage, counters, ETA, PID/device, logs and actual CUDA/PyTorch environment.
- `experiment_note.md`, `experiment_state.json`: local running/completed/failed record, created only when execution starts.

Project memory: Obsidian MCP was not exposed while preparing this code. Before an assistant starts the experiment, follow AGENTS.md to create the running `Experiments/` note if MCP is callable. The launch script also writes a local note with `obsidian_sync: pending`; this is explicitly not a claim that the vault was updated.

## CPU validation

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTHONPATH=/DATA/tmluser1/sap-cgq-torch28:/DATA/tmluser1/sap-cgq-python:/home/tmluser1/sap \
/usr/bin/python3 -m unittest experiments.dlm_multiscale_ac50.test_cpu -v
```

Tests cover the product monotone coupling, fixed queries, unchanged pairs, multiscale versus short-only blind spots, all-pair variance identity, negative marginal costs/ties, exact 50% DP allocation from 64 simulated probes, physical allocation deduplication, interrupted/resumed document and scalar checkpoints, stale/corrupt identity rejection, and a tiny CPU model exercising the real readout loop. The dry-run test forbids subprocess launches and GPU queries.

These checks do not substitute for the first real GPU smoke check. They intentionally do not load the 8B model on either GPU or CPU.
