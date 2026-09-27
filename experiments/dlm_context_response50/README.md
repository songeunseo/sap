# Context-response preservation: allocation50 GSM8K mini100

Hypothesis: preserving changes in gold-vs-rest log odds after context reveal improves allocation over preserving endpoint predictions alone.
A+C vs A-only, same paired data/forward compute and frozen rank mapping; Uniform50 freshly evaluated.
Existing 8 WikiText train spans x10 masks,256tokens,80pairs. Before input unchanged. Reveal 5% of sequence capped at one quarter masked tokens; shared query remains masked. Gold reveal, not rollout.
32 layers,48/52 probes in existing Uniform50 sparse background. Signed cost per actual extra removed weight. Rank allocation45–55%; exact3,489,660,928 removed.
All224 Uniform50 sparse-prefix masks must reproduce. Their calibration vectors are frozen for all final candidates; no candidate-specific recalibration. This isolates allocation.
GSM8K first100,5shot,256steps,temperature0,strictEM and seeds follow existing evaluation config. No automatic full/PPL/tuning.
Primary paired AC vs A; secondary vs Uniform Holm2. Reused development mini100, not confirmatory.

Status:
    python3 experiments/dlm_context_response50/status.py
    tail -f experiments/dlm_context_response50/pipeline.log

All runtime in dedicated tmux context_response50_gpu3, GPU3. Source/input hashes frozen before collection. Resume validates artifacts; stopped mini100 evaluations restart that candidate.
