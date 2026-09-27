# Reveal-KL allocation mini100

## Setup

one entire block rowwise65 in otherwise dense model; verified historical dense80-calibrated StandardWanda65 masks; native batch1 full-forward
FP32 KL(dense||pruned), uniform token mean over dense next reveal (K1) or all currently masked; equal16 states/prompt, equal8 prompts
average ranks across32 scores; s=.65-.10*((rank-1)/31-meanrank); block/inputwidth floor±1 exactbudget DP; no tuning
both candidates native sparse-prefix block-wise StandardWanda on original80 corruptionstates; not trajectory-recalibrated weight ranking

## Results

{
  "status": "complete",
  "config_sha256": "51d7a340fa2f9101ad686ca556320be89b307cea0e9618a322ee27b3325aec24",
  "scores": {
    "reveal": 19,
    "all_masked": 16,
    "uniform": 12
  },
  "primary_reveal_vs_all_masked": {
    "reference_wrong_candidate_correct": 6,
    "reference_correct_candidate_wrong": 3,
    "both_correct": 13,
    "both_wrong": 78,
    "net_correct": 3,
    "discordant": 9,
    "exact_mcnemar_p": 0.5078125
  },
  "secondary_vs_uniform": {
    "reveal": {
      "reference_wrong_candidate_correct": 11,
      "reference_correct_candidate_wrong": 4,
      "both_correct": 8,
      "both_wrong": 77,
      "net_correct": 7,
      "discordant": 15,
      "exact_mcnemar_p": 0.11846923828124999,
      "holm_two_vs_uniform": 0.23693847656249997
    },
    "all_masked": {
      "reference_wrong_candidate_correct": 8,
      "reference_correct_candidate_wrong": 4,
      "both_correct": 8,
      "both_wrong": 80,
      "net_correct": 4,
      "discordant": 12,
      "exact_mcnemar_p": 0.3876953125,
      "holm_two_vs_uniform": 0.3876953125
    }
  },
  "score_spearman": 0.8621700879765395,
  "changed_projection_budgets": 182,
  "mean_absolute_block_rate_difference_pp": 1.088709677419354,
  "sources": {
    "/home/tmluser1/sap/experiments/dlm_reveal_kl_mini100/allocation.json": "068243ee09f354f197008326523c7b635fd0346167a21d26b096164078715d96",
    "/home/tmluser1/sap/experiments/dlm_reveal_kl_mini100/collection_receipt.json": "a1cfe521f4f37118ca332848220592883344367835c32810b9ca89dc72d393b6",
    "/home/tmluser1/sap/experiments/dlm_reveal_kl_mini100/reveal/results.json": "f4437f37af83564604f017374cb96b198a86a793bc153d3ee8873fd553721e09",
    "/home/tmluser1/sap/experiments/dlm_reveal_kl_mini100/all_masked/results.json": "7cf2df7342d6f8a060d79cf113649d9345a1c0e3548ff7afd663b1e775fded7d",
    "/home/tmluser1/sap/experiments/dlm_reveal_kl_mini100/reveal/predictions.jsonl": "b7e34c1fa65414e5d061a5c5a7a8a82388381eebcbda5820f4722269ceb15470",
    "/home/tmluser1/sap/experiments/dlm_reveal_kl_mini100/all_masked/predictions.jsonl": "ef1f140569486038ad9d6f336cd4b115d7caf0d14afeaf8ed01b83742582a454"
  },
  "limits": [
    "only8 WikiText-prefix prompts; decoding-domain/context-length mismatch to five-shot GSM8K",
    "K1 reveal statistic may have high variance; block grouping does not guarantee low noise",
    "single-target loss level is a heuristic, not marginal damage; no additive optimality claim"
  ]
}

## Interpretation / Decision

Primary is reveal versus all-masked under matched conditions. This is a reused mini100 development screen, not a new held-out confirmation. No full/PPL/75% automatically launched.
