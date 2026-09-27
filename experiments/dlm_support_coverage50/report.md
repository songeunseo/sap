# Support coverage50 mini experiment

{
  "status": "complete",
  "config_sha256": "73f728952595cc6d52c9a0f690dcc2f77500dbac5d94a32b83b8ede999603f77",
  "scores": {
    "uniform": 54,
    "pooled": 50,
    "coverage": 54
  },
  "primary_coverage_vs_pooled": {
    "reference_wrong_candidate_correct": 8,
    "reference_correct_candidate_wrong": 4,
    "both_correct": 46,
    "both_wrong": 42,
    "net_correct": 4,
    "discordant": 12,
    "exact_mcnemar_p": 0.3876953125
  },
  "secondary_vs_uniform": {
    "pooled": {
      "reference_wrong_candidate_correct": 4,
      "reference_correct_candidate_wrong": 8,
      "both_correct": 46,
      "both_wrong": 42,
      "net_correct": -4,
      "discordant": 12,
      "exact_mcnemar_p": 0.3876953125,
      "holm_two_vs_uniform": 0.775390625
    },
    "coverage": {
      "reference_wrong_candidate_correct": 5,
      "reference_correct_candidate_wrong": 5,
      "both_correct": 49,
      "both_wrong": 41,
      "net_correct": 0,
      "discordant": 10,
      "exact_mcnemar_p": 1.0,
      "holm_two_vs_uniform": 1
    }
  },
  "mini_nelbo": {
    "uniform": {
      "nelbo": 2.3923108160929445,
      "ppl_bound": 10.938742181258208
    },
    "pooled": {
      "nelbo": 2.392656926280111,
      "ppl_bound": 10.942528846625857
    },
    "coverage": {
      "nelbo": 2.391328859939051,
      "ppl_bound": 10.928006088106518
    }
  },
  "paired_mini_nelbo": {
    "coverage_minus_pooled": {
      "delta_nelbo": -0.0013280663410597526,
      "article_bootstrap95": [
        -0.0022211758027886217,
        -0.00036761595196743745
      ],
      "improved_articles": 13,
      "articles": 16,
      "note": "unadjusted descriptive mini development interval, includes fixed MC noise"
    },
    "pooled_minus_uniform": {
      "delta_nelbo": 0.0003461101871664596,
      "article_bootstrap95": [
        -0.002338705863472512,
        0.0031539211111073925
      ],
      "improved_articles": 8,
      "articles": 16,
      "note": "unadjusted descriptive mini development interval, includes fixed MC noise"
    },
    "coverage_minus_uniform": {
      "delta_nelbo": -0.000981956153893293,
      "article_bootstrap95": [
        -0.0032807855324744396,
        0.0013236964137220697
      ],
      "improved_articles": 10,
      "articles": 16,
      "note": "unadjusted descriptive mini development interval, includes fixed MC noise"
    }
  },
  "changed_projection_counts": 201,
  "limits": [
    "channel support is only proxy for unstructured weight pruning",
    "diagonal energy ignores cancellation/propagation",
    "worst-state requirement and .90 fixed design choice",
    "8 calibration spans only; existing mini is development",
    "rank allocation is heuristic; no DLM-specificity or minimum weight-capacity proof"
  ]
}
