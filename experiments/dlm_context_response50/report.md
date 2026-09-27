# Context response50 mini100

{
  "status": "complete",
  "config_sha256": "5295194cc620a854c20283de0edeca4dabb9b29da920953ddd87ff950c71f38b",
  "scores": {
    "uniform": 54,
    "A": 55,
    "AC": 61
  },
  "primary_AC_vs_A": {
    "reference_wrong_candidate_correct": 9,
    "reference_correct_candidate_wrong": 3,
    "both_correct": 52,
    "both_wrong": 36,
    "net_correct": 6,
    "discordant": 12,
    "exact_mcnemar_p": 0.14599609375
  },
  "secondary_vs_uniform": {
    "A": {
      "reference_wrong_candidate_correct": 8,
      "reference_correct_candidate_wrong": 7,
      "both_correct": 47,
      "both_wrong": 38,
      "net_correct": 1,
      "discordant": 15,
      "exact_mcnemar_p": 1.0,
      "holm_two_vs_uniform": 1
    },
    "AC": {
      "reference_wrong_candidate_correct": 13,
      "reference_correct_candidate_wrong": 6,
      "both_correct": 48,
      "both_wrong": 33,
      "net_correct": 7,
      "discordant": 19,
      "exact_mcnemar_p": 0.1670684814453125,
      "holm_two_vs_uniform": 0.334136962890625
    }
  },
  "limits": [
    "paired gold reveal is not generated context",
    "scalar gold-vs-rest omits wrong-vs-wrong ranking",
    "state response is surrogate, not NELBO",
    "finite marginal may not compose jointly",
    "no fresh test/generalization claim"
  ]
}
