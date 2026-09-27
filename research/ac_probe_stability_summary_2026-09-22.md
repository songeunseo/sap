# Supplementary frozen A/AC probe stability audit

The CPU-only audit `research/ac_probe_stability_2026-09-22.json` verified 33 receipt hashes and exactly reproduced the frozen A/AC costs and ideal rates. A-versus-A+C cost Spearman is 0.942815, C-versus-A cost Spearman is 0.153592, and 27/32 ideal rates change (mean absolute shift 0.7863 pp; maximum 2.2581 pp). C accounts for only 6.55% of the Uniform-to-A+C total surrogate reduction. Leave-one-span-out rank stability is descriptive only: the resulting masks were not evaluated, so this is not held-out validation.
