"""Read-only historical comparison for the live AC screen status.

This optional display command is outside the frozen scientific manifest.  It
reads completed receipts and keeps WikiText likelihood separate from GSM8K.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PPL_RESULTS = (
    ("DSA layer", "experiments/dlm_ppl50/dsa/validation/results.json"),
    ("EvoPress", "experiments/dlm_ppl50/evopress/validation/results.json"),
    ("DSA projection", "experiments/dlm_ppl50_projection/dsa_projection/validation/results.json"),
    ("Uniform row-quota", "experiments/dlm_ppl50/uniform/validation/results.json"),
    ("OWL projection", "experiments/dlm_ppl50_projection/owl_projection/validation/results.json"),
    ("OWL layer", "experiments/dlm_ppl50/owl/validation/results.json"),
    ("LSA layer", "experiments/dlm_ppl50/lsa_layer/validation/results.json"),
    ("LSA projection", "experiments/dlm_ppl50/lsa_projection/validation/results.json"),
    ("Uniform layer-global", "experiments/dlm_ppl50_uniform_layer/validation/results.json"),
    ("AlphaPruning", "experiments/dlm_ppl50/alpha/validation/results.json"),
    ("DLP", "experiments/dlm_ppl50/dlp/validation/results.json"),
)


def historical_rows() -> list[tuple[str, float, float]]:
    rows = []
    for name, relative in PPL_RESULTS:
        record = json.loads((REPO / relative).read_text())
        summary = record.get("summary", {})
        nelbo = summary.get("token_nelbo")
        ppl = summary.get("ppl_upper_bound_estimate")
        if (record.get("status") != "complete" or summary.get("blocks") != 551
                or summary.get("tokens") != 268163
                or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (nelbo, ppl))):
            raise ValueError(f"Historical receipt is incomplete or incompatible: {relative}")
        rows.append((name, float(nelbo), float(ppl)))
    return sorted(rows, key=lambda row: row[1])


def historical_uniform_mini() -> tuple[int, int]:
    relative = "experiments/uniform_wanda_sparsity_sweep/evaluation_results.json"
    record = json.loads((REPO / relative).read_text())
    matches = [row for row in record["results"] if row.get("identifier") == "V0_50_uniform"]
    if record.get("status") != "complete" or len(matches) != 1:
        raise ValueError("Historical Uniform mini100 receipt is missing")
    row = matches[0]
    if (row.get("status") != "complete" or row.get("limit") != 100
            or row.get("achieved_global_sparsity") != 0.5):
        raise ValueError("Historical Uniform mini100 setup differs")
    return int(row["correct"]), int(row["limit"])


def main() -> None:
    rows = historical_rows()
    print("\nHistorical 50% WikiText-2 validation | 551 chunks, 268163 tokens")
    print(f'{"Method":22} {"NELBO":>10} {"PPL estimate":>14}')
    print("-" * 49)
    for name, nelbo, ppl in rows:
        print(f"{name:22} {nelbo:10.6f} {ppl:14.6f}")
    correct, total = historical_uniform_mini()
    print(f"\nHistorical cached Uniform-Wanda mini100: {correct}/{total}")
    print("Different mask construction from the live native row-wise Uniform 54/100.")
    print("WikiText NELBO/PPL and GSM8K correct counts are different metrics; do not rank them together.")


if __name__ == "__main__":
    main()
