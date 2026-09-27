"""Evaluate only the frozen Uniform-Wanda-75 mask on full GSM8K.

This follow-up intentionally writes a separate artifact so that the completed
Role-75 result is never overwritten.
"""

import json
from pathlib import Path

from experiments.dlm_dual_role_allocation.io import atomic_write_json
from experiments.projection_capacity_followup_65.core import paired_binary_comparison

from .run import ROOT, _evaluate_methods, _read_jsonl, event


def main() -> None:
    suite = "target75"
    evaluated = _evaluate_methods(suite, ["uniform"], 1319)
    uniform = evaluated["outcomes"]["uniform"]

    role_result_path = ROOT / "full_target75.json"
    role_result = json.loads(role_result_path.read_text())
    role_path = Path(role_result["methods"]["role"]["predictions"])
    role_rows = _read_jsonl(role_path)

    paired = paired_binary_comparison(
        [row["correct"] for row in uniform["rows"]],
        [row["correct"] for row in role_rows],
    )
    result = {
        "status": "complete",
        "suite": suite,
        "method": "uniform",
        "protocol_hash": evaluated["protocol_hash"],
        "uniform": {key: value for key, value in uniform.items() if key != "rows"},
        "role_reference": role_result["methods"]["role"],
        "paired_role_vs_uniform": paired,
    }
    output = ROOT / "full_target75_uniform.json"
    atomic_write_json(output, result)
    event(
        "uniform75_full_complete",
        uniform_correct=uniform["correct"],
        role_correct=role_result["methods"]["role"]["correct"],
        output=str(output),
    )


if __name__ == "__main__":
    main()
