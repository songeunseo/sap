#!/usr/bin/env python3
"""Compact progress for the two parallel mini-100 evaluations."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("experiments/dlm_role_layer_budget65")


def progress(method: str) -> str:
    receipt = ROOT / "gsm8k" / f"{method}_100_predictions.receipt.json"
    if receipt.exists():
        row = json.loads(receipt.read_text())
        return f"complete {row.get('correct')}/100 correct"
    log = ROOT / "logs" / f"{method}.log"
    if not log.exists():
        return "not started"
    text = log.read_text(errors="replace")
    matches = re.findall(r"(?:Generating[^\r\n]*?)(\d{1,3})/100", text)
    if not matches:
        matches = re.findall(r"(\d{1,3})/100", text)
    observed = max(map(int, matches), default=0)
    if "Traceback" in text or '"event": "evaluation_complete"' not in text and "RuntimeError:" in text:
        return f"failed after observed {observed}/100"
    return f"running/preparing: observed {observed}/100"


for name in ("layer_aggregate", "layer_role"):
    print(f"{name}: {progress(name)}")
result = ROOT / "mini100_results.json"
print("final: complete" if result.exists() else "final: pending")
