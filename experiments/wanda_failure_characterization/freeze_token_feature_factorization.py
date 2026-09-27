import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).parent
EPS = 1e-12


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    held = json.loads((ROOT / "heldout_state_manifest.json").read_text())
    artifact = torch.load(ROOT / "transplant_deltas.pt", map_location="cpu", weights_only=True)
    delta = artifact["delta31"].float()
    norms = delta.norm(dim=-1)
    thresholds = (1e-12, 1e-10, 1e-8, 1e-6)
    timestep_index = torch.arange(5).repeat_interleave(8)
    zero_gate = {
        "status": "pass" if not bool((norms <= 1e-6).any()) else "stop",
        "fixed_epsilon": EPS,
        "total_rows": norms.numel(),
        "minimum": norms.min().item(),
        "median": norms.median().item(),
        "p01": torch.quantile(norms.flatten(), 0.01).item(),
        "below_threshold": {str(x): int((norms < x).sum()) for x in thresholds},
        "by_timestep": [],
    }
    for timestep in range(5):
        values = norms[timestep_index == timestep].flatten()
        zero_gate["by_timestep"].append({
            "timestep_index": timestep,
            "timestep": held["states"][timestep * 8]["timestep"],
            "minimum": values.min().item(),
            "median": values.median().item(),
            "p01": torch.quantile(values, 0.01).item(),
            "below_threshold": {str(x): int((values < x).sum()) for x in thresholds},
        })
    atomic_json(ROOT / "token_feature_zero_row_gate.json", zero_gate)
    if zero_gate["status"] != "pass":
        raise RuntimeError("zero-row sanity gate failed")

    pairs = []
    states = held["states"]
    for receiver in range(40):
        timestep = receiver // 8
        receiver_sequence = receiver % 8
        for donor_sequence in range(8):
            if donor_sequence == receiver_sequence:
                continue
            donor = timestep * 8 + donor_sequence
            pairs.append({
                "pair_index": len(pairs),
                "receiver_index": receiver,
                "donor_index": donor,
                "receiver_sequence": receiver_sequence,
                "donor_sequence": donor_sequence,
                "timestep_index": timestep,
                "timestep": states[receiver]["timestep"],
                "receiver_p_mask": states[receiver]["p_mask"],
                "donor_p_mask": states[donor]["p_mask"],
                "receiver_masked_count": sum(states[receiver]["mask"][0]),
                "donor_masked_count": sum(states[donor]["mask"][0]),
            })
    core = {
        "status": "frozen_before_factorial_evaluation",
        "pairs": pairs,
        "pair_count": len(pairs),
        "delta31_sha256": artifact["hashes"][1],
        "heldout_state_sha256": held["historical_state_sha256"],
    }
    encoded = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    core["mapping_sha256"] = hashlib.sha256(encoded).hexdigest()
    atomic_json(ROOT / "token_feature_donor_manifest.json", core)
    print(core["mapping_sha256"])


if __name__ == "__main__":
    main()
