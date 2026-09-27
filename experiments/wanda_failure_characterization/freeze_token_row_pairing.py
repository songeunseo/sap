import hashlib
import json
from pathlib import Path

import torch

from experiments.wanda_failure_characterization.propagation_core import class_preserving_permutation


ROOT = Path(__file__).parent
EPS = 1e-12


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main():
    held = json.loads((ROOT / "heldout_state_manifest.json").read_text())
    delta_art = torch.load(ROOT / "transplant_deltas.pt", map_location="cpu", weights_only=True)
    previous = json.loads((ROOT / "token_feature_factorization_manifest.json").read_text())
    delta = delta_art["delta31"].float()
    row_norm = delta.norm(dim=-1)
    thresholds = (1e-12, 1e-10, 1e-8, 1e-6)
    zero_gate = {
        "status": "pass" if not bool((row_norm <= 1e-6).any()) else "stop",
        "epsilon": EPS,
        "minimum": row_norm.min().item(),
        "p01": torch.quantile(row_norm.flatten(), .01).item(),
        "median": row_norm.median().item(),
        "total_rows": row_norm.numel(),
        "below": {str(x): int((row_norm < x).sum()) for x in thresholds},
    }
    atomic_json(ROOT / "token_row_pairing_zero_gate.json", zero_gate)
    if zero_gate["status"] != "pass":
        raise RuntimeError("near-zero row gate failed")

    class_permutations, class_source_shifts, class_hashes = [], [], []
    for state in held["states"]:
        mask = torch.tensor(state["mask"][0], dtype=torch.bool)
        seen, permutations, source_shifts = {}, [], []
        for shift in range(1, 256):
            permutation = class_preserving_permutation(mask, shift).cpu()
            digest = tensor_sha(permutation)
            if digest not in seen:
                seen[digest] = len(permutations)
                permutations.append(permutation)
                source_shifts.append([shift])
            else:
                source_shifts[seen[digest]].append(shift)
        stacked = torch.stack(permutations)
        class_permutations.append(stacked)
        class_source_shifts.append(source_shifts)
        class_hashes.append(tensor_sha(stacked))
    permutation_path = ROOT / "token_row_pairing_permutations.pt"
    torch.save({
        "unrestricted_offsets": torch.arange(1, 256),
        "class_permutations": class_permutations,
        "class_source_shifts": class_source_shifts,
        "class_hashes": class_hashes,
    }, permutation_path)
    manifest = {
        "status": "frozen_before_pairing_evaluation",
        "unrestricted_offsets": list(range(1, 256)),
        "unrestricted_condition_count": 40 * 255,
        "class_unique_counts": [int(value.shape[0]) for value in class_permutations],
        "class_condition_count": sum(value.shape[0] for value in class_permutations),
        "class_hashes": class_hashes,
        "permutation_artifact_sha256": file_sha(permutation_path),
        "delta31_sha256": delta_art["hashes"][1],
        "heldout_state_sha256": held["historical_state_sha256"],
        "prior_tau_result_sha256": previous["result_sha256"],
        "epsilon": EPS,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    atomic_json(ROOT / "token_row_pairing_manifest.json", manifest)
    print(json.dumps({"manifest_sha256": manifest["manifest_sha256"], "class_unique_counts": manifest["class_unique_counts"], "class_condition_count": manifest["class_condition_count"]}))


if __name__ == "__main__":
    main()
