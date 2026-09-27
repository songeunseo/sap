import hashlib
import json
import platform
from pathlib import Path

import torch

from experiments.dlm_loss_aggregation.run import _load_model, load_config, validate_config
from experiments.wanda_failure_characterization.propagation_core import factorize_token_feature
from experiments.wanda_failure_characterization.run_failure_map import model_sha, state_tensors
from experiments.wanda_failure_characterization.run_state_swap import l31_sham_hidden
from experiments.wanda_failure_characterization.run_token_feature_factorization import evaluate


ROOT = Path(__file__).parent
SHARD_DIR = ROOT / "token_row_pairing_shards"
EPS = 1e-12
KEYS = ("kl", "loss_delta", "top1_agreement", "confidence_mae", "achieved_relative_l2", "norm_deviation", "final_hidden_relative_energy", "masked_logit_relative_energy", "logit_rms")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def patterns(allocation, direction, permutation):
    shifted_a, shifted_u = allocation[permutation], direction[permutation]
    return (allocation[:, None] * direction, allocation[:, None] * shifted_u, shifted_a[:, None] * direction, shifted_a[:, None] * shifted_u)


def collect_family(model, h31, allocation, direction, permutations, tau, clean, mask, p_mask):
    result = {key: torch.empty(len(permutations), 4) for key in KEYS}
    pre_norm_spread = torch.empty(len(permutations))
    for index, permutation in enumerate(permutations):
        cells = patterns(allocation, direction, permutation.to(allocation.device))
        norms = torch.stack([cell.float().norm() for cell in cells])
        pre_norm_spread[index] = (norms.max() - norms.min()).cpu()
        measured, _, _ = evaluate(model, h31, cells, tau, clean, mask, p_mask)
        for key in KEYS:
            result[key][index] = measured[key]
    return result, pre_norm_spread


@torch.inference_mode()
def main():
    frozen = json.loads((ROOT / "token_row_pairing_manifest.json").read_text())
    held = json.loads((ROOT / "heldout_state_manifest.json").read_text())
    delta_art = torch.load(ROOT / "transplant_deltas.pt", map_location="cpu", weights_only=True)
    prior = torch.load(ROOT / "perturbation_transplant.pt", map_location="cpu", weights_only=True)
    permutations = torch.load(ROOT / "token_row_pairing_permutations.pt", map_location="cpu", weights_only=True)
    transplant_manifest = json.loads((ROOT / "perturbation_transplant_manifest.json").read_text())
    if file_sha(ROOT / "token_row_pairing_permutations.pt") != frozen["permutation_artifact_sha256"]:
        raise RuntimeError("permutation artifact changed")
    if delta_art["hashes"][1] != frozen["delta31_sha256"] or file_sha(ROOT / "transplant_deltas.pt") != transplant_manifest["delta_artifact_sha256"]:
        raise RuntimeError("delta31 artifact changed")

    config = load_config("experiments/dlm_loss_aggregation/config.yaml")
    validate_config(config)
    model, _ = _load_model(config)
    before = model_sha(model)
    device = model.model.transformer.wte.weight.device
    delta = delta_art["delta31"].float()
    factorized = [factorize_token_feature(value, EPS) for value in delta]
    state_cache, nn_gate = [], {key: torch.empty(40) for key in ("kl_abs", "loss_abs", "final_energy_abs", "logit_energy_abs")}

    # All-state native gate precedes every permutation result.
    for state_index, state in enumerate(held["states"]):
        noisy, clean, mask = state_tensors(state, device)
        captured = {}
        def capture(_, inp):
            captured["x"] = inp[0].detach()
        hook = model.model.transformer.blocks[31].register_forward_pre_hook(capture)
        model(noisy)
        hook.remove()
        h31, _ = l31_sham_hidden(model, captured["x"])
        allocation, direction = (value.to(device) for value in factorized[state_index])
        native = allocation[:, None] * direction
        measured, native_logits, native_hidden = evaluate(model, h31, [native, native], prior["tau"][state_index].to(device), clean, mask, state["p_mask"])
        old = prior["results"]
        nn_gate["kl_abs"][state_index] = (measured["kl"][0] - old["kl"][state_index, 1, 1]).abs()
        nn_gate["loss_abs"][state_index] = (measured["loss_delta"][0] - old["loss_delta"][state_index, 1, 1]).abs()
        nn_gate["final_energy_abs"][state_index] = (measured["final_hidden_relative_energy"][0] - old["final_hidden_relative_energy"][state_index, 1, 1]).abs()
        nn_gate["logit_energy_abs"][state_index] = (measured["masked_logit_relative_energy"][0] - old["masked_logit_relative_energy"][state_index, 1, 1]).abs()
        if max(value[state_index].item() for value in nn_gate.values()) > 1e-6:
            raise RuntimeError(f"NN gate failed at state {state_index}")
        state_cache.append((h31.cpu(), clean.cpu(), mask.cpu()))

    # Fresh batch-5 vs established batch-3 gate on state 0 / shift 1.
    h31, clean, mask = (value.to(device) for value in state_cache[0])
    allocation, direction = (value.to(device) for value in factorized[0])
    permutation = torch.roll(torch.arange(256, device=device), -1)
    cells = patterns(allocation, direction, permutation)
    five, five_logits, five_hidden = evaluate(model, h31, cells, prior["tau"][0].to(device), clean, mask, held["states"][0]["p_mask"])
    three, three_logits, three_hidden = evaluate(model, h31, [cells[0], cells[1]], prior["tau"][0].to(device), clean, mask, held["states"][0]["p_mask"])
    path_gate = {
        "nn_kl_abs": abs(five["kl"][0].item() - three["kl"][0].item()),
        "nn_loss_abs": abs(five["loss_delta"][0].item() - three["loss_delta"][0].item()),
        "nn_logits_max_abs": (five_logits[1].float() - three_logits[1].float()).abs().max().item(),
        "nn_final_hidden_max_abs": (five_hidden[1].float() - three_hidden[1].float()).abs().max().item(),
    }
    if max(path_gate.values()) > 1e-6:
        raise RuntimeError("batch-5 path gate failed")
    atomic_json(ROOT / "token_row_pairing_numerical_gate.json", {"path_gate": path_gate, "nn_gate": {key: {"mean": value.mean().item(), "max": value.max().item()} for key, value in nn_gate.items()}})

    SHARD_DIR.mkdir(exist_ok=True)
    offsets = permutations["unrestricted_offsets"]
    for state_index, state in enumerate(held["states"]):
        shard_path = SHARD_DIR / f"state_{state_index:02d}.pt"
        if shard_path.exists():
            existing = torch.load(shard_path, map_location="cpu", weights_only=True)
            if existing["manifest_sha256"] == frozen["manifest_sha256"]:
                print(f"state {state_index + 1}/40 already complete", flush=True)
                continue
            raise RuntimeError(f"incompatible existing shard {shard_path}")
        h31, clean, mask = (value.to(device) for value in state_cache[state_index])
        allocation, direction = (value.to(device) for value in factorized[state_index])
        unrestricted = [torch.roll(torch.arange(256), -int(offset)) for offset in offsets]
        unrestricted_results, unrestricted_spread = collect_family(model, h31, allocation, direction, unrestricted, prior["tau"][state_index].to(device), clean, mask, state["p_mask"])
        class_results, class_spread = collect_family(model, h31, allocation, direction, permutations["class_permutations"][state_index], prior["tau"][state_index].to(device), clean, mask, state["p_mask"])
        payload = {"state_index": state_index, "manifest_sha256": frozen["manifest_sha256"], "unrestricted": unrestricted_results, "class_preserving": class_results, "unrestricted_pre_norm_spread": unrestricted_spread, "class_pre_norm_spread": class_spread}
        tmp = Path(str(shard_path) + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(shard_path)
        print(f"state {state_index + 1}/40 complete: unrestricted=255 class={len(permutations['class_permutations'][state_index])}", flush=True)

    after = model_sha(model)
    if before != after:
        raise RuntimeError("model weights changed")
    shard_hashes = {path.name: file_sha(path) for path in sorted(SHARD_DIR.glob("state_*.pt"))}
    atomic_json(ROOT / "token_row_pairing_run_manifest.json", {"status": "complete", "state_count": len(shard_hashes), "unrestricted_conditions": 10200, "class_conditions": frozen["class_condition_count"], "manifest_sha256": frozen["manifest_sha256"], "shard_hashes": shard_hashes, "weight_sha_before": before, "weight_sha_after": after, "environment": {"python": platform.python_version(), "torch": torch.__version__}})


if __name__ == "__main__":
    main()
