import hashlib
import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_loss_aggregation.run import _load_model, load_config, validate_config
from experiments.wanda_failure_characterization.propagation_core import factorize_token_feature, matched_perturbation, tensor_pair_metrics
from experiments.wanda_failure_characterization.run_failure_map import metrics, model_sha, state_tensors
from experiments.wanda_failure_characterization.run_state_swap import l31_sham_hidden, final_suffix


ROOT = Path(__file__).parent
EPS = 1e-12
KEYS = ("kl", "loss_delta", "top1_agreement", "confidence_mae", "achieved_relative_l2", "norm_deviation", "final_hidden_relative_energy", "masked_logit_relative_energy", "logit_rms")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def build_patterns(ar, ur, ad, ud):
    return (ar[:, None] * ur, ar[:, None] * ud, ad[:, None] * ur, ad[:, None] * ud)


def evaluate(model, h31, patterns, tau, clean, mask, p_mask):
    perturbations, achieved = zip(*(matched_perturbation(pattern, h31, tau, eps=EPS) for pattern in patterns))
    rows = torch.stack([h31.float()] + [h31.float() + value for value in perturbations]).to(h31.dtype)
    logits, hidden = final_suffix(model, rows)
    loss, kl, agreement, mae = metrics(logits, logits[:1], clean, mask, p_mask)
    result = {key: torch.empty(len(patterns)) for key in KEYS}
    for index in range(len(patterns)):
        row = index + 1
        fh = tensor_pair_metrics(hidden[0], hidden[row])
        lg = tensor_pair_metrics(logits[0, mask[0]], logits[row, mask[0]])
        result["kl"][index] = kl[row].cpu()
        result["loss_delta"][index] = (loss[row] - loss[0]).cpu()
        result["top1_agreement"][index] = agreement[row].cpu()
        result["confidence_mae"][index] = mae[row].cpu()
        result["achieved_relative_l2"][index] = achieved[index].cpu()
        result["norm_deviation"][index] = (achieved[index] - tau).abs().cpu()
        result["final_hidden_relative_energy"][index] = fh["relative_energy"]
        result["masked_logit_relative_energy"][index] = lg["relative_energy"]
        result["logit_rms"][index] = (logits[row, mask[0]].float() - logits[0, mask[0]].float()).square().mean().sqrt().cpu()
    return result, logits, hidden


@torch.inference_mode()
def main():
    mapping = json.loads((ROOT / "token_feature_donor_manifest.json").read_text())
    zero = json.loads((ROOT / "token_feature_zero_row_gate.json").read_text())
    held = json.loads((ROOT / "heldout_state_manifest.json").read_text())
    delta_art = torch.load(ROOT / "transplant_deltas.pt", map_location="cpu", weights_only=True)
    prior = torch.load(ROOT / "perturbation_transplant.pt", map_location="cpu", weights_only=True)
    transplant_manifest = json.loads((ROOT / "perturbation_transplant_manifest.json").read_text())
    if zero["status"] != "pass" or mapping["pair_count"] != 280:
        raise RuntimeError("frozen pre-evaluation gate is invalid")
    if delta_art["hashes"][1] != mapping["delta31_sha256"] or file_sha(ROOT / "transplant_deltas.pt") != transplant_manifest["delta_artifact_sha256"]:
        raise RuntimeError("delta31 artifact/hash mismatch")

    delta = delta_art["delta31"].float()
    allocations, directions = zip(*(factorize_token_feature(value, EPS) for value in delta))
    allocations, directions = torch.stack(allocations), torch.stack(directions)
    torch.save({"token_allocation": allocations, "row_norm_min": delta.norm(dim=-1).min(dim=1).values, "epsilon": EPS}, ROOT / "token_feature_sufficient_statistics.pt")

    config = load_config("experiments/dlm_loss_aggregation/config.yaml")
    validate_config(config)
    model, _ = _load_model(config)
    before = model_sha(model)
    device = model.model.transformer.wte.weight.device
    hiddens = []
    state_data = []
    for state in held["states"]:
        noisy, clean, mask = state_tensors(state, device)
        captured = {}
        def capture(_, inp):
            captured["x"] = inp[0].detach()
        hook = model.model.transformer.blocks[31].register_forward_pre_hook(capture)
        model(noisy)
        hook.remove()
        h31, _ = l31_sham_hidden(model, captured["x"])
        hiddens.append(h31.cpu())
        state_data.append((clean.cpu(), mask.cpu()))
    hiddens = torch.stack(hiddens)

    # Validate batch-5 against the established batch-3 path before selecting it.
    r = 0; d = mapping["pairs"][0]["donor_index"]
    patterns = build_patterns(allocations[r].to(device), directions[r].to(device), allocations[d].to(device), directions[d].to(device))
    clean, mask = (value.to(device) for value in state_data[r])
    five, five_logits, five_hidden = evaluate(model, hiddens[r].to(device), patterns, prior["tau"][r].to(device), clean, mask, held["states"][r]["p_mask"])
    group_a, a_logits, a_hidden = evaluate(model, hiddens[r].to(device), [patterns[1], patterns[0]], prior["tau"][r].to(device), clean, mask, held["states"][r]["p_mask"])
    group_b, b_logits, b_hidden = evaluate(model, hiddens[r].to(device), [patterns[3], patterns[2]], prior["tau"][r].to(device), clean, mask, held["states"][r]["p_mask"])
    batch_validation = {
        "batch5_vs_batch3_nn_kl_abs": abs(five["kl"][0].item() - group_a["kl"][1].item()),
        "batch5_vs_batch3_nn_logits_max_abs": (five_logits[1].float() - a_logits[2].float()).abs().max().item(),
        "batch3_repeat_sham_logits_max_abs": (a_logits[0].float() - b_logits[0].float()).abs().max().item(),
    }
    use_batch5 = max(batch_validation.values()) <= 1e-6
    batch_validation["selected_path"] = "batch5" if use_batch5 else "two_batch3_groups"
    atomic_json(ROOT / "token_feature_batch_path_gate.json", batch_validation)

    pair_count = len(mapping["pairs"])
    results = {key: torch.empty(pair_count, 4) for key in KEYS}
    similarities = {key: torch.empty(pair_count) for key in ("allocation_cosine", "allocation_spearman", "feature_weighted_cosine", "feature_mean_cosine")}
    nn_gate = {key: torch.empty(40) for key in ("kl_abs", "loss_abs", "final_energy_abs", "logit_energy_abs")}
    nn_seen = torch.zeros(40, dtype=torch.bool)
    repeatability = {}

    for pair in mapping["pairs"]:
        pi, receiver, donor = pair["pair_index"], pair["receiver_index"], pair["donor_index"]
        ar, ur = allocations[receiver].to(device), directions[receiver].to(device)
        ad, ud = allocations[donor].to(device), directions[donor].to(device)
        patterns = build_patterns(ar, ur, ad, ud)
        clean, mask = (value.to(device) for value in state_data[receiver])
        args = (model, hiddens[receiver].to(device))
        tail = (prior["tau"][receiver].to(device), clean, mask, held["states"][receiver]["p_mask"])
        if use_batch5:
            current, logits, hidden = evaluate(*args, patterns, *tail)
        else:
            a, a_logits, a_hidden = evaluate(*args, [patterns[1], patterns[0]], *tail)
            b, b_logits, b_hidden = evaluate(*args, [patterns[3], patterns[2]], *tail)
            current = {key: torch.stack((a[key][1], a[key][0], b[key][1], b[key][0])) for key in KEYS}
            logits = torch.stack((a_logits[0], a_logits[2], a_logits[1], b_logits[2], b_logits[1]))
            hidden = torch.stack((a_hidden[0], a_hidden[2], a_hidden[1], b_hidden[2], b_hidden[1]))
        for key in KEYS:
            results[key][pi] = current[key]

        similarities["allocation_cosine"][pi] = F.cosine_similarity(ar, ad, dim=0).cpu()
        similarities["allocation_spearman"][pi] = torch.corrcoef(torch.stack((ar.argsort().argsort().float(), ad.argsort().argsort().float())))[0, 1].cpu()
        token_cos = (ur * ud).sum(dim=-1)
        similarities["feature_weighted_cosine"][pi] = (ar.square() * token_cos).sum().cpu()
        similarities["feature_mean_cosine"][pi] = token_cos.mean().cpu()

        if not nn_seen[receiver]:
            old = prior["results"]
            nn_gate["kl_abs"][receiver] = (current["kl"][0] - old["kl"][receiver, 1, 1]).abs()
            nn_gate["loss_abs"][receiver] = (current["loss_delta"][0] - old["loss_delta"][receiver, 1, 1]).abs()
            nn_gate["final_energy_abs"][receiver] = (current["final_hidden_relative_energy"][0] - old["final_hidden_relative_energy"][receiver, 1, 1]).abs()
            nn_gate["logit_energy_abs"][receiver] = (current["masked_logit_relative_energy"][0] - old["masked_logit_relative_energy"][receiver, 1, 1]).abs()
            if max(value[receiver].item() for value in nn_gate.values()) > 1e-6:
                atomic_json(ROOT / "token_feature_nn_gate_failure.json", {key: value[:receiver + 1].tolist() for key, value in nn_gate.items()})
                raise RuntimeError(f"NN reproduction gate failed at receiver {receiver}")
            nn_seen[receiver] = True
        if pi == 0:
            again, again_logits, again_hidden = evaluate(*args, patterns if use_batch5 else [patterns[1], patterns[0]], *tail)
            reference_logits = logits if use_batch5 else a_logits
            reference_hidden = hidden if use_batch5 else a_hidden
            repeatability = {"logits_max_abs": (again_logits.float() - reference_logits.float()).abs().max().item(), "hidden_max_abs": (again_hidden.float() - reference_hidden.float()).abs().max().item()}
        if (pi + 1) % 20 == 0:
            print(f"factorization pair {pi + 1}/{pair_count}", flush=True)

    after = model_sha(model)
    if before != after:
        raise RuntimeError("model weights changed")
    output = ROOT / "token_feature_factorization.pt"
    torch.save({"results": results, "similarities": similarities, "tau": prior["tau"], "sequence_index": prior["sequence_index"], "timestep_index": prior["timestep_index"], "nn_gate": nn_gate, "batch_validation": batch_validation, "repeatability": repeatability, "mapping_sha256": mapping["mapping_sha256"]}, output)
    atomic_json(ROOT / "token_feature_factorization_manifest.json", {"status": "complete", "pairs": pair_count, "cells": ["NN", "NF", "FN", "FF"], "epsilon": EPS, "mapping_sha256": mapping["mapping_sha256"], "delta31_sha256": delta_art["hashes"][1], "result_sha256": file_sha(output), "sufficient_statistics_sha256": file_sha(ROOT / "token_feature_sufficient_statistics.pt"), "batch_validation": batch_validation, "repeatability": repeatability, "weight_sha_before": before, "weight_sha_after": after, "environment": {"python": platform.python_version(), "torch": torch.__version__}})


if __name__ == "__main__":
    main()
