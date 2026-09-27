import torch
import torch.nn.functional as F


def tensor_pair_metrics(sham, variant, token_mask=None, eps=1e-30):
    """Reduce one exact finite perturbation to FP32 scalar sufficient statistics."""
    sham = sham.float()
    delta = variant.float() - sham
    delta_energy = delta.square().sum()
    sham_energy = sham.square().sum()
    cosine = F.cosine_similarity(sham.reshape(1, -1), delta.reshape(1, -1)).item()
    result = {
        "abs_energy": delta.square().mean().item(),
        "relative_energy": (delta_energy / sham_energy.clamp_min(eps)).item(),
        "relative_l2": (delta.norm() / sham.norm().clamp_min(eps)).item(),
        "sham_energy": sham_energy.item(),
        "delta_energy": delta_energy.item(),
        "sham_delta_cosine": cosine,
        "aligned_energy_fraction": cosine * cosine,
        "orthogonal_energy_fraction": max(0.0, 1.0 - cosine * cosine),
        "denominator_floored": bool(sham_energy <= eps),
    }
    if token_mask is not None:
        token_mask = token_mask.to(delta.device)
        masked = delta[token_mask].square().sum()
        unmasked = delta[~token_mask].square().sum()
        fraction = masked / (masked + unmasked).clamp_min(eps)
        position_fraction = token_mask.float().mean()
        result.update({
            "masked_abs_energy": delta[token_mask].square().mean().item(),
            "unmasked_abs_energy": delta[~token_mask].square().mean().item(),
            "masked_energy_fraction": fraction.item(),
            "masked_position_fraction": position_fraction.item(),
            "masked_enrichment": (fraction / position_fraction.clamp_min(eps)).item(),
        })
    return result


def propagation_gain(later_relative_l2, baseline_relative_l2, eps=1e-30):
    floored = baseline_relative_l2.abs() <= eps
    return later_relative_l2 / baseline_relative_l2.abs().clamp_min(eps), floored


def matched_perturbation(direction_delta, location_hidden, tau, eps=1e-30):
    direction = direction_delta.float()
    hidden = location_hidden.float()
    target_norm = torch.as_tensor(tau, device=direction.device, dtype=torch.float32) * hidden.norm()
    perturbation = direction / direction.norm().clamp_min(eps) * target_norm
    achieved = perturbation.norm() / hidden.norm().clamp_min(eps)
    return perturbation, achieved


def factorial_contrasts(cells):
    """Cells are indexed [direction (D30,D31), location (L30,L31)]."""
    k3030, k3031 = cells[..., 0, 0], cells[..., 0, 1]
    k3130, k3131 = cells[..., 1, 0], cells[..., 1, 1]
    location_d30 = k3031 - k3030
    location_d31 = k3131 - k3130
    direction_l30 = k3130 - k3030
    direction_l31 = k3131 - k3031
    return {
        "location_D30": location_d30,
        "location_D31": location_d31,
        "direction_L30": direction_l30,
        "direction_L31": direction_l31,
        "location_main": 0.5 * (location_d30 + location_d31),
        "direction_main": 0.5 * (direction_l30 + direction_l31),
        "interaction": direction_l31 - direction_l30,
    }


def cyclic_donor_map(state_count):
    return [(index + 1) % state_count for index in range(state_count)]


def same_timestep_donor_map(sequence_ids, timestep_ids):
    sequences = sorted(set(sequence_ids))
    lookup = {(sequence, timestep): index for index, (sequence, timestep) in enumerate(zip(sequence_ids, timestep_ids))}
    donors = []
    for sequence, timestep in zip(sequence_ids, timestep_ids):
        next_sequence = sequences[(sequences.index(sequence) + 1) % len(sequences)]
        donors.append(lookup[(next_sequence, timestep)])
    return donors


def factorize_token_feature(delta, eps=1e-12):
    """Return unit-L2 token allocation and per-token unit feature directions."""
    delta = delta.float()
    row_norms = delta.norm(dim=-1)
    if bool((row_norms <= eps).any()):
        raise ValueError("token-feature direction is undefined for a near-zero row")
    allocation = row_norms / row_norms.norm().clamp_min(eps)
    direction = delta / row_norms[:, None]
    return allocation, direction


def token_feature_factorial_contrasts(cells):
    """Cells use the preregistered order NN, NF, FN, FF."""
    nn, nf, fn, ff = (cells[..., index] for index in range(4))
    token_main = 0.5 * ((nn - fn) + (nf - ff))
    feature_main = 0.5 * ((nn - nf) + (fn - ff))
    return {
        "token_main": token_main,
        "feature_main": feature_main,
        "interaction": (nn - nf) - (fn - ff),
        "native_excess": nn - torch.maximum(nf, fn),
    }


def token_row_pairing_contrasts(cells):
    """Cells use the preregistered order NN, NS, SN, SS."""
    nn, ns, sn, ss = (cells[..., index] for index in range(4))
    return {
        "pairing": 0.5 * (nn + ss) - 0.5 * (ns + sn),
        "position": nn - ss,
        "feature_placement": sn - ns,
    }


def class_preserving_permutation(mask, shift):
    """Map each receiver row to a cyclically shifted source in the same mask class."""
    mask = mask.bool().flatten()
    permutation = torch.empty(mask.numel(), dtype=torch.long, device=mask.device)
    for value in (True, False):
        positions = torch.nonzero(mask == value, as_tuple=False).flatten()
        permutation[positions] = positions.roll(-int(shift) % positions.numel())
    return permutation
