import hashlib

import torch
import torch.nn.functional as F

from lib.dlm_gradient_sensitivity import make_masked_state


def cgq_token_weights(input_ids, logits, mask_id):
    if logits.shape[:-1] != input_ids.shape:
        raise ValueError("logits and input_ids token shapes must match")
    confidence = logits.float().softmax(dim=-1).amax(dim=-1).detach()
    if not torch.isfinite(confidence).all() or not (
        (0 <= confidence) & (confidence <= 1)
    ).all():
        raise ValueError("confidence must be finite and in [0, 1]")
    mask_weight = torch.where(
        input_ids.eq(mask_id),
        torch.tensor(1.0, device=input_ids.device),
        torch.tensor(0.7, device=input_ids.device),
    )
    weights = mask_weight + confidence.sqrt()
    if not torch.isfinite(weights).all():
        raise ValueError("CGQ weights must be finite")
    return weights.detach(), confidence, mask_weight


def build_corrupted_states(clean_ids, timesteps, mask_id, seed):
    states = []
    for timestep_index, timestep in enumerate(timesteps):
        for sequence_index, ids in enumerate(clean_ids):
            mask_seed = seed + timestep_index * len(clean_ids) + sequence_index
            input_ids, mask, p_mask = make_masked_state(
                ids, timestep, mask_id, mask_seed
            )
            states.append(
                {
                    "input_ids": input_ids.cpu(),
                    "mask": mask.cpu(),
                    "p_mask": p_mask,
                    "timestep": float(timestep),
                    "timestep_index": timestep_index,
                    "sequence_index": sequence_index,
                    "mask_seed": mask_seed,
                }
            )
    return states


def state_digest(states):
    digest = hashlib.sha256()
    for state in states:
        ids = state["input_ids"].detach().cpu().contiguous()
        digest.update(str(tuple(ids.shape)).encode())
        digest.update(str(ids.dtype).encode())
        digest.update(format(float(state["timestep"]), ".17g").encode())
        digest.update(ids.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def cache_dense_references(model, states, device):
    references = []
    for state in states:
        input_ids = state["input_ids"].to(device)
        mask = state["mask"].to(device)
        logits = model(input_ids).logits
        references.append(
            {
                "logits": logits[mask].detach().to(device="cpu", dtype=torch.float16),
                "timestep": state["timestep"],
            }
        )
    return references


def masked_logit_sums(dense_logits, sparse_logits):
    if dense_logits.shape != sparse_logits.shape or dense_logits.ndim != 2:
        raise ValueError("masked dense and sparse logits must have the same 2D shape")
    dense_logp = F.log_softmax(dense_logits.float(), dim=-1)
    sparse_logp = F.log_softmax(sparse_logits.float(), dim=-1)
    dense_probability = dense_logp.exp()
    dense_confidence, dense_prediction = dense_probability.max(dim=-1)
    sparse_confidence, sparse_prediction = sparse_logp.exp().max(dim=-1)
    return {
        "count": dense_logits.shape[0],
        "kl_sum": (
            dense_probability * (dense_logp - sparse_logp)
        ).sum(dtype=torch.float64).item(),
        "agreement_count": dense_prediction.eq(sparse_prediction).sum().item(),
        "confidence_abs_error_sum": dense_confidence.sub(
            sparse_confidence
        ).abs().sum(dtype=torch.float64).item(),
    }


def diagonal_change(plain, cgq):
    if plain.shape != cgq.shape:
        raise ValueError("Hessian diagonals must have the same shape")
    plain = plain.float()
    cgq = cgq.float()
    return {
        "cosine_similarity": F.cosine_similarity(plain, cgq, dim=0).item(),
        "relative_l2_difference": (
            torch.linalg.vector_norm(cgq - plain)
            / torch.linalg.vector_norm(plain)
        ).item(),
    }
