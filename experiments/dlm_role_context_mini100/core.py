"""Pure role-error definitions. No new mask ranking or allocation solver."""
import numpy as np
import torch


def role_sums(candidate, reference, mask):
    if candidate.shape != reference.shape or candidate.ndim != 3 or candidate.shape[0] != 1:
        raise ValueError('Aligned batch-one outputs required')
    mask = mask.reshape(-1).to(candidate.device)
    if mask.numel() != candidate.shape[1] or not mask.any() or mask.all():
        raise ValueError('Both roles required')
    error = (candidate.float() - reference.float()).square()[0]
    result = torch.stack([error[mask].sum().double(), error[~mask].sum().double()])
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite error')
    return result.cpu().numpy()


def pooled_curves(numerators, denominators):
    num, den = np.asarray(numerators, float), np.asarray(denominators, float)
    if num.ndim != 4 or num.shape[1:] != (224, 6, 2) or den.shape != (224, 2):
        raise ValueError('Expected states x 224 x 6 x roles and fixed denominators')
    if not np.isfinite(num).all() or not np.isfinite(den).all() or (den <= 0).any() or (num < 0).any():
        raise ValueError('Invalid role statistics')
    values = num.sum(axis=0) / den[:, None, :]
    return values, values.max(axis=-1)
