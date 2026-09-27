"""Two frozen local damage proxies. Neither is a new weight selector."""
import numpy as np
import torch

METHODS = ("token_relative", "token_angular")


def sufficient_statistics(candidate, dense, mask):
    if candidate.shape != dense.shape or dense.ndim != 3 or dense.shape[0] != 1:
        raise ValueError("aligned batch1 outputs required")
    m = mask.reshape(-1).to(device=dense.device, dtype=torch.bool)
    if m.numel() != dense.shape[1] or not m.any() or m.all():
        raise ValueError("both roles required")
    y, z = dense.float()[0], candidate.float()[0]
    error = (z-y).square()
    ey = y.square().sum(-1)
    ez = z.square().sum(-1)
    if not torch.isfinite(ey).all() or (ey <= 0).any():
        raise ValueError("zero/nonfinite dense token energy; no tunable epsilon")
    relative = error.sum(-1) / ey
    denominator = (ey.double()*ez.double()).sqrt()
    cosine = (y*z).sum(-1).double()/torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    angular = 1-cosine.clamp(-1, 1)
    # Zero candidate vector has undefined angle; frozen convention: damage=1.
    angular = torch.where(ez > 0, angular, torch.ones_like(angular))
    result = []
    for group in (m, ~m):
        result.append(torch.stack((error[group].sum().double(), ey[group].sum().double(),
            relative[group].double().sum(), angular[group].sum(), group.sum().double(),
            (ez[group] == 0).sum().double())))
    values = torch.stack(result)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid proxy statistics")
    return values.cpu().numpy()


def pool(records, historical_denominators):
    x = np.asarray(records, dtype=float)
    if x.ndim != 5 or x.shape[1:] != (224, 6, 2, 6):
        raise ValueError("states×224×6levels×2roles×6statistics required")
    if not np.isfinite(x).all() or (x < 0).any() or (x[..., 4] <= 0).any():
        raise ValueError("invalid counts/statistics")
    summed = x.sum(0)
    den = np.asarray(historical_denominators, dtype=float)
    if den.shape != (224, 2) or (den <= 0).any():
        raise ValueError("historical role denominator required")
    return dict(control=summed[..., 0]/den[:, None, :],
                token_relative=summed[..., 2]/summed[..., 4],
                token_angular=summed[..., 3]/summed[..., 4])
