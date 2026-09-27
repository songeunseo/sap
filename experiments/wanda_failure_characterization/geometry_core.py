"""Matrix-free dense readout geometry. No targets or interventions in this module."""
import torch
import torch.nn.functional as F


def rms_forward(h, gamma, eps, bias=None):
    y = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * gamma
    return y if bias is None else y + bias


def rms_jvp(h, d, gamma, eps):
    """Exact differential gamma*(d/r - h*mean(h*d)/r^3)."""
    inv = torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)
    return gamma * (d * inv - h * (h * d).mean(-1, keepdim=True) * inv.pow(3))


@torch.no_grad()
def score_directions(h, directions, gamma, norm_eps, head_weight, norm_bias=None,
                     head_bias=None, scale=1.0, vocab_chunk=8192):
    """Masked h[T,H], directions[N,T,H]; output[N,3] in B_hidden,B_logit,Q order.

    Only vocabulary slices of the existing head are promoted to FP32. All
    intermediate projections have [tokens,vocabulary chunk] dimensions; there
    is no Jacobian, Fisher matrix, or hidden-square Gram matrix.
    """
    h, directions, gamma = h.float(), directions.float(), gamma.float()
    y = rms_forward(h, gamma, norm_eps, norm_bias)
    tangent = rms_jvp(h, directions, gamma, norm_eps)
    logits = []
    for start in range(0, head_weight.shape[0], vocab_chunk):
        stop = start + vocab_chunk
        bias = None if head_bias is None else head_bias[start:stop].float()
        logits.append(F.linear(y, head_weight[start:stop].float(), bias) * scale)
    dense_logits = torch.cat(logits, -1)
    del logits
    p = dense_logits.softmax(-1)
    first = torch.zeros(directions.shape[:2], device=h.device, dtype=torch.float64)
    second = torch.zeros_like(first)
    logit_energy = torch.zeros_like(first)
    for start in range(0, head_weight.shape[0], vocab_chunk):
        stop = start + vocab_chunk
        v = F.linear(tangent, head_weight[start:stop].float()) * scale
        prob = p[:, start:stop].unsqueeze(0)
        first += (prob * v).sum(-1, dtype=torch.float64)
        second += (prob * v.square()).sum(-1, dtype=torch.float64)
        logit_energy += v.square().sum(-1, dtype=torch.float64)
    variance = second - first.square()
    if variance.min() < -1e-9:
        raise RuntimeError('negative Fisher quadratic form beyond roundoff')
    hidden = directions.square().sum(-1, dtype=torch.float64).mean(-1)
    return torch.stack((hidden, logit_energy.mean(-1), .5 * variance.clamp_min(0).mean(-1)), -1)
