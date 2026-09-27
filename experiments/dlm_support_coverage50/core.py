"""Fixed pooled-order channel support: state coverage versus pooled energy."""
import numpy as np
from scipy.stats import rankdata


def support_metrics(state_energy, column_weight_energy, coverage=.9):
    """Channel diagonal energy; NOT exact output/reconstruction energy.

    Both conditions use the SAME descending pooled-energy channel order.
    State coverage is the shortest prefix covering >=coverage of EACH state.
    This is optimal among these prefixes, not among arbitrary channel subsets.
    """
    x = np.asarray(state_energy, dtype=np.float64)
    w = np.asarray(column_weight_energy, dtype=np.float64)
    if x.ndim != 2 or w.shape != (x.shape[1],) or not 0 < coverage < 1:
        raise ValueError('Invalid shape or coverage')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (x < 0).any() or (w < 0).any():
        raise ValueError('Finite nonnegative energies required')
    energy = x * w[None, :]
    totals = energy.sum(1)
    if (totals <= 0).any():
        raise ValueError('Zero-energy state; no implicit epsilon policy')
    pooled = energy.mean(0)
    order = np.argsort(-pooled, kind='stable')
    cdf = np.cumsum(energy[:, order], axis=1) / totals[:, None]
    pooled_cdf = np.cumsum(pooled[order]) / pooled.sum()
    state_k = (cdf < coverage).sum(1) + 1
    pooled_k = int(np.searchsorted(pooled_cdf, coverage) + 1)
    shared_k = int(state_k.max())
    if not 1 <= pooled_k <= shared_k <= x.shape[1]:
        raise AssertionError('Invalid nested support')
    if cdf[:, shared_k-1].min() < coverage - 1e-12:
        raise AssertionError('Shared support does not cover every state')
    return dict(pooled_count=pooled_k, coverage_count=shared_k,
                pooled_score=pooled_k/x.shape[1], coverage_score=shared_k/x.shape[1],
                excess_fraction=(shared_k-pooled_k)/x.shape[1],
                state_required_counts=state_k.tolist(),
                state_coverage_at_pooled=cdf[:, pooled_k-1].tolist(),
                min_coverage_at_shared=float(cdf[:, shared_k-1].min()),
                min_coverage_one_fewer=float(cdf[:, shared_k-2].min()) if shared_k > 1 else 0.,
                mean_state_coverage_at_pooled=float(cdf[:, pooled_k-1].mean()))


def budget_rates(scores, numel, target=.5, halfwidth=.05):
    """Larger support requirement -> lower sparsity; same map for both arms."""
    scores = np.asarray(scores, dtype=float)
    sizes = np.asarray(numel, dtype=float)
    if scores.ndim != 1 or len(scores) < 2 or scores.shape != sizes.shape:
        raise ValueError('Invalid score sizes')
    if not np.isfinite(scores).all() or not np.isfinite(sizes).all() or (sizes <= 0).any():
        raise ValueError('Invalid scores or parameter counts')
    ranks = (rankdata(scores, method='average')-1)/(len(scores)-1)
    raw = target-2*halfwidth*(ranks-np.average(ranks, weights=sizes))
    lo, hi = -1., 1.
    for _ in range(80):
        shift = (lo+hi)/2
        rates = np.clip(raw+shift, target-halfwidth, target+halfwidth)
        if np.average(rates, weights=sizes) < target:
            lo = shift
        else:
            hi = shift
    rates = np.clip(raw+(lo+hi)/2, target-halfwidth, target+halfwidth)
    if abs(np.average(rates, weights=sizes)-target) > 1e-12:
        raise AssertionError('Continuous budget mismatch')
    return rates
