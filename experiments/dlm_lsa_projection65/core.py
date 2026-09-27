"""Official LSA lsac mapping, with explicitly separate integer-row rounding."""
import math
import numpy as np
import torch


def projection_rates(metrics, numel, target=.65, lam=.07):
    # Literal blk_score_global arithmetic; no abs, block averaging, or sign flip.
    metric = torch.tensor(metrics, dtype=torch.float32)
    sizes = torch.tensor(numel, dtype=torch.int64)
    if metric.shape != sizes.shape or metric.ndim != 1 or not torch.isfinite(metric).all():
        raise ValueError('Invalid projection statistics')
    if (sizes <= 0).any() or float(metric.sum()) == 0:
        raise ValueError('Undefined projection mapping')
    importance = 1 - metric / metric.sum()
    if float(importance.max() - importance.min()) == 0:
        raise ValueError('Undefined public minmax')
    importance = (importance - importance.min()) / (importance.max() - importance.min()) * lam * 2
    pruned = sizes * target + (importance.mean() - importance) * sizes.float().mean()
    rates = pruned / sizes
    if not torch.isfinite(rates).all() or (rates <= 0).any() or (rates >= 1).any():
        raise ValueError('Official lsac produced out-of-range sparsity; do not clip')
    return rates.double().numpy()


def exact_projection_rows(entries, rates, target):
    """Each projection independently chooses floor-1/floor/floor+1; exact count DP."""
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (len(entries),) or not np.isfinite(rates).all() or np.any((rates <= 0) | (rates >= 1)):
        raise ValueError('Invalid rates')
    gains = [int(r['shape'][0]) for r in entries]
    widths = [int(r['shape'][1]) for r in entries]
    unit = math.gcd(*gains)
    if target % unit:
        raise ValueError('Infeasible exact row-count budget')
    bases = [math.floor(w*s) for w,s in zip(widths,rates)]
    original = sum(g*k for g,k in zip(gains,bases))
    needed = (target-original)//unit
    dp = {0: (0., ())}
    for gain,width,base,s in zip(gains,widths,bases,rates):
        nxt = {}
        for offset,(cost,path) in dp.items():
            for delta in (-1,0,1):
                k = base+delta
                if not 0 <= k <= width:
                    continue
                key = offset+delta*(gain//unit)
                value = cost+gain*width*(k/width-s)**2
                if key not in nxt or value < nxt[key][0]:
                    nxt[key] = (value,path+(k,))
        dp = nxt
    if needed not in dp:
        raise ValueError('Exact budget outside frozen rounding neighborhood')
    cost,counts = dp[needed]
    assert sum(k*g for k,g in zip(counts,gains)) == target
    return list(counts), dict(raw_row_floor_pruned=original, corrected_pruned=target,
        correction_pruned=target-original, budget_error=0, rounding_unit=unit,
        weighted_squared_rate_error=cost,
        rule='independent projection floor-1/floor/floor+1; exact parameter-count DP')
