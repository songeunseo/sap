"""OWL's official min-max/mean-centered allocation, plus explicit integer budgeting."""
import math
import numpy as np


def owl_sparsities(outlier_percent, target=.65, lam=.08):
    d = np.asarray(outlier_percent, dtype=np.float64)
    if d.ndim != 1 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('invalid outlier ratios')
    if np.ptp(d) == 0:
        raise ValueError('official OWL min-max undefined for identical outlier ratios')
    # Literal operations from prune_wanda_outlier at upstream dddb7a4b.
    density = (d-d.min()) * (1/(d.max()-d.min()) * lam*2)
    density = density-np.mean(density)+(1-target)
    s = 1-density
    if (s <= 0).any() or (s >= 1).any():
        raise ValueError('OWL requires clipping at this setting; stop rather than silently change rule')
    return s


def exact_row_counts(entries, sparsities, target):
    """Same block/input-width shares a row count. Minimize weighted squared rate error.

    Each group chooses floor(ideal)-1, floor(ideal), or floor(ideal)+1.
    DP enforces the exact historical Uniform65 integer budget; no damage/labels used.
    If this small rounding neighborhood is infeasible, stop rather than widen it.
    """
    groups = {}
    for i, row in enumerate(entries):
        block = i//7
        if row['name'].split('.')[0] != f'block_{block:02d}':
            raise ValueError('unexpected block ordering')
        out, width = row['shape']
        groups.setdefault((block, width), []).append((i, out))
    gains = [sum(o for _,o in members) for members in groups.values()]
    unit = math.gcd(*gains)
    if target % unit:
        raise ValueError('target not divisible by exact row-count unit')
    bases = [int(width*sparsities[b]) for b,width in groups]
    original = sum(g*k for g,k in zip(gains,bases))
    needed = (target-original)//unit
    dp = {0: (0., ())}
    for ((b,width), members), gain, base in zip(groups.items(), gains, bases):
        nxt = {}
        for offset,(cost,path) in dp.items():
            for delta in (-1,0,1):
                k = base+delta
                if not 0 <= k <= width: continue
                key = offset + delta*(gain//unit)
                val = cost + gain*width*(k/width-sparsities[b])**2
                if key not in nxt or val < nxt[key][0]:
                    nxt[key] = (val, path+(k,))
        dp = nxt
    if needed not in dp: raise ValueError('exact budget outside frozen rounding neighborhood')
    counts = [None]*len(entries)
    for members,k in zip(groups.values(),dp[needed][1]):
        for i,_ in members: counts[i]=k
    actual = sum(k*r['shape'][0] for k,r in zip(counts,entries))
    if actual != target: raise AssertionError('DP budget mismatch')
    return counts, dict(raw_owl_row_floor_pruned=original, corrected_pruned=actual,
        correction_pruned=actual-original, budget_error=actual-target,
        weighted_squared_rate_error=dp[needed][0], rounding_unit=unit,
        rule='block/input-width grouped floor-1,floor,floor+1; min parameter-weighted squared deviation; exact DP')
