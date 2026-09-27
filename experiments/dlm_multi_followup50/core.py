"""Allocation transforms only: no model, GPU, diagnostic or task inputs."""
import numpy as np
from experiments.dlm_multiscale_ac50.core import rank_rates

ARMS = ('Multi-Bag', 'Multi-R2', 'Multi-R3.5')
SEED, DRAWS = 20260924, 200

def allocation_rates(span_costs):
    values = np.asarray(span_costs, dtype=np.float64)
    if values.shape != (32, 8) or not np.isfinite(values).all():
        raise ValueError('Expected finite block x span costs (32,8)')
    original = rank_rates(values.mean(axis=1))
    draws = np.random.default_rng(SEED).integers(0, 8, size=(DRAWS, 8))
    bag = np.mean([rank_rates(values[:, draw].mean(axis=1)) for draw in draws], axis=0)
    rates = dict(zip(ARMS, (bag, .5+.4*(original-.5), .5+.7*(original-.5))))
    if any(not np.isclose(v.mean(), .5, atol=1e-14, rtol=0) for v in rates.values()):
        raise AssertionError('Block-mean budget changed')
    return rates, draws
