import unittest
import numpy as np
import torch
from experiments.dlm_allocation_backtrace.analyze import factorization,normalized_mapping,association
from experiments.dlm_allocation_baselines65.core import dlp_rates,lsa_rates
from experiments.dlm_owl65.core import owl_sparsities

class Tests(unittest.TestCase):
    def test_factor_identity_and_scaling(self):
        torch.manual_seed(0);w=torch.randn(8,7);a=torch.rand(7)
        f=factorization(w,a);g=factorization(w,a*9)
        self.assertAlmostEqual(f['mean_score'],f['mean_abs_weight']*f['mean_channel_rms']*f['alignment'])
        self.assertAlmostEqual(g['mean_score']/f['mean_score'],3,places=6)
        self.assertAlmostEqual(g['alignment'],f['alignment'],places=6)
    def test_mapping_algebra(self):
        q=np.linspace(1,5,32)**2
        np.testing.assert_allclose(normalized_mapping(q,.15,1),dlp_rates(q),atol=1e-6)
        np.testing.assert_allclose(normalized_mapping(q,.1,1),lsa_rates(np.repeat(q,7)),atol=1e-6)
        np.testing.assert_allclose(normalized_mapping(q,.08,-1),owl_sparsities(q),atol=1e-12)
    def test_bootstrap_axes(self):
        x=np.arange(32.);y=np.tile(x[:,None],(1,8));draw=np.random.default_rng(0).integers(0,8,(100,8))
        r=association(x,y,draw)
        self.assertAlmostEqual(r['rho'],1)
        np.testing.assert_allclose(r['conditional_ci95'],[1,1])

if __name__=='__main__':unittest.main()
