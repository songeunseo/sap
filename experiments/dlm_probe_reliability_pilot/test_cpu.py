"""CPU checks for the reliability estimator (no model)."""
import unittest

import numpy as np

from experiments.dlm_probe_reliability_pilot.run import reliability


class Reliability(unittest.TestCase):
    def test_pure_noise_is_unreliable(self):
        rng = np.random.default_rng(0)
        out = reliability(rng.normal(size=(32, 32)), np.random.default_rng(1), splits=200)
        self.assertLess(out["reliability"], 0.3)
        self.assertLess(abs(out["half_rho_mean"]), 0.2)

    def test_strong_block_effect_is_reliable(self):
        rng = np.random.default_rng(0)
        cost = rng.normal(size=(32, 1)) * 3 + rng.normal(size=(32, 32)) + rng.normal(size=(1, 32)) * 5
        out = reliability(cost, np.random.default_rng(1), splits=200)
        self.assertGreater(out["reliability"], 0.95)   # sequence main effect must not count as noise
        self.assertGreater(out["half_rho_mean"], 0.8)
        self.assertLess(out["reliability_8"], out["reliability"])


if __name__ == "__main__":
    unittest.main()
