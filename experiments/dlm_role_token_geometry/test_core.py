import unittest
import numpy as np
import torch
from experiments.dlm_role_token_geometry.core import sufficient_statistics, pool


class Tests(unittest.TestCase):
    def test_scale_vs_direction(self):
        y = torch.tensor([[[1., 0.], [0., 2.]]]); m = torch.tensor([True, False])
        s = sufficient_statistics(2*y, y, m)
        np.testing.assert_allclose(s[:, 2], [1, 1])
        np.testing.assert_allclose(s[:, 3], 0, atol=1e-12)

    def test_orthogonal(self):
        y = torch.tensor([[[1., 0.], [1., 0.]]]); z = y.flip(-1)
        s = sufficient_statistics(z, y, torch.tensor([True, False]))
        np.testing.assert_allclose(s[:, 2:4], [[2, 1], [2, 1]])

    def test_token_balancing_not_energy_weighting(self):
        y = torch.tensor([[[10., 0.], [1., 0.], [1., 0.]]]); z = y.clone(); z[:, 1] = 0
        s = sufficient_statistics(z, y, torch.tensor([True, True, False]))
        self.assertAlmostEqual(s[0, 2]/s[0, 4], .5)
        self.assertAlmostEqual(s[0, 0]/s[0, 1], 1/101)

    def test_zero_candidate_convention(self):
        y = torch.ones(1, 2, 2)
        s = sufficient_statistics(torch.zeros_like(y), y, torch.tensor([True, False]))
        np.testing.assert_equal(s[:, 3], 1)
        np.testing.assert_equal(s[:, 5], 1)

    def test_zero_dense_fails(self):
        with self.assertRaises(ValueError):
            sufficient_statistics(torch.ones(1, 2, 2), torch.zeros(1, 2, 2), torch.tensor([True, False]))

    def test_pool_by_total_token_count(self):
        x = np.zeros((2, 224, 6, 2, 6)); x[0, ..., 4] = 1; x[1, ..., 4] = 3
        x[0, ..., 2] = 2; x[1, ..., 2] = 6
        p = pool(x, np.ones((224, 2)))
        np.testing.assert_equal(p["token_relative"], 2)


if __name__ == "__main__": unittest.main()
