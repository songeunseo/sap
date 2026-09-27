import unittest
import numpy as np
import torch
from experiments.dlm_role_context_mini100.core import role_sums, pooled_curves


class TestCore(unittest.TestCase):
    def test_separate_roles(self):
        x = torch.tensor([[[1., 2.], [3., 4.]]])
        np.testing.assert_array_equal(role_sums(x, torch.zeros_like(x), torch.tensor([True, False])), [5, 25])

    def test_cross_term(self):
        a = torch.tensor([[[1., 2.], [3., 4.]]]); b = -a/2
        actual = role_sums(a+b, torch.zeros_like(a), torch.tensor([True, False]))
        np.testing.assert_allclose(actual, np.array([5, 25])/4)

    def test_pool_before_max(self):
        num = np.ones((2, 224, 6, 2)); den = np.tile([2., 4.], (224, 1))
        r, m = pooled_curves(num, den)
        np.testing.assert_allclose(r[:,:,0], 1); np.testing.assert_allclose(r[:,:,1], .5)
        np.testing.assert_allclose(m, 1)

    def test_reject_empty_role(self):
        with self.assertRaises(ValueError):
            role_sums(torch.zeros(1,2,3), torch.zeros(1,2,3), torch.ones(2, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
