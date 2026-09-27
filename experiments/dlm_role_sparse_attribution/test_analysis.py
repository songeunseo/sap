import unittest
import numpy as np
from experiments.dlm_role_sparse_attribution.analyze import feature, MODELS
from experiments.dlm_role_proxy_units.analyze import feature as previous
from experiments.dlm_role_proxy_units.test_analysis import TestAnalysis as Fixture
from experiments.dlm_role_exchange_prediction_v2.common import ridge


class TestAttribution(unittest.TestCase):
    def setUp(self):
        f = Fixture(); f.setUp(); self.row = f.row

    def test_both_dense_roles_retained(self):
        base = previous(self.row, 'D-both')
        for m in MODELS:
            self.assertEqual(feature(self.row, m)[:len(base)], base)

    def test_only_requested_sparse_role(self):
        self.assertEqual(feature(self.row, 'plus_sparse_M')[-2:], [2, 6])
        self.assertEqual(feature(self.row, 'plus_sparse_U')[-2:], [4, 8])
        self.assertEqual(feature(self.row, 'plus_sparse_MU')[-4:], [2, 4, 6, 8])

    def test_ridge_column_permutation(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50, 8)); y = rng.normal(size=50); t = rng.normal(size=(10, 8))
        order = rng.permutation(8)
        a, _ = ridge(x, y, t); b, _ = ridge(x[:, order], y, t[:, order])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
