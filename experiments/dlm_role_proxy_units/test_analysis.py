import unittest
import numpy as np
from experiments.dlm_role_proxy_units.analyze import feature, decomposition, MODELS
from experiments.dlm_role_exchange_prediction.analyze import features


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.row = dict(layer=3, projection_type='q_proj', base_level=3, new_level=4, parameter_delta=50,
                        features={c: dict(masked_token=1+i, unmasked_token=3+i, masked_energy=5+i, unmasked_energy=7+i)
                                  for i, c in enumerate(['dense', 'sparse'])})

    def test_old_specifications_reproduced(self):
        for new, old in [('D-raw', 'P2_role_dense'), ('DS-raw', 'P3_role_dense_sparse'), ('DS-both', 'P4_role_energy')]:
            self.assertEqual(feature(self.row, new), features(self.row, old))

    def test_roles_and_units(self):
        self.assertEqual(feature(self.row, 'D-relative')[-2:], [5, 7])
        self.assertEqual(feature(self.row, 'DS-relative')[-4:], [5, 7, 6, 8])
        self.assertEqual(len(feature(self.row, 'DS-both')) - len(feature(self.row, 'D-raw')), 6)

    def test_reject_nonfinite(self):
        self.row['features']['dense']['masked_energy'] = float('nan')
        with self.assertRaises(ValueError):
            feature(self.row, 'D-relative')

    def test_decomposition_cross_term(self):
        rows = [dict(document=0, layer=0), dict(document=0, layer=0), dict(document=1, layer=0)]
        r = decomposition(np.array([1., 2., 3.]), np.array([2., 3., 4.]), np.zeros(3), rows)
        self.assertLess(r['cross_term'], 0)
        self.assertAlmostEqual(r['mse'], r['nonadditivity_mse'] + r['single_prediction_sum_mse'] + r['cross_term'])


if __name__ == '__main__':
    unittest.main()
