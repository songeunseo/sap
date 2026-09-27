import unittest
import numpy as np
from experiments.dlm_support_coverage50.core import support_metrics, budget_rates
from experiments.dlm_lsa_projection65.core import exact_projection_rows


class CoverageTests(unittest.TestCase):
    def test_same_pooled_energy_different_state_coverage(self):
        # Both have pooled energy [9,1]; second needs channel 2 for its rare state.
        a = support_metrics([[9,1],[9,1]], [1,1])
        b = support_metrics([[18,0],[0,2]], [1,1])
        self.assertEqual(a['pooled_count'], b['pooled_count'])
        self.assertEqual(a['coverage_count'], 1)
        self.assertEqual(b['coverage_count'], 2)

    def test_state_order_duplication_and_global_scale(self):
        x = np.array([[1.,3,6],[6,3,1],[3,5,2]])
        a = support_metrics(x, [2,1,3])
        for y in (x[::-1], np.tile(x,(2,1)), x*100):
            b = support_metrics(y, [2,1,3])
            self.assertEqual(a['pooled_count'], b['pooled_count'])
            self.assertEqual(a['coverage_count'], b['coverage_count'])

    def test_prefix_minimality_against_brute_force(self):
        rng = np.random.default_rng(2026)
        x = rng.uniform(.01,1,(7,13)); w = rng.uniform(.1,2,13)
        r = support_metrics(x,w)
        e = x*w; order = np.argsort(-e.mean(0),kind='stable')
        brute = next(k for k in range(1,14) if np.all(e[:,order[:k]].sum(1)/e.sum(1) >= .9))
        self.assertEqual(brute,r['coverage_count'])
        self.assertLess(r['min_coverage_one_fewer'],.9)

    def test_weight_energy_and_invalid_state(self):
        r = support_metrics([[1,1],[1,1]], [100,1])
        self.assertEqual(r['coverage_count'],1)
        with self.assertRaises(ValueError): support_metrics([[0,0]],[1,1])

    def test_parameter_weighted_budget_and_direction(self):
        sizes = [16,32,64,128]
        rates = budget_rates([0,1,2,3],sizes)
        self.assertAlmostEqual(np.average(rates,weights=sizes),.5)
        self.assertTrue(np.all(np.diff(rates) <= 0))
        self.assertTrue(np.all((rates >= .45)&(rates <= .55)))
        self.assertTrue(np.allclose(budget_rates([1]*4,sizes),.5))
        entries = [dict(shape=[2,8]),dict(shape=[4,8]),dict(shape=[8,8]),dict(shape=[16,8])]
        counts,receipt = exact_projection_rows(entries,rates,120)
        self.assertEqual(sum(k*e['shape'][0] for k,e in zip(counts,entries)),120)
        self.assertEqual(receipt['budget_error'],0)


if __name__ == '__main__': unittest.main()
