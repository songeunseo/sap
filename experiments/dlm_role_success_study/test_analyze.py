import unittest

import numpy as np

from experiments.dlm_role_success_study.analyze import decompose, groups, contrasts, design_rank


class StudyTests(unittest.TestCase):
    def test_partition(self):
        g = groups([0, 1, 1, 0], [1, 0, 1, 0])
        self.assertTrue(all(x.sum() == 1 for x in g.values()))
        np.testing.assert_equal(sum(g.values()), np.ones(4))

    def test_collateral_is_not_gain_loss(self):
        d = decompose([0, 1], [1, 1], np.array([[0], [1]]), np.array([[1], [0]]))["0"]
        self.assertEqual(d["role_background_benefit"], 1)
        self.assertEqual(d["by_group"]["role_only"]["role_background_benefit"], 0)
        self.assertEqual(d["by_group"]["both_correct"]["role_background_benefit"], 1)

    def test_group_effects_sum_to_total(self):
        a = np.array([0, 1, 1, 0]); r = np.array([1, 0, 1, 0])
        add = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        for p in decompose(a, r, add, 1-add).values():
            self.assertEqual(p["add_benefit"], sum(g["add_benefit"] for g in p["by_group"].values()))
            self.assertEqual(p["role_background_benefit"], sum(g["role_background_benefit"] for g in p["by_group"].values()))

    def test_additive_contrasts_zero(self):
        a = np.array([0, 1]); effects = np.array([[1, 0], [0, -1]])
        r = a+effects.sum(1); add = a[:, None]+effects; revert = r[:, None]-effects
        np.testing.assert_equal(contrasts(a, r, add, revert)[:, 1:], np.zeros((2, 3)))

    def test_rank_deficient_pairwise(self):
        d = design_rank()
        self.assertEqual(d["additive_rank"], 7)
        self.assertLess(d["pairwise_rank"], d["pairwise_columns"])


if __name__ == "__main__": unittest.main()
