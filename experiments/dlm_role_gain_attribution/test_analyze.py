import unittest

from experiments.dlm_role_gain_attribution.analyze import balanced_bundles, compare, outcome


def row(i, correct, answer):
    return dict(example_id=i, doc_hash=str(i), prompt_hash=str(i), target_hash=str(i),
                reference_answer=str(i), evaluation_config_hash="same", correct=correct,
                extracted_answer=answer)


class AuditTests(unittest.TestCase):
    def test_balanced_pairing_deterministic(self):
        changes = {0: 2, 1: -2, 2: 2, 3: -2}
        self.assertEqual(balanced_bundles(changes), [[0, 1], [2, 3]])

    def test_different_projection_sizes(self):
        changes = {0: 3, 1: -1, 2: -1, 3: -1}
        self.assertEqual(balanced_bundles(changes), [[0, 1, 2, 3]])

    def test_nonzero_budget_rejected(self):
        with self.assertRaises(ValueError):
            balanced_bundles({0: 2, 1: -1})

    def test_all_indices_once_and_budget(self):
        changes = {0: 3, 1: -2, 2: -1, 3: 7, 4: -7, 5: 4, 6: -4}
        bundles = balanced_bundles(changes)
        self.assertEqual(sorted(i for b in bundles for i in b), list(changes))
        for b in bundles:
            self.assertEqual(sum(changes[i] for i in b), 0)

    def test_transition_accounting(self):
        a = [row(0, False, "[invalid]"), row(1, True, "1"), row(2, False, "4")]
        r = [row(0, True, "0"), row(1, False, "3"), row(2, True, "2")]
        result = compare(a, r)
        self.assertEqual(result["net_via_strict_invalid"], 1)
        self.assertEqual(result["net_via_valid_wrong"], 0)
        self.assertEqual(result["paired_role_minus_aggregate"]["net_correct_a_minus_b"], 1)
        self.assertEqual(sum(sum(x.values()) for x in result["transition_aggregate_rows_role_columns"].values()), 3)

    def test_pair_identity_rejected(self):
        with self.assertRaises(ValueError):
            compare([row(0, True, "0")], [row(1, True, "1")])

    def test_valid_wrong_not_called_invalid(self):
        self.assertEqual(outcome(row(0, False, "-3")), "valid_wrong")


if __name__ == "__main__":
    unittest.main()
