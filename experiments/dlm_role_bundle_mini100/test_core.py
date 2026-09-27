import unittest

from experiments.dlm_role_bundle_mini100.core import make_hybrids, holm, paired_ci, summarize
from experiments.dlm_role_bundle_mini100.status import generation_count


def manifest(counts, ids):
    return dict(pruned=sum(counts), weights=100, entries=[dict(name=f"m{i}", shape=[1, 10], level=i,
        selected_mask=dict(pruned=n, mask_sha256=s)) for i, (n, s) in enumerate(zip(counts, ids))])


class Tests(unittest.TestCase):
    def test_exact_budget_and_only_requested_changes(self):
        a = manifest([3, 5, 3, 5], ["a", "b", "c", "d"])
        r = manifest([5, 3, 5, 3], ["A", "B", "C", "D"])
        h = make_hybrids(a, r, [dict(id=0, indices=[0, 1]), dict(id=1, indices=[2, 3])])
        self.assertEqual([e["selected_mask"]["mask_sha256"] for e in h["add_b0"]["entries"]], ["A", "B", "c", "d"])
        self.assertEqual([e["selected_mask"]["mask_sha256"] for e in h["revert_b0"]["entries"]], ["a", "b", "C", "D"])
        self.assertTrue(all(x["pruned"] == 16 for x in h.values()))

    def test_unbalanced_rejected(self):
        with self.assertRaises(ValueError):
            make_hybrids(manifest([3, 5], ["a", "b"]), manifest([5, 3], ["A", "B"]), [dict(id=0, indices=[0]), dict(id=1, indices=[1])])

    def test_incomplete_partition_rejected(self):
        with self.assertRaises(ValueError):
            make_hybrids(manifest([3, 5], ["a", "b"]), manifest([5, 3], ["A", "B"]), [])

    def test_holm(self):
        self.assertEqual(holm({"a": .01, "b": .03, "c": .04}), {"a": .03, "b": .06, "c": .06})

    def test_ci_determinism(self):
        self.assertEqual(paired_ci([1]*10), [1.0, 1.0])

    def test_effect_signs(self):
        a = [{"correct": False}]*4
        r = [{"correct": True}]*4
        y = {f"{d}_b{i}": r if d == "add" else a for d in ("add", "revert") for i in range(6)}
        result = summarize(dict(aggregate=a, role=r), y)
        self.assertTrue(all(p["net_correct_a_minus_b"] == 4 for p in result["pairs"].values()))
        self.assertTrue(all(p["role_background_benefit_minus_aggregate_background_benefit"] == 0 for p in result["background_interactions"].values()))


if __name__ == "__main__": unittest.main()
