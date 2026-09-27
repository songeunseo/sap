import unittest

import torch

from experiments.dlm_block_uniform_wanda65.run import pooled_masks


class BlockUniformMaskTest(unittest.TestCase):
    def test_threshold_tie_fill_exact_budget(self):
        # Two one-row projections with tied scores exercise deterministic filling.
        class M:
            def __init__(self, w):
                self.weight = w

        model = object()
        names = list('abcdefg')
        mapping = {name: M(torch.ones(1, 4)) for name in names}
        refs = [{'name': name, 'shape': [1, 4], 'weights': 4} for name in names]
        states = []
        # Replace collector so this unit test does not run a model.
        import experiments.dlm_block_uniform_wanda65.run as mod
        original = mod.seq.collect_block
        try:
            mod.seq.collect_block = lambda *args, **kwargs: {name: torch.ones(4) for name in names}
            masks = pooled_masks(model, mapping, refs, states, 0, 3)
        finally:
            mod.seq.collect_block = original
        self.assertEqual(sum(int(m.sum()) for m in masks), 3)
        self.assertEqual(tuple(masks[0].shape), (1, 4))
        self.assertEqual(tuple(masks[1].shape), (1, 4))


if __name__ == '__main__':
    unittest.main()
