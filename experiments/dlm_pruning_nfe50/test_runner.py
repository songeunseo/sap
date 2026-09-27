"""CPU checks for decoder instrumentation and shared-state diagnostic math."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.dlm_multiscale_ac50.artifacts import digest
from experiments.dlm_pruning_nfe50.diagnostic import check_trace, metrics
from experiments.dlm_pruning_nfe50.runner import MeteredModel, capture_state


class FakeModel:
    device = torch.device('cpu')

    def __init__(self, vocab=16):
        self.vocab = vocab

    def __call__(self, ids):
        b, n = ids.shape
        logits = torch.zeros((b, n, self.vocab), dtype=torch.bfloat16)
        for pos in range(n):
            logits[:, pos, pos % self.vocab] = 1 + pos / 10
        return SimpleNamespace(logits=logits)


class InstrumentationTests(unittest.TestCase):
    def test_capture_preserves_native_topk_order_and_reveal(self):
        ids = torch.tensor([[1, 2] + [126336] * 10])
        logits = FakeModel()(ids).logits
        state = capture_state(ids, logits, 8)
        self.assertEqual(state['step'], 8)
        self.assertEqual(state['input_ids_sha256'], digest(ids[0].tolist()))
        self.assertEqual(len(state['targets']), 7)
        self.assertEqual(state['source']['position'], 11)
        self.assertTrue(all(t['position'] >= 2 for t in state['targets']))
        after = ids[0].tolist()
        after[11] = state['source']['token_id']
        self.assertEqual(state['after_input_ids_sha256'], digest(after))
        self.assertAlmostEqual(state['mask_fraction'], 10 / 12)
        self.assertAlmostEqual(state['response_mask_fraction'], 10 / 256)

    def test_failed_call_is_counted(self):
        class Broken:
            device = torch.device('cpu')
            def __call__(self, ids):
                raise RuntimeError('failed')
        count = dict(forwards=0, forward_tokens=0)
        model = MeteredModel(Broken(), count)
        with self.assertRaises(RuntimeError):
            model(torch.zeros(1, 17, dtype=torch.long))
        self.assertEqual(count, dict(forwards=1, forward_tokens=17))

    def test_trace_rejects_visible_selected_position(self):
        state = capture_state(torch.tensor([[1, 2] + [126336] * 10]),
                              FakeModel()(torch.zeros(1, 12, dtype=torch.long)).logits, 0)
        trace = dict(example_id=3, prompt_hash='prompt', config_identity='config',
                     states=[dict(state, step=step) for step in (0, 8, 16, 24)])
        request = dict(example_id=3, prompt_hash='prompt')
        with tempfile.TemporaryDirectory() as directory:
            check_trace(Path(directory), request, trace, 'config')
            trace['states'][0]['targets'][0]['position'] = 0
            with self.assertRaises(RuntimeError):
                check_trace(Path(directory), request, trace, 'config')

    def test_response_and_endpoint_metrics_are_distinct(self):
        before = torch.tensor([[.5, .3, .2]]).log()
        after = torch.tensor([[.2, .5, .3]]).log()
        sparse_before = torch.tensor([[.6, .2, .2]]).log()
        sparse_after = torch.tensor([[.3, .4, .3]]).log()
        state = dict(targets=[dict(position=4)], mask_fraction=.5, response_mask_fraction=.75)
        row = metrics(before, after, sparse_before, sparse_after, state)[0]
        self.assertAlmostEqual(row['dense_response_l1'], .6, places=6)
        self.assertAlmostEqual(row['response_error_l1'], 0, places=6)
        self.assertGreater(row['endpoint_kl_before'], 0)
        self.assertGreater(row['endpoint_kl_after'], 0)
        self.assertTrue(row['dense_flip'])
        self.assertFalse(row['disagreement_before'])


if __name__ == '__main__':
    unittest.main()
