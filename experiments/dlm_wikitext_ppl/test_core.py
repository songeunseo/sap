import math
import unittest
import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.dlm_wikitext_ppl.core import articles, masks, mask_digest, normalized_nelbo, summarize


class Tests(unittest.TestCase):
    def test_runner_syntax(self):
        ast.parse(Path(__file__).with_name('run.py').read_text())

    def test_explicit_importance_correction(self):
        torch.manual_seed(2)
        logits=torch.randn(7,11); clean=torch.arange(7); mask=torch.tensor([1,0,1,0,0,1,0],dtype=torch.bool)
        explicit=(7/int(mask.sum()))*F.cross_entropy(logits[mask],clean[mask],reduction='sum')/7
        torch.testing.assert_close(normalized_nelbo(logits,clean,mask),explicit)

    def test_no_causal_shift(self):
        clean=torch.tensor([2,0,1]); logits=torch.full((3,4),-30.)
        logits[torch.arange(3),clean]=30.
        self.assertLess(float(normalized_nelbo(logits,clean,torch.ones(3,dtype=torch.bool))),1e-6)

    def test_uniform_model(self):
        logits=torch.zeros(9,17); clean=torch.arange(9)
        for mask in masks(9,128,2025):
            self.assertAlmostEqual(float(normalized_nelbo(logits,clean,mask)),math.log(17),places=5)

    def test_draw_reproducibility_and_cardinality(self):
        self.assertEqual(mask_digest(16,128,2),mask_digest(16,128,2))
        self.assertNotEqual(mask_digest(16,128,2),mask_digest(16,128,3))
        counts=[int(m.sum()) for m in masks(4,4000,2)]
        self.assertEqual(set(counts),{1,2,3,4})
        self.assertTrue(all(800<counts.count(k)<1200 for k in range(1,5)))

    def test_token_weighted_aggregation(self):
        rows=[dict(tokens=2,token_nelbo=1.,mc_variance=0.,mc_samples=128),
              dict(tokens=8,token_nelbo=3.,mc_variance=0.,mc_samples=128)]
        result=summarize(rows)
        self.assertAlmostEqual(result['token_nelbo'],2.6)
        self.assertAlmostEqual(result['ppl_upper_bound_estimate'],math.exp(2.6))

    def test_article_boundaries(self):
        result=articles([' = A = \n','text',' = = Nested = = \n','more',' = B = \n','tail'])
        self.assertEqual(len(result),2)
        self.assertIn('Nested',result[0]['text'])

    def test_invalid(self):
        with self.assertRaises(ValueError): normalized_nelbo(torch.zeros(3,4),torch.zeros(3,dtype=torch.long),torch.zeros(3,dtype=torch.bool))
        with self.assertRaises(ValueError): summarize([])


if __name__=='__main__': unittest.main()
