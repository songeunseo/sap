import unittest
import numpy as np
import torch
from experiments.dlm_reveal_kl_mini100.core import rank_rates,select_reveal,token_kl,pool_scores
from experiments.dlm_owl65.core import exact_row_counts
import json
from pathlib import Path

class Tests(unittest.TestCase):
    def test_direction_range_ties_and_scale(self):
        x=np.arange(32,dtype=float);r=rank_rates(x)
        self.assertAlmostEqual(r[0],.70);self.assertAlmostEqual(r[-1],.60);self.assertAlmostEqual(r.mean(),.65)
        np.testing.assert_array_equal(r,rank_rates(x*100+17));np.testing.assert_allclose(rank_rates(np.ones(32)),.65)
    def test_exact_actual_budget(self):
        ref=json.loads(Path('experiments/projection_capacity_allocation_65/candidate_mask_manifest.json').read_text())['entries']
        counts,meta=exact_row_counts(ref,rank_rates(np.arange(32)),4536008704)
        self.assertEqual(sum(k*r['shape'][0] for k,r in zip(counts,ref)),4536008704)
    def test_native_selection_ignores_observed(self):
        z=torch.tensor([[[20.,0.,0.],[0.,3.,0.],[0.,0.,1.]]]);ids=torch.tensor([[1,9,9]])
        tok,p=select_reveal(z,ids,9,1);self.assertEqual(p.tolist(),[1]);self.assertEqual(int(tok[0,p]),1)
    def test_kl_is_teacher_to_student_and_detects_content(self):
        d=torch.tensor([[.9,.1],[.5,.5]],dtype=torch.float64);p=torch.tensor([[.1,.9],[.5,.5]],dtype=torch.float64)
        kl=token_kl(d.log(),p.log());expected=(d*(d.log()-p.log())).sum(1)
        np.testing.assert_allclose(kl.numpy(),expected.numpy(),atol=2e-7);self.assertGreater(float(kl[0]),1.)
        np.testing.assert_allclose(token_kl(d.log(),d.log()).numpy(),0,atol=0)
    def test_prompt_pooling_and_duplicate_rejection(self):
        rows=[dict(sequence_index=q,step=t,reveal=q,all_masked=2*q) for q in range(8) for t in range(8,256,16)]
        self.assertEqual(pool_scores(rows)['reveal']['score'],3.5)
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):pool_scores(rows)
if __name__=='__main__':unittest.main()
