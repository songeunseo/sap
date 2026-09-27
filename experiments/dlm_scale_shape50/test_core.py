import unittest
import numpy as np
import torch
from experiments.dlm_scale_shape50.core import summarize_tensor,boundary_changes,apply_change,select_pairs

class DiagnosticTests(unittest.TestCase):
    def test_scale_invariance_and_zero_mass(self):
        x=torch.tensor([0.,1.,2.,4.,8.])
        a=summarize_tensor(x,0);b=summarize_tensor(x*64,0)
        self.assertAlmostEqual(a['positive_log_variance'],b['positive_log_variance'],12)
        self.assertEqual(a['zero_fraction'],.2)
        self.assertAlmostEqual(a['mean_abs']*64,b['mean_abs'])
        self.assertEqual(a['log10_mean_normalized_hist'],b['log10_mean_normalized_hist'])
    def test_nested_exchange_exact_budget_and_restore(self):
        dense=torch.arange(1,33,dtype=torch.float32).reshape(4,8)
        order=torch.argsort(dense,dim=1,stable=True)
        change=boundary_changes(dense,order,4)
        base=dense.clone();base[:,:4]=0
        a,b=base.clone(),base.clone()
        apply_change(a,change,'prune');apply_change(b,change,'protect')
        self.assertEqual(int((a==0).sum()+(b==0).sum()),32)
        self.assertTrue(torch.equal(a[:,5:],dense[:,5:]))
        apply_change(a,change,'prune',restore=True);apply_change(b,change,'protect',restore=True)
        self.assertTrue(torch.equal(a,base));self.assertTrue(torch.equal(b,base))
    def test_selection_control_and_no_outcomes(self):
        rows=[]
        for layer in range(32):
            for ti,typ in enumerate(['a','b','c','d']):
                score=1+layer*.1+ti*.02
                rows.append(dict(name=f'block_{layer:02d}.{typ}',layer=layer,type=typ,shape=[4,64],
                    score_dense=dict(mean_abs=score,positive_log_variance=10-score)))
        pairs=select_pairs(rows,4,16)
        self.assertEqual(len(pairs),12)
        self.assertEqual(len({p[k] for p in pairs for k in ('prune_by_raw','prune_by_shape')}),24)
        for p in pairs:
            self.assertGreater(*p['raw_values']);self.assertLess(*p['shape_values'])
            a=int(p['prune_by_raw'].split('.')[0].split('_')[1]);b=int(p['prune_by_shape'].split('.')[0].split('_')[1])
            if p['stratum']=='same_layer_same_shape':self.assertEqual(a,b)
            if p['stratum']=='near_depth_same_type':self.assertLessEqual(abs(a-b),3)
    def test_impossible_budget_rejected(self):
        with self.assertRaises(ValueError):boundary_changes(torch.ones(4,8),torch.arange(8).repeat(4,1),3)

if __name__=='__main__':unittest.main()
