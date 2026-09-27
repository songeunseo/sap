import unittest
import numpy as np
from experiments.dlm_role_exchange_prediction_v2.common import ridge,equal_layer_weights,bootstrap_cells

class CorrectiveTests(unittest.TestCase):
    def test_mean_loss_ridge_closed_form_and_replication(self):
        x=np.array([[-1.],[1.]]); y=np.array([-2.,2.])
        p,fit=ridge(x,y,x)
        np.testing.assert_allclose(p,[-1.,1.]) # var(x)=1 -> beta=2/(1+alpha)
        pp,_=ridge(np.tile(x,(100,1)),np.tile(y,100),x)
        np.testing.assert_allclose(pp,p,rtol=1e-12)
        self.assertAlmostEqual(fit['beta'][0],1.)

    def test_intercept_unpenalized(self):
        x=np.array([[0.],[1.],[2.]])
        p,_=ridge(x,np.ones(3)*17,x)
        np.testing.assert_allclose(p,17)

    def test_heldout_values_do_not_change_scaling(self):
        x=np.array([[0.],[2.],[4.]])
        _,a=ridge(x,np.arange(3),[[0.]])
        _,b=ridge(x,np.arange(3),[[1e12]])
        self.assertEqual(a,b)

    def test_equal_layer_estimand_and_bootstrap_agree(self):
        rows=[{'document':0,'layer':0}]*10+[{'document':0,'layer':4}]
        values=np.array([0.]*10+[2.]);w=equal_layer_weights(rows)
        self.assertAlmostEqual(np.average(values,weights=w),1.)
        result=bootstrap_cells(values,rows)
        self.assertAlmostEqual(result['mean'],1.)

if __name__=='__main__':unittest.main()
