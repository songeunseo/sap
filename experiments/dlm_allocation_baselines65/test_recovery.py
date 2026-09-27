import tempfile
import unittest
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from experiments.dlm_allocation_baselines65.recovery import checkpoint_generate,rng_state,restore_rng,rng_equal

def generate(obj,requests):
    return [f'{r.args}:{random.random()}:{np.random.rand()}:{torch.rand(1).item()}' for r in requests]

class RecoveryTests(unittest.TestCase):
    def test_actual_harness_loop_parity(self):
        from unittest.mock import patch
        import eval_llada
        class Tokenizer:
            def __call__(self,text):return {'input_ids':[ord(c) for c in text]}
            def decode(self,ids,**kw):return ''.join(chr(int(i)) for i in ids)
        h=SimpleNamespace(tokenizer=Tokenizer(),device='cpu',model=None,steps=2,gen_length=2,
            block_length=2,cfg=0,remasking='low_confidence',mask_id=126336,accelerator=None)
        requests=[SimpleNamespace(args=(f'prompt{i}',{'until':['STOP']})) for i in range(3)]
        def fake_generate(model,prompt,**kw):
            return torch.cat([prompt,torch.randint(65,90,(1,2))],dim=1)
        original=eval_llada.LLaDAEvalHarness.generate_until
        initial=rng_state()
        with patch.object(eval_llada,'generate',fake_generate),tempfile.TemporaryDirectory() as directory:
            expected=original(h,requests);final=rng_state();restore_rng(initial)
            checkpoint_generate(original,h,requests[:1],directory,{'v':1});restore_rng(initial)
            actual=checkpoint_generate(original,h,requests,directory,{'v':1})
            self.assertEqual(expected,actual);self.assertTrue(rng_equal(final,rng_state()))

    def test_resume_preserves_outputs_and_rng(self):
        requests=[SimpleNamespace(args=(str(i),{'until':['STOP']})) for i in range(5)]
        random.seed(19);np.random.seed(19);torch.manual_seed(19);initial=rng_state()
        expected=generate(None,requests);final=rng_state();restore_rng(initial)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_generate(generate,None,requests[:2],directory,{'model':'frozen'})
            restore_rng(initial)
            actual=checkpoint_generate(generate,None,requests,directory,{'model':'frozen'})
            self.assertEqual(actual,expected);self.assertTrue(rng_equal(final,rng_state()))

    def test_rejects_changed_input_or_rng(self):
        r=[SimpleNamespace(args=('a',{}))];initial=rng_state()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_generate(generate,None,r,directory,{'v':1})
            with self.assertRaises(RuntimeError):checkpoint_generate(generate,None,r,directory,{'v':1})
            restore_rng(initial)
            with self.assertRaises(RuntimeError):checkpoint_generate(generate,None,r,directory,{'v':2})
            with self.assertRaises(RuntimeError):checkpoint_generate(generate,None,[SimpleNamespace(args=('b',{}))],directory,{'v':1})

if __name__=='__main__':unittest.main()
