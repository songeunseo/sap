import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from experiments.dlm_wikitext_ppl.core import mask_digest
from experiments.dlm_wikitext_ppl.evaluate import evaluate_block,seal,verify_row


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(11,11)
    def forward(self,ids):return SimpleNamespace(logits=self.embedding(ids))


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2);self.model=Model().eval()
        self.cfg=dict(mc_samples=16,mask_id=10)
        self.source=dict(block_id='validation:test',clean_ids=[1,2,3],mask_seed=2,
                         mask_sha256=mask_digest(3,16,2))
    def test_evaluation_and_checkpoint_validation(self):
        ticks=[];row=evaluate_block(self.model,self.source,self.cfg,ticks.append)
        row=seal(dict(row,config_sha256='config'))
        verify_row(row,self.source,self.cfg,'config')
        self.assertEqual(ticks,[16]);self.assertEqual(row['tokens'],3)
        self.assertEqual(len(row['sample_token_nelbo']),16)
        again=evaluate_block(self.model,self.source,self.cfg)
        self.assertEqual(row['sample_token_nelbo'],again['sample_token_nelbo'])
    def test_changed_mask_rejected(self):
        source=dict(self.source,mask_seed=3)
        with self.assertRaises(RuntimeError):evaluate_block(self.model,source,self.cfg)
    def test_tampered_checkpoint_rejected(self):
        row=seal(dict(evaluate_block(self.model,self.source,self.cfg),config_sha256='config'))
        changed=copy.deepcopy(row);changed['sample_token_nelbo'][0]+=1
        with self.assertRaises(RuntimeError):verify_row(changed,self.source,self.cfg,'config')
    def test_wrong_candidate_checkpoint_rejected(self):
        row=seal(dict(evaluate_block(self.model,self.source,self.cfg),config_sha256='other'))
        with self.assertRaises(RuntimeError):verify_row(row,self.source,self.cfg,'config')
    def test_short_tail(self):
        source=dict(self.source,clean_ids=[1],mask_sha256=mask_digest(1,16,2))
        row=evaluate_block(self.model,source,self.cfg)
        self.assertEqual(row['tokens'],1)


if __name__=='__main__':unittest.main()
