import os
import unittest
from unittest.mock import patch
from .runner import check_gpu

class SharedGPUChecks(unittest.TestCase):
    def check(self, free, owner):
        with patch('experiments.dlm_multi_nfe32.runner.idle_gpus',return_value=set()), patch.dict(
            os.environ,{'MULTI_ALLOW_SAME_USER_GPU':'1'}), patch(
            'experiments.dlm_multi_nfe32.runner.subprocess.check_output',
            side_effect=[f'1, GPU-abc, {free}\n','GPU-abc, 12345\n',str(owner)]):
            return check_gpu('1')

    def test_allows_own_job_with_headroom(self):
        self.assertEqual(self.check(80000,os.getuid())['mode'],'shared_same_user')

    def test_rejects_other_owner(self):
        with self.assertRaisesRegex(RuntimeError,'different user'):
            self.check(80000,os.getuid()+1)

    def test_rejects_insufficient_headroom(self):
        with self.assertRaisesRegex(RuntimeError,'48GiB'):
            self.check(40000,os.getuid())

if __name__=='__main__':unittest.main()
