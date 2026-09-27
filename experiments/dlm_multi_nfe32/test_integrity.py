"""Check artifact gates against deliberately damaged synthetic files."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from experiments.dlm_multiscale_ac50.artifacts import digest, sha, write
from experiments.dlm_multi_nfe32.analysis import read_cell

class ArtifactGates(unittest.TestCase):
    def fixture(self, root):
        fp = {'cell': 'Multi_32'}
        cfg = {'cells': {'Multi_32': {'fingerprint': fp, 'protocol_hash': 'synthetic'}}}
        requests = [{'example_id': 7}]
        row = dict(example_id=7,method='Multi',denoising_steps=32,generated_text='synthetic',
                   extracted_answer='1',correct=True)
        folder = root/'gsm8k/Multi_32'
        write(folder/'identity.json',dict(fingerprint=fp,document_ids=[7]))
        saved = dict(fingerprint=fp,row=row,row_sha256=digest(row))
        write(folder/'examples/0007.json',saved)
        write(folder/'predictions.json',[row])
        write(folder/'results.json',dict(status='complete',fingerprint=fp,total=1,correct=1,
                                         predictions_sha256=sha(folder/'predictions.json')))
        return cfg,requests,folder,saved

    def read(self,root,cfg,requests):
        # Isolate file-integrity gates; official grading is tested on the real caches.
        with patch('experiments.dlm_multi_nfe32.analysis.validate_prediction'), patch(
            'experiments.dlm_multi_nfe32.analysis.grade',return_value=dict(correct=True,extracted_answer='1')):
            return read_cell(root,cfg,requests,'Multi_32',None)

    def test_intact_fixture_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,requests,_,_=self.fixture(root)
            self.assertEqual(set(self.read(root,cfg,requests)),{7})

    def test_duplicate_numeric_filename_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,requests,folder,saved=self.fixture(root)
            write(folder/'examples/7.json',saved)
            with self.assertRaisesRegex(RuntimeError,'checkpoint filenames'):
                self.read(root,cfg,requests)

    def test_changed_sealed_answer_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,requests,folder,saved=self.fixture(root)
            saved['row']['generated_text']='changed'
            write(folder/'examples/0007.json',saved)
            with self.assertRaisesRegex(RuntimeError,'sealed checkpoint'):
                self.read(root,cfg,requests)

if __name__=='__main__':unittest.main()
