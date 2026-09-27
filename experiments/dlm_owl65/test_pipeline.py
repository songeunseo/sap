import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from experiments.dlm_owl65 import run


class PipelineTests(unittest.TestCase):
    def test_save_resume_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp);stack.enter_context(patch.object(run,'ROOT',root))
            rows=[dict(example_id=i,doc_hash=str(i),prompt_hash=str(i),target_hash=str(i),
                reference_answer='#### 1',extracted_answer='1',correct=True,evaluation_config_hash='p',
                generated_text='#### 1',method='owl') for i in range(100)]
            b=root/'baseline.jsonl';b.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            stack.enter_context(patch.object(run,'BASELINES',{'uniform':b,'role':b}))
            c=dict(pruned=10,protocol_hash='p',method='owl');run.write(root/'config.json',c)
            manifest=dict(entries=[],pruned=10);run.write(root/'mask_manifest.json',manifest)
            stack.enter_context(patch.object(run,'validate',return_value=c))
            stack.enter_context(patch.object(run,'build',return_value=manifest))
            stack.enter_context(patch('experiments.projection_capacity_allocation_65.run.load_dense',return_value=(object(),{})))
            stack.enter_context(patch('experiments.projection_capacity_followup_65.run_heldout.apply_manifest',return_value=10))
            stack.enter_context(patch('experiments.wanda_failure_characterization.run_failure_map.model_sha',return_value='modelhash'))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=object()))
            stack.enter_context(patch('experiments.dlm_loss_aggregation.exp002.run.load_config',return_value=dict(model=dict(id='fake',revision='fake'))))
            stack.enter_context(patch('experiments.dlm_loss_aggregation.exp002.run._evaluation_config_hash',return_value=('p',{})))
            ev=stack.enter_context(patch('experiments.dlm_loss_aggregation.exp002.run._evaluate_gsm8k',return_value=({'eval_seconds':1},rows)))
            run.run();self.assertTrue(run.completed(c));run.run();self.assertEqual(ev.call_count,1)
            self.assertEqual(run.read(root/'pending_eval.json')['sparse_model_sha256'],'modelhash')
            (root/'predictions.jsonl').write_text('tampered')
            with self.assertRaises(RuntimeError):run.completed(c)


if __name__=='__main__':unittest.main()
