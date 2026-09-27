"""Mock expensive model work, but exercise real receipt writing/reloading."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from experiments.dlm_role_token_geometry import run


class PipelineTests(unittest.TestCase):
    def test_evaluate_save_reload_without_gpu(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root=Path(temp); old=root/'old'; dest=root/'token_relative'
            stack.enter_context(patch.object(run,'ROOT',root))
            stack.enter_context(patch.object(run,'OLD',old))
            data=[dict(example_id=i,doc_hash=str(i),prompt_hash=str(i),target_hash=str(i),
                       reference_answer='#### 1',extracted_answer='1',correct=True,
                       evaluation_config_hash='protocol',generated_text='#### 1',method='token_relative') for i in range(100)]
            baseline=root/'baseline.jsonl';baseline.write_text(''.join(json.dumps(r)+'\n' for r in data))
            c=dict(pruned=10,baseline=str(baseline),protocol_hash='protocol')
            run.write(root/'config.json',c)
            run.write(old/'role65_mask_manifest.json',dict(entries=[]))
            run.write(dest/'mask_manifest.json',dict(entries=[dict(name='test',selected_mask=dict(mask_sha256='mask',pruned=10))]))
            run.write(dest/'allocation.json',dict(manifest_sha256=run.sha(str(dest/'mask_manifest.json'))))
            stack.enter_context(patch.object(run,'validate',return_value=c))
            stack.enter_context(patch('experiments.projection_capacity_allocation_65.run.load_dense',return_value=(object(),{})))
            stack.enter_context(patch('experiments.projection_capacity_followup_65.run_heldout.apply_manifest',return_value=10))
            stack.enter_context(patch('experiments.wanda_failure_characterization.run_failure_map.model_sha',return_value='modelhash'))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained',return_value=object()))
            stack.enter_context(patch('experiments.dlm_loss_aggregation.exp002.run.load_config',return_value=dict(model=dict(id='fake',revision='fake'))))
            stack.enter_context(patch('experiments.dlm_loss_aggregation.exp002.run._evaluation_config_hash',return_value=('protocol',{})))
            evaluator=stack.enter_context(patch('experiments.dlm_loss_aggregation.exp002.run._evaluate_gsm8k',return_value=({'eval_seconds':1},data)))
            run.evaluate('token_relative')
            self.assertEqual(run.completed('token_relative',c),data)
            self.assertEqual(run.read(dest/'pending_eval.json')['sparse_model_sha256'],'modelhash')
            self.assertEqual(run.read(dest/'results.json')['correct'],100)
            run.evaluate('token_relative')
            self.assertEqual(evaluator.call_count,1)


if __name__=='__main__':unittest.main()
