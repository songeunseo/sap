import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.dlm_capacity_predictor import pipeline


class PipelineTests(unittest.TestCase):
    def test_live_exact_dependency_blocks_even_if_status_says_complete(self):
        done = {'status': 'complete', 'evaluations': [{'limit': 100}, {'limit': 1319}]}
        self.assertEqual(pipeline.dependency_state([pipeline.DEPENDENCY], done, []), 'waiting_existing_gsm8k')
        self.assertEqual(pipeline.dependency_state([pipeline.DEPENDENCY+'_another'], done, []), 'ready')
        self.assertEqual(pipeline.dependency_state([], done, [42]), 'waiting_gpu0')

    def test_dependency_failure_cannot_launch_next_stage(self):
        with self.assertRaises(RuntimeError):
            pipeline.dependency_state([], {'status': 'running', 'evaluations': []}, [])

    def test_diagnostics_boundary_has_no_heldout_or_gsm8k(self):
        stages = pipeline.stage_commands('diagnostics')
        self.assertEqual([name for name, _ in stages], ['audit', 'dense_collection', 'dense_analysis'])
        self.assertTrue(all('evaluate' not in ' '.join(command) for _, command in stages))
        self.assertEqual(pipeline.stage_commands('downstream')[-1][0], 'downstream')

    def test_failed_subprocess_is_recorded_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(subprocess.CalledProcessError):
                pipeline.run_stage('failed_fixture', [sys.executable, '-c', 'raise SystemExit(7)'], root, root)
            result = json.loads((root/'stage_failed_fixture.json').read_text())
            self.assertEqual(result['returncode'], 7)
            self.assertEqual(result['status'], 'failed')

    def test_code_snapshot_detects_changes_before_running_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.py'
            path.write_text('first')
            expected = pipeline.file_digest(path)
            pipeline.verify_snapshot({str(path): expected})
            path.write_text('changed')
            with self.assertRaises(RuntimeError):
                pipeline.verify_snapshot({str(path): expected})

    def test_live_dependency_does_not_need_a_result_file_yet(self):
        live = subprocess.CompletedProcess([], 0, stdout=pipeline.DEPENDENCY+'\n', stderr='')
        with patch.object(pipeline.subprocess, 'run', return_value=live), \
             patch.object(pipeline, 'DEPENDENCY_RESULT', Path('/missing/unfinished-result.json')):
            self.assertEqual(pipeline.observe_dependency(), 'waiting_existing_gsm8k')

    def test_readiness_is_rechecked_after_resource_becomes_busy(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(pipeline, 'ROOT', Path(directory)), \
             patch.object(pipeline, 'observe_dependency', side_effect=['waiting_gpu0', 'ready']), \
             patch.object(pipeline.time, 'sleep') as sleeper:
            pipeline.wait_for_ready('diagnostics')
            self.assertEqual(json.loads((Path(directory)/'status.json').read_text())['status'], 'ready')
            sleeper.assert_called_once_with(30)

    def test_snapshot_covers_imported_experiment_code(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo/'experiments/older/run.py'
            source.parent.mkdir(parents=True)
            source.write_text('pass')
            root = repo/'experiments/new'
            root.mkdir()
            for name in ['config.json', 'PLAN.md', 'run_tmux.sh']:
                (root/name).write_text('fixture')
            self.assertIn(source, pipeline.snapshot_paths(repo, root))


if __name__ == '__main__':
    unittest.main()
