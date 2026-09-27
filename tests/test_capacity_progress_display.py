import unittest
from scripts.watch_capacity_progress import Progress, OverallProgress


class ProgressDisplayTest(unittest.TestCase):
    def test_historical_events_do_not_fabricate_speed(self):
        p = Progress()
        for n in (8, 16):
            p.consume(dict(event='heldout_state', method='probe16', state=n), 10.)
        self.assertIn('계산 중', p.render(10.))

    def test_eta_uses_observed_deltas_and_resets_for_new_method(self):
        p = Progress()
        p.consume(dict(event='heldout_state', method='uniform', state=10), 100.)
        p.consume(dict(event='heldout_state', method='uniform', state=20), 120.)
        self.assertIn('0:00:40', p.render(120.))
        p.consume(dict(event='heldout_state', method='probe16', state=1), 200.)
        self.assertIn('계산 중', p.render(200.))

    def test_completed_counter_does_not_claim_whole_experiment_done(self):
        p = Progress()
        p.consume(dict(event='dense_reference', state=40, total=40), 100.)
        self.assertFalse(p.finished)
        self.assertIn('다음 단계 준비', p.render(100.))

    def test_overall_eta_includes_remaining_models_and_hash_after_current_forward(self):
        p = OverallProgress(['a', 'b'], load_seconds=10, state_seconds=2)
        p.consume(dict(event='heldout_state', method='a', state=20), 100.)
        # Current 20 states*2 + final hash10 + next (prep30 + 40 states*2).
        self.assertEqual(p.remaining_seconds(100.), 160.)
        p.consume(dict(event='heldout_state', method='a', state=40), 140.)
        self.assertEqual(p.remaining_seconds(140.), 120.)
        self.assertIn('전체 ETA', p.render(140.))

    def test_dense_done_still_includes_all_sparse_models(self):
        p = OverallProgress(['a', 'b', 'c'], load_seconds=10, state_seconds=2)
        p.consume(dict(event='dense_reference', state=40, total=40), 100.)
        self.assertEqual(p.remaining_seconds(100.), 330.)

    def test_complete_event_is_only_unconditional_zero_eta(self):
        p = OverallProgress(['a'], load_seconds=10, state_seconds=2)
        p.consume(dict(event='heldout_state', method='a', state=40), 100.)
        self.assertGreater(p.remaining_seconds(1000.), 0.)
        p.consume(dict(event='heldout_complete', decisions={'a': 'PASS'}), 1001.)
        self.assertEqual(p.remaining_seconds(1001.), 0.)


if __name__ == '__main__':
    unittest.main()
