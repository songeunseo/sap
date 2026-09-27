"""Catch parameter-budget, precedence, and decision-gate errors."""
import unittest
import contextlib
import io
import json
import numpy as np

from experiments.projection_capacity_allocation_65.core import allocate, paired_gate


class AllocationTest(unittest.TestCase):
    def test_progress_event_accepts_projection_name(self):
        from experiments.projection_capacity_allocation_65.run import event
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            event('capacity_projection_complete', name='block_00.q_proj', module=1)
        self.assertEqual(json.loads(stream.getvalue())['name'], 'block_00.q_proj')

    def test_parameter_weighted_cost_and_budget(self):
        # Equal raw damage increments: a 3x larger matrix must go first.
        result = allocate([[0, 1, 2, 3, 4, 5]] * 2, [(1, 20), (3, 20)])
        self.assertEqual(result['levels'], [0, 4])
        self.assertEqual(result['pruned'], 52)
        self.assertEqual(result['budget_error'], 0)

    def test_negative_cost_retained_and_precedence_enforced(self):
        # The negative second increment is locked behind its costly first.
        result = allocate([[0, 100, -100, -99, -98, -97], [0, 1, 2, 3, 4, 5]],
                          [(1, 20), (1, 20)])
        self.assertEqual(result['trace'][0]['module_index'], 1)
        self.assertEqual(result['levels'], [1, 5])

    def test_row_floor_feasibility(self):
        shapes = [(4096, 4096)] * 4 + [(12288, 4096)] * 2 + [(4096, 12288)]
        curves = [[float(k * (i + 1)) for k in range(6)] for i in range(7)]
        result = allocate(curves, shapes)
        self.assertEqual(result['pruned'], 141750272)
        self.assertEqual(result['budget_error'], 0)

    def test_gate_requires_sequence_consistency(self):
        rows = [{'sequence_index': s, 'timestep': t, 'mean_kl': 10.0}
                for s in range(8) for t in [.1, .3, .5, .7, .9]]
        capacity = [dict(r, mean_kl=r['mean_kl'] + (-8 if r['sequence_index'] < 3 else .01)) for r in rows]
        gate = paired_gate(rows, capacity)
        self.assertLess(gate['bootstrap_95_ci'][1], 0)
        self.assertEqual(gate['sequence_means_improved'], 3)
        self.assertFalse(gate['passed'])

    def test_gate_sign_and_pairing(self):
        rows = [{'sequence_index': s, 'timestep': t, 'mean_kl': 2.0}
                for s in range(8) for t in [.1, .3, .5, .7, .9]]
        gate = paired_gate(rows, [dict(r, mean_kl=1.0) for r in rows])
        self.assertTrue(gate['passed'])
        self.assertEqual(gate['bootstrap_95_ci'], [-1.0, -1.0])
        self.assertEqual(gate['states_improved'], 40)
        with self.assertRaises(ValueError):
            paired_gate(rows, list(reversed(rows)))


if __name__ == '__main__':
    unittest.main()
