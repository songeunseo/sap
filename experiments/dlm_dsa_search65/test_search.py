import random
import unittest
import torch
from experiments.dlm_dsa_search65.run import engine_class, random_graph, offspring, feasible, FIXED
from experiments.dlm_allocation_baselines65.core import dsa_score

class SearchTests(unittest.TestCase):
    def test_fixed_public_math(self):
        x=torch.linspace(.1,2,256)
        self.assertAlmostEqual(engine_class().from_string(FIXED).compute_importance(x.clone()), dsa_score(x)['value'])
    def test_evolution_reproducible(self):
        E=engine_class(); a=random.Random(0); b=random.Random(0)
        gs=[random_graph(a,E) for _ in range(8)]
        self.assertEqual(gs,[random_graph(b,E) for _ in range(8)])
        children=offspring(a,gs[:4],set(gs),E,6)
        self.assertEqual(len(set(children)),6)
        for g in gs+children:
            self.assertTrue(feasible(g)); E.from_string(g)
if __name__=='__main__':
    unittest.main()
