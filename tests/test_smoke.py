import math
import unittest

from edge_tree_attacks import STRATEGIES, run_attack
from edge_tree_attacks.attack import graph_from_edges


class AttackSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        edges = [
            (0, 1, 4.0), (1, 2, 2.0), (2, 3, 1.0), (3, 4, 3.0),
            (4, 5, 1.0), (5, 0, 2.0), (1, 4, 5.0), (2, 5, 1.5),
        ]
        self.graph = graph_from_edges("test", edges)

    def test_all_strategies_finish(self) -> None:
        for index, strategy in enumerate(STRATEGIES):
            rows, summary = run_attack(
                self.graph,
                strategy,
                efficiency_pairs=30,
                efficiency_sources=6,
                betweenness_samples=6,
                weighted_betweenness_samples=6,
                seed=41 + index,
            )
            self.assertEqual(rows[0]["EWLCC_prime"], 1.0)
            self.assertGreaterEqual(summary["final_step"], 1.0)
            self.assertTrue(all(math.isfinite(value) for value in summary.values()))

    def test_fixed_seed_is_repeatable(self) -> None:
        left = run_attack(self.graph, "RRN", efficiency_pairs=30, efficiency_sources=6, seed=7)
        right = run_attack(self.graph, "RRN", efficiency_pairs=30, efficiency_sources=6, seed=7)
        self.assertEqual(left, right)

if __name__ == "__main__":
    unittest.main()
