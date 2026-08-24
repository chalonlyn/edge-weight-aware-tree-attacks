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
        for strategy in STRATEGIES:
            rows, summary = run_attack(
                self.graph,
                strategy,
                efficiency_pairs=30,
                efficiency_sources=6,
                betweenness_samples=6,
                weighted_betweenness_samples=6,
            )
            self.assertEqual(rows[0]["EWLCC_prime"], 1.0)
            self.assertGreaterEqual(summary["final_step"], 1.0)
            self.assertTrue(all(math.isfinite(value) for value in summary.values()))

    def test_output_schema(self) -> None:
        rows, summary = run_attack(
            self.graph,
            "MaxWDRN",
            efficiency_pairs=30,
            efficiency_sources=6,
        )
        self.assertEqual(
            set(rows[0]),
            {"Step", "RN", "REW", "EWLCC_prime", "Ee_prime", "LCC_size", "active_count"},
        )
        self.assertEqual(
            set(summary),
            {
                "auc_EWLCC_by_Step",
                "auc_EWLCC_by_RN",
                "auc_Ee_by_Step",
                "auc_Ee_by_RN",
                "final_step",
                "final_RN",
            },
        )

if __name__ == "__main__":
    unittest.main()
