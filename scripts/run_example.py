from pathlib import Path

from edge_tree_attacks import STRATEGIES, load_graph, run_attack
from edge_tree_attacks.output import write_rows


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    graph = load_graph(ROOT / "data" / "example" / "weighted_network.csv", "example")
    summaries = []
    for strategy in STRATEGIES:
        rows, summary = run_attack(
            graph,
            strategy,
            efficiency_pairs=100,
            efficiency_sources=10,
            betweenness_samples=10,
            weighted_betweenness_samples=10,
        )
        write_rows(ROOT / "results" / "example" / f"{strategy}.csv", rows)
        summaries.append({"strategy": strategy, **summary})
    write_rows(ROOT / "results" / "example" / "summary.csv", summaries)
    print(ROOT / "results" / "example" / "summary.csv")


if __name__ == "__main__":
    main()
