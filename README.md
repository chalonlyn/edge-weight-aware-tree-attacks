# Edge-weight-aware tree attacks

This repository provides a minimal executable implementation of the six tree-attack root-selection rules described in *Edge-weight-aware tree attacks for cross-domain vulnerability analysis of complex networks*.

The current release is intended to verify installation, execution, input format, attack-tree construction, and output format on a small example network. Paper-scale datasets, experiment-specific configuration, random seeds, batch runners, and numerical result tables are not part of this initial release. A versioned reproduction archive will be deposited no later than publication.

## Requirements

- Python 3.10 or later

No third-party Python package is required.

## Run the example

From the repository root:

```bash
python -m scripts.run_example
```

The command writes one trajectory for each strategy and a summary file to `results/example/`.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

## Input format

Weighted networks are stored as CSV files with three columns:

```text
source,target,weight
0,1,1.2
1,2,2.5
```

Weights must be positive. Self-loops are ignored and repeated undirected edges are combined.

## Strategies

- `RRN`: random root node;
- `MaxDRN`: maximum-degree root node;
- `MaxBRN`: maximum-betweenness root node;
- `MaxWDRN`: maximum-weighted-degree root node;
- `MaxWBRN`: maximum-weighted-betweenness root node;
- `MaxEWTRN`: maximum edge-weight attack-tree root node.

## Output

Each trajectory contains the attack step, removed-node ratio, removed-edge-weight ratio, normalized edge weight of the largest connected component, and normalized edge-weighted efficiency.

## License

The source code is released under the MIT License. The example network is synthetic and is included only for testing the implementation.

