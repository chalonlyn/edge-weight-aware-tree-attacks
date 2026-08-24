from __future__ import annotations

import csv
import heapq
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


STRATEGIES = ("RRN", "MaxDRN", "MaxBRN", "MaxWDRN", "MaxWBRN", "MaxEWTRN")


@dataclass(frozen=True)
class Graph:
    name: str
    adjacency: dict[int, list[tuple[int, float, float]]]
    total_weight: float
    edge_count: int


def graph_from_edges(name: str, edges: Iterable[tuple[int, int, float]]) -> Graph:
    adjacency: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    merged: dict[tuple[int, int], float] = defaultdict(float)
    for u, v, weight in edges:
        if u == v or not math.isfinite(weight) or weight <= 0:
            continue
        edge = (u, v) if u < v else (v, u)
        merged[edge] += weight
    for (u, v), weight in sorted(merged.items()):
        length = 1.0 / max(weight, 1e-12)
        adjacency[u].append((v, weight, length))
        adjacency[v].append((u, weight, length))
    return Graph(name, dict(adjacency), sum(merged.values()), len(merged))


def load_graph(path: str | Path, name: str | None = None) -> Graph:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        edges = ((int(row["source"]), int(row["target"]), float(row["weight"])) for row in reader)
        return graph_from_edges(name or path.parent.name, edges)


def active_scores(graph: Graph, active: set[int]) -> tuple[dict[int, int], dict[int, float], list[int]]:
    degree: dict[int, int] = {}
    strength: dict[int, float] = {}
    candidates: list[int] = []
    for node in active:
        neighbors = [(other, weight) for other, weight, _ in graph.adjacency.get(node, ()) if other in active]
        degree[node] = len(neighbors)
        strength[node] = sum(weight for _, weight in neighbors)
        if neighbors:
            candidates.append(node)
    return degree, strength, candidates


def attack_tree(graph: Graph, active: set[int], root: int, depth: int) -> set[int]:
    selected = {root}
    frontier = {root}
    for _ in range(1, depth):
        next_frontier = {
            neighbor
            for node in frontier
            for neighbor, _, _ in graph.adjacency.get(node, ())
            if neighbor in active and neighbor not in selected
        }
        if not next_frontier:
            break
        selected.update(next_frontier)
        frontier = next_frontier
    return selected


def incident_weight(graph: Graph, active: set[int], selected: set[int]) -> float:
    edges: set[tuple[int, int]] = set()
    total = 0.0
    for node in selected:
        for neighbor, weight, _ in graph.adjacency.get(node, ()):
            if neighbor not in active:
                continue
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if edge not in edges:
                edges.add(edge)
                total += weight
    return total


def _tie_choice(nodes: Sequence[int], rng: random.Random) -> int:
    ordered = sorted(nodes)
    return ordered[rng.randrange(len(ordered))]


def _best(scores: dict[int, float], rng: random.Random) -> int:
    value = max(scores.values())
    return _tie_choice([node for node, score in scores.items() if score == value], rng)


def _unweighted_betweenness(graph: Graph, active: set[int], samples: int, rng: random.Random) -> dict[int, float]:
    nodes = sorted(active)
    sources = nodes if samples <= 0 or samples >= len(nodes) else rng.sample(nodes, samples)
    centrality = {node: 0.0 for node in active}
    for source in sources:
        stack: list[int] = []
        predecessors = {node: [] for node in active}
        paths: dict[int, float] = defaultdict(float, {source: 1.0})
        distance = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            stack.append(node)
            for neighbor, _, _ in graph.adjacency.get(node, ()):
                if neighbor not in active:
                    continue
                if neighbor not in distance:
                    distance[neighbor] = distance[node] + 1
                    queue.append(neighbor)
                if distance[neighbor] == distance[node] + 1:
                    paths[neighbor] += paths[node]
                    predecessors[neighbor].append(node)
        dependency: dict[int, float] = defaultdict(float)
        while stack:
            node = stack.pop()
            if paths[node]:
                coefficient = (1.0 + dependency[node]) / paths[node]
                for predecessor in predecessors[node]:
                    dependency[predecessor] += paths[predecessor] * coefficient
            if node != source:
                centrality[node] += dependency[node]
    return centrality


def _weighted_betweenness(graph: Graph, active: set[int], samples: int, rng: random.Random) -> dict[int, float]:
    nodes = sorted(active)
    sources = nodes if samples <= 0 or samples >= len(nodes) else rng.sample(nodes, samples)
    centrality = {node: 0.0 for node in active}
    tolerance = 1e-12
    for source in sources:
        stack: list[int] = []
        predecessors = {node: [] for node in active}
        paths: dict[int, float] = defaultdict(float, {source: 1.0})
        distance = {source: 0.0}
        queue = [(0.0, source)]
        settled: set[int] = set()
        while queue:
            current_distance, node = heapq.heappop(queue)
            if node in settled:
                continue
            settled.add(node)
            stack.append(node)
            for neighbor, _, length in graph.adjacency.get(node, ()):
                if neighbor not in active:
                    continue
                candidate = current_distance + length
                known = distance.get(neighbor)
                if known is None or candidate < known - tolerance:
                    distance[neighbor] = candidate
                    paths[neighbor] = paths[node]
                    predecessors[neighbor] = [node]
                    heapq.heappush(queue, (candidate, neighbor))
                elif abs(candidate - known) <= tolerance:
                    paths[neighbor] += paths[node]
                    predecessors[neighbor].append(node)
        dependency: dict[int, float] = defaultdict(float)
        while stack:
            node = stack.pop()
            if paths[node]:
                coefficient = (1.0 + dependency[node]) / paths[node]
                for predecessor in predecessors[node]:
                    dependency[predecessor] += paths[predecessor] * coefficient
            if node != source:
                centrality[node] += dependency[node]
    return centrality


def select_root(
    graph: Graph,
    active: set[int],
    strategy: str,
    depth: int,
    rng: random.Random,
    betweenness_samples: int,
    weighted_betweenness_samples: int,
    betweenness: dict[int, float] | None = None,
    weighted_betweenness: dict[int, float] | None = None,
) -> int | None:
    degree, strength, candidates = active_scores(graph, active)
    if not candidates:
        return None
    if strategy == "RRN":
        return _tie_choice(candidates, rng)
    if strategy == "MaxDRN":
        return _best({node: degree[node] for node in candidates}, rng)
    if strategy == "MaxWDRN":
        return _best({node: strength[node] for node in candidates}, rng)
    if strategy == "MaxEWTRN":
        return _best({node: incident_weight(graph, active, attack_tree(graph, active, node, depth)) for node in candidates}, rng)
    if strategy == "MaxBRN":
        scores = betweenness or _unweighted_betweenness(graph, active, betweenness_samples, rng)
        return _best({node: scores.get(node, 0.0) for node in candidates}, rng)
    if strategy == "MaxWBRN":
        scores = weighted_betweenness or _weighted_betweenness(graph, active, weighted_betweenness_samples, rng)
        return _best({node: scores.get(node, 0.0) for node in candidates}, rng)
    raise ValueError(f"unknown strategy: {strategy}")


def _largest_component_weight(graph: Graph, active: set[int]) -> tuple[int, float]:
    remaining = set(active)
    best = (0, 0.0)
    while remaining:
        start = remaining.pop()
        component = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, _, _ in graph.adjacency.get(node, ()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        edges: set[tuple[int, int]] = set()
        weight = 0.0
        for node in component:
            for neighbor, edge_weight, _ in graph.adjacency.get(node, ()):
                if neighbor not in component:
                    continue
                edge = (node, neighbor) if node < neighbor else (neighbor, node)
                if edge not in edges:
                    edges.add(edge)
                    weight += edge_weight
        if weight > best[1] or (weight == best[1] and len(component) > best[0]):
            best = (len(component), weight)
    return best


def _sample_pairs(nodes: Sequence[int], pair_count: int, source_count: int, rng: random.Random) -> dict[int, list[int]]:
    if len(nodes) < 2 or pair_count <= 0:
        return {}
    sources = rng.sample(list(nodes), min(max(1, source_count), len(nodes)))
    per_source = max(1, math.ceil(pair_count / len(sources)))
    grouped: dict[int, list[int]] = {}
    for source in sources:
        targets: set[int] = set()
        attempts = 0
        while len(targets) < per_source and attempts < per_source * 30:
            target = nodes[rng.randrange(len(nodes))]
            attempts += 1
            if target != source:
                targets.add(target)
        grouped[source] = sorted(targets)
    return grouped


def _efficiency(graph: Graph, active: set[int], pairs: dict[int, list[int]]) -> float:
    total = 0.0
    for source, requested_targets in pairs.items():
        if source not in active:
            continue
        targets = {node for node in requested_targets if node in active}
        if not targets:
            continue
        distance = {source: 0.0}
        queue = [(0.0, source)]
        found: dict[int, float] = {}
        while queue and len(found) < len(targets):
            current_distance, node = heapq.heappop(queue)
            if current_distance != distance.get(node):
                continue
            if node in targets and node != source:
                found[node] = current_distance
            for neighbor, _, length in graph.adjacency.get(node, ()):
                if neighbor not in active:
                    continue
                candidate = current_distance + length
                if candidate < distance.get(neighbor, float("inf")):
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        total += sum(1.0 / found[node] for node in targets if node in found and found[node] > 0)
    return total


def _auc(rows: Sequence[dict[str, float]], x: str, y: str) -> float:
    return sum((right[x] - left[x]) * (left[y] + right[y]) / 2.0 for left, right in zip(rows, rows[1:]))


def run_attack(
    graph: Graph,
    strategy: str,
    *,
    depth: int = 2,
    efficiency_pairs: int = 200,
    efficiency_sources: int = 12,
    betweenness_samples: int = 12,
    weighted_betweenness_samples: int = 12,
    betweenness_refresh: int = 3,
    seed: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    rng = random.Random(seed)
    active = set(graph.adjacency)
    initial_nodes = len(active)
    pair_seed = None if seed is None else seed + 1
    pairs = _sample_pairs(sorted(active), efficiency_pairs, efficiency_sources, random.Random(pair_seed))
    initial_efficiency = _efficiency(graph, active, pairs)
    removed_nodes = 0
    removed_weight = 0.0
    rows: list[dict[str, float]] = []
    cached_betweenness: dict[int, float] | None = None
    cached_weighted: dict[int, float] | None = None
    cached_step = -1

    def record(step: int) -> None:
        component_size, component_weight = _largest_component_weight(graph, active)
        efficiency = _efficiency(graph, active, pairs)
        rows.append({
            "Step": float(step),
            "RN": removed_nodes / initial_nodes if initial_nodes else 0.0,
            "REW": removed_weight / graph.total_weight if graph.total_weight else 0.0,
            "EWLCC_prime": component_weight / graph.total_weight if graph.total_weight else 0.0,
            "Ee_prime": efficiency / initial_efficiency if initial_efficiency else 0.0,
            "LCC_size": float(component_size),
            "active_count": float(len(active)),
        })

    record(0)
    step = 0
    while True:
        if strategy in {"MaxBRN", "MaxWBRN"} and (cached_step < 0 or step - cached_step >= max(1, betweenness_refresh)):
            cached_betweenness = None
            cached_weighted = None
            if strategy == "MaxBRN":
                cached_betweenness = _unweighted_betweenness(graph, active, betweenness_samples, rng)
            else:
                cached_weighted = _weighted_betweenness(graph, active, weighted_betweenness_samples, rng)
            cached_step = step
        root = select_root(
            graph, active, strategy, depth, rng,
            betweenness_samples, weighted_betweenness_samples,
            cached_betweenness, cached_weighted,
        )
        if root is None:
            break
        selected = attack_tree(graph, active, root, depth)
        removed_weight += incident_weight(graph, active, selected)
        removed_nodes += len(selected)
        active.difference_update(selected)
        step += 1
        record(step)

    summary = {
        "auc_EWLCC_by_Step": _auc(rows, "Step", "EWLCC_prime"),
        "auc_EWLCC_by_RN": _auc(rows, "RN", "EWLCC_prime"),
        "auc_Ee_by_Step": _auc(rows, "Step", "Ee_prime"),
        "auc_Ee_by_RN": _auc(rows, "RN", "Ee_prime"),
        "final_step": rows[-1]["Step"],
        "final_RN": rows[-1]["RN"],
    }
    return rows, summary
