"""Generate DAG datasets for lexicographic topological sorting."""

from __future__ import annotations

import argparse
import heapq
import json
import random
from collections import Counter
from pathlib import Path


def lexicographic_topological_sort(n: int, edges: list[tuple[int, int]]) -> tuple[list[int], list[list[int]]]:
    adj = [[] for _ in range(n)]
    indegree = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    heap = [v for v in range(n) if indegree[v] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    states: list[list[int]] = []
    removed = [False] * n

    while heap:
        u = heapq.heappop(heap)
        states.append([-1 if removed[v] else indegree[v] for v in range(n)])
        order.append(u)
        removed[u] = True
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(heap, v)

    if len(order) != n:
        raise ValueError("Generated graph is not a DAG")
    return order, states


def build_example(n: int, edges: set[tuple[int, int]], split_type: str) -> dict:
    sorted_edges = sorted(edges)
    order, indegree_states = lexicographic_topological_sort(n, sorted_edges)
    return {
        "n": n,
        "edges": [[u, v] for u, v in sorted_edges],
        "order": order,
        "indegree_states": indegree_states,
        "num_edges": len(sorted_edges),
        "split_type": split_type,
    }


def sample_dag(
    rng: random.Random,
    min_n: int,
    max_n: int,
    edge_prob: float,
    split_type: str = "random",
) -> dict:
    n = rng.randint(min_n, max_n)
    topo = list(range(n))
    rng.shuffle(topo)
    edges: set[tuple[int, int]] = set()
    for i, u in enumerate(topo):
        for v in topo[i + 1 :]:
            if rng.random() < edge_prob:
                edges.add((u, v))

    if not edges and n > 1:
        edges.add((topo[0], topo[1]))
    return build_example(n, edges, split_type)


def sample_long_chain(rng: random.Random, min_n: int, max_n: int) -> dict:
    n = rng.randint(min_n, max_n)
    chain = list(range(n))
    rng.shuffle(chain)
    edges = {(chain[i], chain[i + 1]) for i in range(n - 1)}
    return build_example(n, edges, "long_chain")


def sample_adversarial_hub(rng: random.Random, min_n: int, max_n: int) -> dict:
    n = rng.randint(max(4, min_n), max_n)
    vertices = list(range(n))
    rng.shuffle(vertices)
    hub = vertices[-1]
    chain = vertices[:-1]
    edges = {(chain[i], chain[i + 1]) for i in range(len(chain) - 1)}
    edges.update((u, hub) for u in chain)

    # Add a few forward edges inside the forced chain to vary static degree patterns.
    for i, u in enumerate(chain):
        for v in chain[i + 2 :]:
            if rng.random() < 0.15:
                edges.add((u, v))
    return build_example(n, edges, "adversarial_hub")


def sample_layered_dag(rng: random.Random, min_n: int, max_n: int) -> dict:
    n = rng.randint(min_n, max_n)
    vertices = list(range(n))
    rng.shuffle(vertices)
    n_layers = rng.randint(3, min(6, n))
    layers = [[] for _ in range(n_layers)]
    for idx, vertex in enumerate(vertices):
        layers[idx % n_layers].append(vertex)

    edges: set[tuple[int, int]] = set()
    for i, layer in enumerate(layers[:-1]):
        later = [v for next_layer in layers[i + 1 :] for v in next_layer]
        for u in layer:
            targets = rng.sample(later, k=rng.randint(1, min(3, len(later))))
            edges.update((u, v) for v in targets)
    return build_example(n, edges, "layered")


def generate_dataset(
    rng: random.Random,
    n_examples: int,
    min_n: int,
    max_n: int,
    edge_prob: float,
    split_type: str = "random",
) -> list[dict]:
    return [sample_dag(rng, min_n, max_n, edge_prob, split_type) for _ in range(n_examples)]


def summarize(name: str, items: list[dict]):
    types = Counter(item["split_type"] for item in items)
    avg_n = sum(item["n"] for item in items) / max(len(items), 1)
    avg_edges = sum(item["num_edges"] for item in items) / max(len(items), 1)
    print(
        f"{name}: n={len(items)} | "
        f"N=[{min(item['n'] for item in items)}, {max(item['n'] for item in items)}] | "
        f"avg_N={avg_n:.2f} avg_edges={avg_edges:.2f} | types={dict(types)}"
    )


def write_json(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate topological sorting data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=80_000)
    parser.add_argument("--n_val", type=int, default=2_000)
    parser.add_argument("--n_test", type=int, default=2_000)
    parser.add_argument("--min_n", type=int, default=6)
    parser.add_argument("--max_n", type=int, default=12)
    parser.add_argument("--edge_prob", type=float, default=0.22)
    parser.add_argument("--dense_edge_prob", type=float, default=0.38)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits = {
        "train": generate_dataset(rng, args.n_train, args.min_n, args.max_n, args.edge_prob),
        "val": generate_dataset(rng, args.n_val, args.min_n, args.max_n, args.edge_prob),
        "test_id": generate_dataset(rng, args.n_test, args.min_n, args.max_n, args.edge_prob),
        "test_ood_dense": generate_dataset(
            rng, args.n_test, args.min_n, args.max_n, args.dense_edge_prob, "dense"
        ),
        "test_ood_long_chain": [
            sample_long_chain(rng, args.min_n, args.max_n) for _ in range(args.n_test)
        ],
        "test_ood_adversarial_hub": [
            sample_adversarial_hub(rng, args.min_n, args.max_n) for _ in range(args.n_test)
        ],
        "test_ood_layered": [
            sample_layered_dag(rng, args.min_n, args.max_n) for _ in range(args.n_test)
        ],
    }

    output_dir = Path(args.output_dir)
    for name, items in splits.items():
        write_json(output_dir / f"{name}.json", items)
        summarize(name, items)


if __name__ == "__main__":
    main()
