"""Generate fixed-size weighted graph datasets for Dijkstra shortest paths."""

from __future__ import annotations

import argparse
import heapq
import json
import random
from collections import Counter
from pathlib import Path


INF = 10**9


def dijkstra_trace(
    matrix: list[list[int | None]],
    source: int,
    state_inf: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    n = len(matrix)
    dist = [INF] * n
    visited = [False] * n
    dist[source] = 0
    heap: list[tuple[int, int]] = [(0, source)]
    dist_states: list[list[int]] = []
    visited_states: list[list[int]] = []

    for _ in range(n):
        u = None
        while heap:
            _, candidate = heapq.heappop(heap)
            if not visited[candidate]:
                u = candidate
                break
        if u is None:
            for candidate in range(n):
                if not visited[candidate]:
                    u = candidate
                    break
        if u is None:
            break

        visited[u] = True
        for v, weight in enumerate(matrix[u]):
            if weight is None or visited[v]:
                continue
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

        dist_states.append([state_inf if value >= INF else value for value in dist])
        visited_states.append([1 if flag else 0 for flag in visited])

    final_dist = [state_inf if value >= INF else value for value in dist]
    return final_dist, dist_states, visited_states


def empty_matrix(n: int) -> list[list[int | None]]:
    matrix: list[list[int | None]] = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    return matrix


def build_example(
    matrix: list[list[int | None]],
    source: int,
    split_type: str,
    state_inf: int,
) -> dict:
    distances, dist_states, visited_states = dijkstra_trace(matrix, source, state_inf)
    edges = [
        [u, v, weight]
        for u, row in enumerate(matrix)
        for v, weight in enumerate(row)
        if u != v and weight is not None
    ]
    return {
        "n": len(matrix),
        "source": source,
        "matrix": matrix,
        "distances": distances,
        "dist_states": dist_states,
        "visited_states": visited_states,
        "num_edges": len(edges),
        "edges": edges,
        "split_type": split_type,
    }


def ensure_reachable_backbone(
    rng: random.Random,
    matrix: list[list[int | None]],
    max_weight: int,
):
    n = len(matrix)
    order = list(range(n))
    rng.shuffle(order)
    if order[0] != 0:
        zero_idx = order.index(0)
        order[0], order[zero_idx] = order[zero_idx], order[0]
    for u, v in zip(order, order[1:]):
        matrix[u][v] = rng.randint(1, max_weight)


def sample_random_graph(
    rng: random.Random,
    n: int,
    edge_prob: float,
    max_weight: int,
    state_inf: int,
    split_type: str = "random",
) -> dict:
    matrix = empty_matrix(n)
    ensure_reachable_backbone(rng, matrix, max_weight)
    for u in range(n):
        for v in range(n):
            if u == v or matrix[u][v] is not None:
                continue
            if rng.random() < edge_prob:
                matrix[u][v] = rng.randint(1, max_weight)
    return build_example(matrix, 0, split_type, state_inf)


def sample_dense_graph(
    rng: random.Random,
    n: int,
    edge_prob: float,
    max_weight: int,
    state_inf: int,
) -> dict:
    return sample_random_graph(rng, n, edge_prob, max_weight, state_inf, "dense")


def sample_long_chain(
    rng: random.Random,
    n: int,
    max_weight: int,
    state_inf: int,
) -> dict:
    matrix = empty_matrix(n)
    chain = list(range(n))
    rng.shuffle(chain)
    zero_idx = chain.index(0)
    chain[0], chain[zero_idx] = chain[zero_idx], chain[0]
    for i, (u, v) in enumerate(zip(chain, chain[1:])):
        matrix[u][v] = 1 + (i % min(max_weight, 3))
    for u in chain[:-1]:
        for v in chain[1:]:
            if u != v and rng.random() < 0.08 and matrix[u][v] is None:
                matrix[u][v] = rng.randint(max(2, max_weight - 2), max_weight)
    return build_example(matrix, 0, "long_chain", state_inf)


def sample_adversarial_direct_edges(
    rng: random.Random,
    n: int,
    max_weight: int,
    state_inf: int,
) -> dict:
    matrix = empty_matrix(n)
    path = list(range(n))
    rng.shuffle(path)
    zero_idx = path.index(0)
    path[0], path[zero_idx] = path[zero_idx], path[0]

    for u, v in zip(path, path[1:]):
        matrix[u][v] = 1

    # Misleading direct edges from the source are present but worse than the
    # multi-hop path. This attacks the "just read direct edge weight" shortcut.
    for depth, v in enumerate(path[2:], start=2):
        matrix[0][v] = min(max_weight, depth + rng.randint(2, max(2, max_weight // 2)))

    for i, u in enumerate(path):
        for v in path[i + 2 :]:
            if matrix[u][v] is None and rng.random() < 0.12:
                matrix[u][v] = rng.randint(3, max_weight)
    return build_example(matrix, 0, "adversarial_direct", state_inf)


def sample_equal_ties(
    rng: random.Random,
    n: int,
    max_weight: int,
    state_inf: int,
) -> dict:
    matrix = empty_matrix(n)
    midpoint = max(2, n // 2)
    for v in range(1, midpoint):
        matrix[0][v] = 2
    for u in range(1, midpoint):
        for v in range(midpoint, n):
            if rng.random() < 0.65:
                matrix[u][v] = 2
    ensure_reachable_backbone(rng, matrix, max_weight)
    return build_example(matrix, 0, "equal_ties", state_inf)


def sample_layered_bipartite(
    rng: random.Random,
    n: int,
    max_weight: int,
    state_inf: int,
) -> dict:
    matrix = empty_matrix(n)
    order = list(range(n))
    if order[0] != 0:
        order.remove(0)
        rng.shuffle(order)
        order = [0] + order
    else:
        tail = order[1:]
        rng.shuffle(tail)
        order = [0] + tail

    n_layers = min(4, max(3, n // 3))
    layer_sizes = [1] + [1] * (n_layers - 1)
    remaining = n - sum(layer_sizes)
    idx = 1
    while remaining > 0:
        layer_sizes[idx] += 1
        remaining -= 1
        idx = 1 + (idx % (n_layers - 1))

    layers: list[list[int]] = []
    cursor = 0
    for size in layer_sizes:
        layers.append(order[cursor : cursor + size])
        cursor += size

    for left, right in zip(layers, layers[1:]):
        for u in left:
            good_v = rng.choice(right)
            matrix[u][good_v] = 1
            for v in right:
                if matrix[u][v] is None:
                    matrix[u][v] = rng.randint(max(2, max_weight - 2), max_weight)

    for i, left in enumerate(layers[:-2]):
        for u in left:
            for right in layers[i + 2 :]:
                for v in right:
                    if matrix[u][v] is None and rng.random() < 0.20:
                        matrix[u][v] = rng.randint(max(3, max_weight - 1), max_weight)

    return build_example(matrix, 0, "layered_bipartite", state_inf)


def sample_near_complete_hidden_path(
    rng: random.Random,
    n: int,
    max_weight: int,
    state_inf: int,
) -> dict:
    matrix = empty_matrix(n)
    path = list(range(n))
    if path[0] != 0:
        path.remove(0)
        rng.shuffle(path)
        path = [0] + path
    else:
        tail = path[1:]
        rng.shuffle(tail)
        path = [0] + tail

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            matrix[u][v] = rng.randint(max(3, max_weight - 2), max_weight)

    for u, v in zip(path, path[1:]):
        matrix[u][v] = 1

    for depth, v in enumerate(path[1:], start=1):
        matrix[0][v] = min(matrix[0][v], min(max_weight, depth + 3))

    return build_example(matrix, 0, "near_complete_hidden_path", state_inf)


def generate_dataset(
    rng: random.Random,
    n_examples: int,
    n: int,
    edge_prob: float,
    max_weight: int,
    state_inf: int,
) -> list[dict]:
    return [sample_random_graph(rng, n, edge_prob, max_weight, state_inf) for _ in range(n_examples)]


def summarize(name: str, items: list[dict]):
    types = Counter(item["split_type"] for item in items)
    avg_edges = sum(item["num_edges"] for item in items) / max(len(items), 1)
    max_dist = max(max(item["distances"]) for item in items) if items else 0
    print(
        f"{name}: n={len(items)} | graph_N={items[0]['n'] if items else 'n/a'} | "
        f"avg_edges={avg_edges:.2f} max_dist={max_dist} | types={dict(types)}"
    )


def write_json(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate fixed-size Dijkstra shortest-path data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=80_000)
    parser.add_argument("--n_val", type=int, default=2_000)
    parser.add_argument("--n_test", type=int, default=2_000)
    parser.add_argument("--graph_n", type=int, default=10)
    parser.add_argument("--edge_prob", type=float, default=0.25)
    parser.add_argument("--dense_edge_prob", type=float, default=0.55)
    parser.add_argument("--max_weight", type=int, default=9)
    parser.add_argument("--max_distance", type=int, default=99)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    splits = {
        "train": generate_dataset(
            rng, args.n_train, args.graph_n, args.edge_prob, args.max_weight, args.max_distance
        ),
        "val": generate_dataset(
            rng, args.n_val, args.graph_n, args.edge_prob, args.max_weight, args.max_distance
        ),
        "test_id": generate_dataset(
            rng, args.n_test, args.graph_n, args.edge_prob, args.max_weight, args.max_distance
        ),
        "test_ood_dense": [
            sample_dense_graph(
                rng, args.graph_n, args.dense_edge_prob, args.max_weight, args.max_distance
            )
            for _ in range(args.n_test)
        ],
        "test_ood_long_chain": [
            sample_long_chain(rng, args.graph_n, args.max_weight, args.max_distance)
            for _ in range(args.n_test)
        ],
        "test_ood_adversarial_direct": [
            sample_adversarial_direct_edges(
                rng, args.graph_n, args.max_weight, args.max_distance
            )
            for _ in range(args.n_test)
        ],
        "test_ood_equal_ties": [
            sample_equal_ties(rng, args.graph_n, args.max_weight, args.max_distance)
            for _ in range(args.n_test)
        ],
        "test_ood_layered_bipartite": [
            sample_layered_bipartite(rng, args.graph_n, args.max_weight, args.max_distance)
            for _ in range(args.n_test)
        ],
        "test_ood_near_complete_hidden_path": [
            sample_near_complete_hidden_path(
                rng, args.graph_n, args.max_weight, args.max_distance
            )
            for _ in range(args.n_test)
        ],
    }

    output_dir = Path(args.output_dir)
    for name, items in splits.items():
        write_json(output_dir / f"{name}.json", items)
        summarize(name, items)


if __name__ == "__main__":
    main()
