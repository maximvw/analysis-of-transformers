"""Dataset and collation utilities for topological graph sorting."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, TopologicalSortTokenizer


class TopologicalSortDataset(Dataset):
    def __init__(self, data_path: str | Path, tokenizer: TopologicalSortTokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.data = json.load(f)
        self._validate(data_path)

    def _validate(self, data_path: str | Path):
        for idx, item in enumerate(self.data):
            n = item["n"]
            if n > self.tokenizer.max_n:
                raise ValueError(
                    f"{data_path}: example {idx} has n={n}, "
                    f"but tokenizer max_n={self.tokenizer.max_n}"
                )
            if len(item["order"]) != n:
                raise ValueError(f"{data_path}: example {idx} order length does not match n={n}")
            if sorted(item["order"]) != list(range(n)):
                raise ValueError(f"{data_path}: example {idx} order is not a permutation")
            if len(item["indegree_states"]) != n:
                raise ValueError(f"{data_path}: example {idx} has wrong number of in-degree states")
            for step, state in enumerate(item["indegree_states"]):
                if len(state) != n:
                    raise ValueError(
                        f"{data_path}: example {idx} state {step} length does not match n={n}"
                    )
                if any(value < -1 for value in state):
                    raise ValueError(f"{data_path}: example {idx} state {step} has invalid values")
            for edge in item["edges"]:
                if len(edge) != 2:
                    raise ValueError(f"{data_path}: example {idx} has malformed edge {edge!r}")
                u, v = edge
                if not (0 <= u < n and 0 <= v < n) or u == v:
                    raise ValueError(f"{data_path}: example {idx} has invalid edge {edge!r}")
            if item["order"] != _lexicographic_topological_order(n, item["edges"]):
                raise ValueError(
                    f"{data_path}: example {idx} order is not lexicographic topological order"
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        n = item["n"]
        edges = item["edges"]
        order = item["order"]
        input_ids = self.tokenizer.encode_example(n, edges, order)
        prompt_len = len(self.tokenizer.encode_prompt(n, edges))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "n": torch.tensor(n, dtype=torch.long),
            "order": torch.tensor(order, dtype=torch.long),
            "states": torch.tensor(item["indegree_states"], dtype=torch.long),
            "num_edges": torch.tensor(item["num_edges"], dtype=torch.long),
            "split_type": item.get("split_type", "random"),
        }


def _lexicographic_topological_order(n: int, edges: list[list[int]]) -> list[int]:
    import heapq

    adj = [[] for _ in range(n)]
    indegree = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    heap = [vertex for vertex in range(n) if indegree[vertex] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(heap, v)
    return order


def collate_fn(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    max_n = max(item["n"].item() for item in batch)

    input_ids = torch.full((batch_size, max_seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    prompt_len = torch.zeros(batch_size, dtype=torch.long)
    n_values = torch.zeros(batch_size, dtype=torch.long)
    orders = torch.full((batch_size, max_n), -100, dtype=torch.long)
    states = torch.full((batch_size, max_n, max_n), -100, dtype=torch.long)
    state_mask = torch.zeros(batch_size, max_n, max_n, dtype=torch.bool)
    step_mask = torch.zeros(batch_size, max_n, dtype=torch.bool)
    probe_positions = torch.zeros(batch_size, max_n, dtype=torch.long)
    num_edges = torch.zeros(batch_size, dtype=torch.long)
    split_types: list[str] = []

    for batch_idx, item in enumerate(batch):
        seq = item["input_ids"]
        seq_len = seq.size(0)
        current_prompt_len = item["prompt_len"].item()
        n = item["n"].item()

        input_ids[batch_idx, :seq_len] = seq
        prompt_len[batch_idx] = current_prompt_len
        n_values[batch_idx] = n
        orders[batch_idx, :n] = item["order"]
        states[batch_idx, :n, :n] = item["states"]
        state_mask[batch_idx, :n, :n] = item["states"] >= 0
        step_mask[batch_idx, :n] = True
        num_edges[batch_idx] = item["num_edges"]

        start = current_prompt_len - 1
        end = current_prompt_len + n
        labels[batch_idx, start:end] = seq[start + 1 : end + 1]
        probe_positions[batch_idx, :n] = torch.arange(start, start + n)
        split_types.append(item["split_type"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_len,
        "n": n_values,
        "orders": orders,
        "states": states,
        "state_mask": state_mask,
        "step_mask": step_mask,
        "probe_positions": probe_positions,
        "num_edges": num_edges,
        "split_types": split_types,
    }
