"""Dataset and collation utilities for fixed-size shortest paths."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, ShortestPathTokenizer


class ShortestPathDataset(Dataset):
    def __init__(self, data_path: str | Path, tokenizer: ShortestPathTokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.data = json.load(f)
        self._validate(data_path)

    def _validate(self, data_path: str | Path):
        for idx, item in enumerate(self.data):
            n = item["n"]
            if n != self.tokenizer.n:
                raise ValueError(
                    f"{data_path}: example {idx} has n={n}, "
                    f"but tokenizer expects fixed n={self.tokenizer.n}"
                )
            if item["source"] >= n:
                raise ValueError(f"{data_path}: example {idx} has invalid source")
            if len(item["matrix"]) != n or any(len(row) != n for row in item["matrix"]):
                raise ValueError(f"{data_path}: example {idx} matrix is not {n}x{n}")
            if len(item["distances"]) != n:
                raise ValueError(f"{data_path}: example {idx} has wrong distance vector length")
            if len(item["dist_states"]) != n or len(item["visited_states"]) != n:
                raise ValueError(f"{data_path}: example {idx} has wrong number of Dijkstra states")
            for step in range(n):
                if len(item["dist_states"][step]) != n:
                    raise ValueError(f"{data_path}: example {idx} dist state {step} has wrong size")
                if len(item["visited_states"][step]) != n:
                    raise ValueError(
                        f"{data_path}: example {idx} visited state {step} has wrong size"
                    )
            for row in item["matrix"]:
                for weight in row:
                    if weight is not None and not (0 <= weight <= self.tokenizer.max_weight):
                        raise ValueError(f"{data_path}: example {idx} has out-of-range weight")
            max_seen_distance = max(
                max(item["distances"]),
                max(max(state) for state in item["dist_states"]),
            )
            if max_seen_distance > self.tokenizer.max_distance:
                raise ValueError(
                    f"{data_path}: example {idx} has distance {max_seen_distance}, "
                    f"but tokenizer max_distance={self.tokenizer.max_distance}"
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        input_ids = self.tokenizer.encode_example(
            item["source"],
            item["matrix"],
            item["distances"],
        )
        prompt_len = len(self.tokenizer.encode_prompt(item["source"], item["matrix"]))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "distances": torch.tensor(item["distances"], dtype=torch.long),
            "dist_states": torch.tensor(item["dist_states"], dtype=torch.long),
            "visited_states": torch.tensor(item["visited_states"], dtype=torch.long),
            "num_edges": torch.tensor(item["num_edges"], dtype=torch.long),
            "split_type": item.get("split_type", "random"),
        }


def collate_fn(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    n = batch[0]["distances"].size(0)

    input_ids = torch.full((batch_size, max_seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    prompt_len = torch.zeros(batch_size, dtype=torch.long)
    distances = torch.zeros(batch_size, n, dtype=torch.long)
    dist_states = torch.zeros(batch_size, n, n, dtype=torch.long)
    visited_states = torch.zeros(batch_size, n, n, dtype=torch.long)
    probe_positions = torch.zeros(batch_size, n, dtype=torch.long)
    num_edges = torch.zeros(batch_size, dtype=torch.long)
    split_types: list[str] = []

    for batch_idx, item in enumerate(batch):
        seq = item["input_ids"]
        seq_len = seq.size(0)
        current_prompt_len = item["prompt_len"].item()

        input_ids[batch_idx, :seq_len] = seq
        prompt_len[batch_idx] = current_prompt_len
        distances[batch_idx] = item["distances"]
        dist_states[batch_idx] = item["dist_states"]
        visited_states[batch_idx] = item["visited_states"]
        num_edges[batch_idx] = item["num_edges"]

        start = current_prompt_len - 1
        end = current_prompt_len + n
        labels[batch_idx, start:end] = seq[start + 1 : end + 1]
        probe_positions[batch_idx] = torch.arange(start, start + n)
        split_types.append(item["split_type"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_len,
        "distances": distances,
        "dist_states": dist_states,
        "visited_states": visited_states,
        "probe_positions": probe_positions,
        "num_edges": num_edges,
        "split_types": split_types,
    }

