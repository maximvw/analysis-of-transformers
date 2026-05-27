"""Dataset and collation utilities for the string edit sequence experiment."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, StringEditTokenizer


class StringEditDataset(Dataset):
    def __init__(self, data_path: str | Path, tokenizer: StringEditTokenizer):
        self.tokenizer = tokenizer
        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        source = item["source"]
        target = item["target"]
        script = item["script"]
        cursor_states = item["cursor_states"]
        input_ids = self.tokenizer.encode_example(source, target, script)
        prompt_len = len(self.tokenizer.encode_prompt(source, target))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "num_ops": torch.tensor(len(script), dtype=torch.long),
            "cursor_states": torch.tensor(cursor_states, dtype=torch.long),
            "source": source,
            "target": target,
            "script": script,
            "edit_distance": torch.tensor(item["edit_distance"], dtype=torch.long),
            "split_type": item.get("split_type", "random"),
        }


def collate_fn(batch: list[dict]) -> dict:
    batch_size = len(batch)
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    max_steps = max(item["num_ops"].item() for item in batch)

    input_ids = torch.full((batch_size, max_seq_len), PAD_ID, dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    prompt_len = torch.zeros(batch_size, dtype=torch.long)
    num_ops = torch.zeros(batch_size, dtype=torch.long)
    cursor_states = torch.zeros(batch_size, max_steps, 2, dtype=torch.long)
    step_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    probe_positions = torch.zeros(batch_size, max_steps, dtype=torch.long)
    sources: list[str] = []
    targets: list[str] = []
    scripts: list[list[str]] = []
    split_types: list[str] = []

    for batch_idx, item in enumerate(batch):
        seq = item["input_ids"]
        seq_len = seq.size(0)
        current_prompt_len = item["prompt_len"].item()
        current_num_ops = item["num_ops"].item()
        input_ids[batch_idx, :seq_len] = seq
        prompt_len[batch_idx] = current_prompt_len
        num_ops[batch_idx] = current_num_ops

        start = current_prompt_len - 1
        end = current_prompt_len + current_num_ops
        labels[batch_idx, start:end] = seq[start + 1 : end + 1]

        if current_num_ops > 0:
            cursor_states[batch_idx, :current_num_ops] = item["cursor_states"]
            step_mask[batch_idx, :current_num_ops] = True
            probe_positions[batch_idx, :current_num_ops] = torch.arange(start, start + current_num_ops)

        sources.append(item["source"])
        targets.append(item["target"])
        scripts.append(item["script"])
        split_types.append(item["split_type"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "prompt_len": prompt_len,
        "num_ops": num_ops,
        "cursor_states": cursor_states,
        "step_mask": step_mask,
        "probe_positions": probe_positions,
        "sources": sources,
        "targets": targets,
        "scripts": scripts,
        "split_types": split_types,
    }
