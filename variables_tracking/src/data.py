"""Dataset and DataLoader for variable tracking."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, VariableTrackingTokenizer


class VariableTrackingDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer: VariableTrackingTokenizer,
        num_vars: int = 4,
    ):
        self.tokenizer = tokenizer
        self.num_vars = num_vars

        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        program = item["program"]
        query_var = item["query_var"]
        answer = item["answer"]
        states = item["states"]
        program_length = item["length"]

        input_ids = self.tokenizer.encode_program(program, query_var, answer)
        ans_pos = self.tokenizer.get_answer_position(program_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(answer, dtype=torch.long),
            "ans_pos": torch.tensor(ans_pos, dtype=torch.long),
            "program_length": torch.tensor(program_length, dtype=torch.long),
            "states": torch.tensor(states, dtype=torch.long),  # [L, num_vars]
            "dependency_depth": torch.tensor(item.get("dependency_depth", 0), dtype=torch.long),
            "query_var_idx": torch.tensor(item.get("query_var_idx", 0), dtype=torch.long),
            "program_type": item.get("program_type", "random"),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    max_steps = max(item["program_length"].item() for item in batch)
    num_vars = batch[0]["states"].size(1)

    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_seq_len), PAD_ID, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    ans_pos = torch.zeros(batch_size, dtype=torch.long)
    program_length = torch.zeros(batch_size, dtype=torch.long)
    states = torch.zeros(batch_size, max_steps, num_vars, dtype=torch.long)
    step_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    dependency_depth = torch.zeros(batch_size, dtype=torch.long)
    query_var_idx = torch.zeros(batch_size, dtype=torch.long)
    program_types: list[str] = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        steps = item["program_length"].item()
        input_ids[i, :seq_len] = item["input_ids"]
        targets[i] = item["targets"]
        ans_pos[i] = item["ans_pos"]
        program_length[i] = steps
        states[i, :steps] = item["states"]
        step_mask[i, :steps] = True
        dependency_depth[i] = item["dependency_depth"]
        query_var_idx[i] = item["query_var_idx"]
        program_types.append(item["program_type"])

    return {
        "input_ids": input_ids,
        "targets": targets,
        "ans_pos": ans_pos,
        "program_length": program_length,
        "states": states,
        "step_mask": step_mask,
        "dependency_depth": dependency_depth,
        "query_var_idx": query_var_idx,
        "program_types": program_types,
    }

