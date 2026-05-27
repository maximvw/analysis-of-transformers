"""Dataset: Python code snippets with per-token auxiliary state targets."""

from __future__ import annotations

import ast
from typing import Any

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from .state_tracking import compute_char_states


def _is_valid_python(code: str) -> bool:
    """Check that code parses without errors."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _align_char_states_to_tokens(
    source: str,
    encoding,
    char_states,
    max_bracket_depth: int = 20,
    max_indent_level: int = 15,
) -> dict[str, list[int]]:
    """Map char-level states to token-level using offset_mapping from tokenizer.

    Uses the first character of each token's span to determine its state.
    Falls back to sequential decode if offset_mapping is unavailable.
    """
    bracket = []
    indent = []
    mode = []

    n_chars = len(source)

    if hasattr(encoding, "offset_mapping") and encoding.offset_mapping is not None:
        for start, end in encoding.offset_mapping:
            if start < n_chars:
                bracket.append(min(char_states.bracket_depth[start], max_bracket_depth))
                indent.append(min(char_states.indent_level[start], max_indent_level))
                mode.append(char_states.lexical_mode[start])
            else:
                bracket.append(0)
                indent.append(0)
                mode.append(0)
    else:
        # Fallback: decode tokens one by one and track character offset
        char_offset = 0
        for tid in encoding.input_ids:
            if char_offset < n_chars:
                bracket.append(min(char_states.bracket_depth[char_offset], max_bracket_depth))
                indent.append(min(char_states.indent_level[char_offset], max_indent_level))
                mode.append(char_states.lexical_mode[char_offset])
            else:
                bracket.append(0)
                indent.append(0)
                mode.append(0)
            token_text = encoding.tokenizer.decode([tid]) if hasattr(encoding, "tokenizer") else ""
            char_offset += len(token_text)

    return {"bracket_depth": bracket, "indent_level": indent, "lexical_mode": mode}


class PythonCodeDataset(IterableDataset):
    """Streams Python code from codeparrot-clean, tokenizes, computes aux targets.

    Each yielded item is a dict with:
        input_ids:     LongTensor [seq_length]
        labels:        LongTensor [seq_length]  (shifted input_ids for NTP)
        bracket_depth: LongTensor [seq_length]
        indent_level:  LongTensor [seq_length]
        lexical_mode:  LongTensor [seq_length]
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int = 512,
        max_bracket_depth: int = 20,
        max_indent_level: int = 15,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 42,
        min_tokens: int = 64,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_bracket_depth = max_bracket_depth
        self.max_indent_level = max_indent_level
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.min_tokens = min_tokens

    def _load_stream(self):
        ds = load_dataset(
            "codeparrot/codeparrot-clean-valid",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        if self.max_samples is not None:
            ds = ds.take(self.max_samples)
        return ds

    def _process_example(self, code: str) -> dict[str, torch.Tensor] | None:
        """Tokenize code and compute auxiliary targets. Returns None if invalid."""
        if not _is_valid_python(code):
            return None

        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            max_length=self.seq_length + 1,  # +1 for shifting
            return_tensors=None,
            return_offsets_mapping=True,
        )

        ids = encoding["input_ids"]
        if len(ids) < self.min_tokens:
            return None

        # Compute char-level states
        char_states = compute_char_states(code)

        # Map to token level using offset_mapping
        offsets = encoding.get("offset_mapping")
        bracket = []
        indent = []
        mode = []
        n_chars = len(code)

        if offsets is not None:
            for start, end in offsets:
                if start < n_chars:
                    bracket.append(min(char_states.bracket_depth[start], self.max_bracket_depth))
                    indent.append(min(char_states.indent_level[start], self.max_indent_level))
                    mode.append(char_states.lexical_mode[start])
                else:
                    bracket.append(0)
                    indent.append(0)
                    mode.append(0)
        else:
            # Fallback: sequential decode
            char_offset = 0
            for tid in ids:
                if char_offset < n_chars:
                    bracket.append(min(char_states.bracket_depth[char_offset], self.max_bracket_depth))
                    indent.append(min(char_states.indent_level[char_offset], self.max_indent_level))
                    mode.append(char_states.lexical_mode[char_offset])
                else:
                    bracket.append(0)
                    indent.append(0)
                    mode.append(0)
                token_text = self.tokenizer.decode([tid])
                char_offset += len(token_text)

        # Truncate to seq_length (input) + create labels (shifted by 1)
        # input_ids = ids[:-1], labels = ids[1:]
        # aux targets align with input_ids (state at position t predicts state at t)
        seq_len = min(len(ids) - 1, self.seq_length)

        input_ids = torch.tensor(ids[:seq_len], dtype=torch.long)
        labels = torch.tensor(ids[1 : seq_len + 1], dtype=torch.long)
        bracket_t = torch.tensor(bracket[:seq_len], dtype=torch.long)
        indent_t = torch.tensor(indent[:seq_len], dtype=torch.long)
        mode_t = torch.tensor(mode[:seq_len], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "bracket_depth": bracket_t,
            "indent_level": indent_t,
            "lexical_mode": mode_t,
        }

    def __iter__(self):
        stream = self._load_stream()
        for example in stream:
            code = example["content"]
            item = self._process_example(code)
            if item is not None:
                yield item


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Pad batch to same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    labels = []
    bracket = []
    indent = []
    mode = []
    attention_mask = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        bracket.append(torch.cat([item["bracket_depth"], torch.zeros(pad_len, dtype=torch.long)]))
        indent.append(torch.cat([item["indent_level"], torch.zeros(pad_len, dtype=torch.long)]))
        mode.append(torch.cat([item["lexical_mode"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "bracket_depth": torch.stack(bracket),
        "indent_level": torch.stack(indent),
        "lexical_mode": torch.stack(mode),
    }
