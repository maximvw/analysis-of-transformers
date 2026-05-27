"""Compute syntactic state for each character in Python source code.

Three independent state signals:
1. Bracket depth  — nesting level of ()[]{}
2. Indentation level — block depth (leading spaces // 4)
3. Lexical mode — code / string / comment

All are computed in a single O(n) pass over the source text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from transformers import PreTrainedTokenizerBase


class LexicalMode(IntEnum):
    CODE = 0
    STRING = 1
    COMMENT = 2


@dataclass
class CharStates:
    """Per-character syntactic state arrays (length = len(source))."""
    bracket_depth: list[int]
    indent_level: list[int]
    lexical_mode: list[int]


# ---------------------------------------------------------------------------
# Core: per-character state computation
# ---------------------------------------------------------------------------

def compute_char_states(source: str) -> CharStates:
    """Compute bracket depth, indent level, and lexical mode per character.

    Handles Python single/double quotes, triple quotes, # comments, and
    nested brackets. Bracket counting is paused inside strings/comments.
    """
    n = len(source)
    bracket_depth = [0] * n
    indent_level = [0] * n
    lexical_mode = [0] * n

    # Pre-compute per-line indent level and assign to all chars on the line
    line_start = 0
    for line in source.split("\n"):
        leading = len(line) - len(line.lstrip())
        level = leading // 4
        line_end = line_start + len(line)
        for j in range(line_start, min(line_end, n)):
            indent_level[j] = level
        # Account for the \n character
        if line_end < n:
            indent_level[line_end] = level
        line_start = line_end + 1  # +1 for \n

    depth = 0
    mode = LexicalMode.CODE
    # Track which quote opened the current string: ", ', """, '''
    string_delim: str | None = None

    i = 0
    while i < n:
        ch = source[i]

        # --- Lexical mode transitions ---
        if mode == LexicalMode.COMMENT:
            # Comment lasts until end of line
            if ch == "\n":
                mode = LexicalMode.CODE
            lexical_mode[i] = LexicalMode.COMMENT
            bracket_depth[i] = depth
            i += 1
            continue

        if mode == LexicalMode.STRING:
            lexical_mode[i] = LexicalMode.STRING
            bracket_depth[i] = depth

            assert string_delim is not None
            delim_len = len(string_delim)

            # Check for escape
            if ch == "\\" and i + 1 < n:
                # Escaped char — skip next
                if i + 1 < n:
                    lexical_mode[i + 1] = LexicalMode.STRING
                    bracket_depth[i + 1] = depth
                i += 2
                continue

            # Check for closing delimiter
            if source[i : i + delim_len] == string_delim:
                for j in range(1, delim_len):
                    if i + j < n:
                        lexical_mode[i + j] = LexicalMode.STRING
                        bracket_depth[i + j] = depth
                i += delim_len
                mode = LexicalMode.CODE
                string_delim = None
                continue

            i += 1
            continue

        # --- mode == CODE ---
        # Check for triple-quote strings first
        if source[i : i + 3] in ('"""', "'''"):
            string_delim = source[i : i + 3]
            mode = LexicalMode.STRING
            for j in range(3):
                if i + j < n:
                    lexical_mode[i + j] = LexicalMode.STRING
                    bracket_depth[i + j] = depth
            i += 3
            continue

        # Single/double quote
        if ch in ('"', "'"):
            string_delim = ch
            mode = LexicalMode.STRING
            lexical_mode[i] = LexicalMode.STRING
            bracket_depth[i] = depth
            i += 1
            continue

        # Comment
        if ch == "#":
            mode = LexicalMode.COMMENT
            lexical_mode[i] = LexicalMode.COMMENT
            bracket_depth[i] = depth
            i += 1
            continue

        # Brackets
        if ch in ("(", "[", "{"):
            lexical_mode[i] = LexicalMode.CODE
            bracket_depth[i] = depth
            depth += 1
            i += 1
            continue

        if ch in (")", "]", "}"):
            depth = max(0, depth - 1)
            lexical_mode[i] = LexicalMode.CODE
            bracket_depth[i] = depth
            i += 1
            continue

        # Default: regular code character
        lexical_mode[i] = LexicalMode.CODE
        bracket_depth[i] = depth
        i += 1

    return CharStates(
        bracket_depth=bracket_depth,
        indent_level=indent_level,
        lexical_mode=lexical_mode,
    )


# ---------------------------------------------------------------------------
# Map char-level states to BPE token-level states
# ---------------------------------------------------------------------------

def char_states_to_token_states(
    source: str,
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
    max_bracket_depth: int = 20,
    max_indent_level: int = 15,
) -> dict[str, torch.Tensor]:
    """Convert per-char states to per-token states.

    For each BPE token we take the state at its **first** character.
    This is the state the model "sees" when it starts processing the token.

    Returns dict with:
        bracket_depth: LongTensor [T]  (clamped to [0, max_bracket_depth])
        indent_level:  LongTensor [T]  (clamped to [0, max_indent_level])
        lexical_mode:  LongTensor [T]  (0=code, 1=string, 2=comment)
    """
    char_states = compute_char_states(source)

    # Decode each token to figure out its character span
    token_bracket = []
    token_indent = []
    token_mode = []

    char_offset = 0
    for tid in token_ids:
        token_str = tokenizer.decode([tid])
        token_len = len(token_str)

        if char_offset < len(source):
            # Try to align: find where this token's text appears
            # BPE tokens should reconstruct the source sequentially
            idx = char_offset
            token_bracket.append(min(char_states.bracket_depth[idx], max_bracket_depth))
            token_indent.append(min(char_states.indent_level[idx], max_indent_level))
            token_mode.append(char_states.lexical_mode[idx])
        else:
            # Past end of source (padding, EOS, etc.)
            token_bracket.append(0)
            token_indent.append(0)
            token_mode.append(0)

        char_offset += token_len

    return {
        "bracket_depth": torch.tensor(token_bracket, dtype=torch.long),
        "indent_level": torch.tensor(token_indent, dtype=torch.long),
        "lexical_mode": torch.tensor(token_mode, dtype=torch.long),
    }


def batch_compute_token_states(
    sources: list[str],
    batch_token_ids: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    max_bracket_depth: int = 20,
    max_indent_level: int = 15,
) -> dict[str, list[torch.Tensor]]:
    """Compute token-level states for a batch of sources."""
    all_bracket = []
    all_indent = []
    all_mode = []

    for source, token_ids in zip(sources, batch_token_ids):
        states = char_states_to_token_states(
            source, token_ids, tokenizer,
            max_bracket_depth, max_indent_level,
        )
        all_bracket.append(states["bracket_depth"])
        all_indent.append(states["indent_level"])
        all_mode.append(states["lexical_mode"])

    return {
        "bracket_depth": all_bracket,
        "indent_level": all_indent,
        "lexical_mode": all_mode,
    }
