"""Evaluation: generate Python code and measure syntactic correctness."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from .model import CausalLMWithAuxProbes


def check_syntax(code: str) -> bool:
    """Check if code is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def check_bracket_match(code: str) -> bool:
    """Check if all brackets are properly matched."""
    stack = []
    matching = {")": "(", "]": "[", "}": "{"}
    in_string = False
    string_delim = None
    i = 0

    while i < len(code):
        ch = code[i]

        # Handle string context (skip brackets inside strings)
        if not in_string:
            if code[i : i + 3] in ('"""', "'''"):
                in_string = True
                string_delim = code[i : i + 3]
                i += 3
                continue
            if ch in ('"', "'"):
                in_string = True
                string_delim = ch
                i += 1
                continue
            if ch == "#":
                # Skip to end of line
                while i < len(code) and code[i] != "\n":
                    i += 1
                continue
        else:
            if ch == "\\" and i + 1 < len(code):
                i += 2
                continue
            dl = len(string_delim)
            if code[i : i + dl] == string_delim:
                in_string = False
                string_delim = None
                i += dl
                continue
            i += 1
            continue

        if ch in ("(", "[", "{"):
            stack.append(ch)
        elif ch in (")", "]", "}"):
            if not stack or stack[-1] != matching[ch]:
                return False
            stack.pop()

        i += 1

    return len(stack) == 0


def check_indentation_consistency(code: str) -> bool:
    """Check if indentation uses consistent 4-space increments.

    Returns True if all non-empty lines have indentation that is a multiple of 4
    (or the file uses tabs consistently).
    """
    lines = code.split("\n")
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            continue
        leading = len(line) - len(stripped)
        # Allow 0 or multiples of 4 (or any tab-based indent)
        if "\t" not in line and leading % 4 != 0:
            return False
    return True


@torch.no_grad()
def evaluate_generation(
    model: CausalLMWithAuxProbes,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: torch.device | None = None,
) -> dict:
    """Generate code completions and evaluate syntactic quality.

    Args:
        model: trained model
        tokenizer: tokenizer
        prompts: list of code prefixes to complete
        max_new_tokens: max tokens to generate
        temperature: sampling temperature
        top_p: nucleus sampling threshold
        device: device

    Returns:
        dict with:
            syntax_valid_rate: fraction passing ast.parse()
            bracket_match_rate: fraction with matched brackets
            indent_consistent_rate: fraction with consistent indentation
            generated_codes: list of generated code strings
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    generated_codes = []
    syntax_valid = 0
    bracket_matched = 0
    indent_consistent = 0

    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        input_ids = inputs["input_ids"].to(device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_codes.append(generated)

        if check_syntax(generated):
            syntax_valid += 1
        if check_bracket_match(generated):
            bracket_matched += 1
        if check_indentation_consistency(generated):
            indent_consistent += 1

    n = len(prompts)
    return {
        "syntax_valid_rate": syntax_valid / n if n > 0 else 0.0,
        "bracket_match_rate": bracket_matched / n if n > 0 else 0.0,
        "indent_consistent_rate": indent_consistent / n if n > 0 else 0.0,
        "n_samples": n,
        "generated_codes": generated_codes,
    }


def get_eval_prompts(n: int = 200, seed: int = 42) -> list[str]:
    """Get code prompts for evaluation.

    Takes first few lines of Python snippets from codeparrot as prompts.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "codeparrot/codeparrot-clean-valid",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    prompts = []
    for example in ds:
        code = example["content"]
        try:
            ast.parse(code)
        except SyntaxError:
            continue

        # Take first 3-5 lines as prompt
        lines = code.split("\n")
        if len(lines) < 5:
            continue
        prompt = "\n".join(lines[:4])
        if len(prompt) < 20:
            continue
        prompts.append(prompt)
        if len(prompts) >= n:
            break

    return prompts


@torch.no_grad()
def evaluate_probe_accuracy(
    model: CausalLMWithAuxProbes,
    eval_loader,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate how accurately probes predict the auxiliary states.

    Returns accuracy for each probe head.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    correct_bracket = 0
    correct_indent = 0
    correct_mode = 0
    total = 0

    for batch in tqdm(eval_loader, desc="Probe eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        mask = attention_mask.bool()

        # Bracket
        pred_bracket = outputs["bracket_logits"].argmax(dim=-1)
        correct_bracket += (pred_bracket[mask] == batch["bracket_depth"].to(device)[mask]).sum().item()

        # Indent
        pred_indent = outputs["indent_logits"].argmax(dim=-1)
        correct_indent += (pred_indent[mask] == batch["indent_level"].to(device)[mask]).sum().item()

        # Mode
        pred_mode = outputs["mode_logits"].argmax(dim=-1)
        correct_mode += (pred_mode[mask] == batch["lexical_mode"].to(device)[mask]).sum().item()

        total += mask.sum().item()

    return {
        "bracket_accuracy": correct_bracket / total if total > 0 else 0.0,
        "indent_accuracy": correct_indent / total if total > 0 else 0.0,
        "mode_accuracy": correct_mode / total if total > 0 else 0.0,
        "total_tokens": total,
    }
