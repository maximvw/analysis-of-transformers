"""Generate datasets for the string edit sequence experiment."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def compute_edit_script(source: str, target: str) -> tuple[list[str], list[list[int]]]:
    m = len(source)
    n = len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m:
                dp[i][j] = n - j
            elif j == n:
                dp[i][j] = m - i
            elif source[i] == target[j]:
                dp[i][j] = dp[i + 1][j + 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i + 1][j + 1],
                    dp[i + 1][j],
                    dp[i][j + 1],
                )

    script: list[str] = []
    cursor_states: list[list[int]] = []
    i = 0
    j = 0
    while i < m or j < n:
        cursor_states.append([i, j])
        if i < m and j < n and source[i] == target[j] and dp[i][j] == dp[i + 1][j + 1]:
            script.append(f"COPY_{source[i]}")
            i += 1
            j += 1
            continue
        if i < m and j < n and dp[i][j] == 1 + dp[i + 1][j + 1]:
            script.append(f"SUB_{target[j]}")
            i += 1
            j += 1
            continue
        if i < m and dp[i][j] == 1 + dp[i + 1][j]:
            script.append(f"DEL_{source[i]}")
            i += 1
            continue
        if j < n and dp[i][j] == 1 + dp[i][j + 1]:
            script.append(f"INS_{target[j]}")
            j += 1
            continue
        raise RuntimeError(f"Failed to decode script for {source!r} -> {target!r}")

    return script, cursor_states


def apply_script(source: str, script: list[str]) -> str:
    src_idx = 0
    output: list[str] = []
    for token in script:
        op, char = token.split("_", 1)
        if op == "COPY":
            assert src_idx < len(source) and source[src_idx] == char
            output.append(char)
            src_idx += 1
        elif op == "SUB":
            assert src_idx < len(source)
            output.append(char)
            src_idx += 1
        elif op == "DEL":
            assert src_idx < len(source) and source[src_idx] == char
            src_idx += 1
        elif op == "INS":
            output.append(char)
        else:
            raise ValueError(f"Unknown op token: {token}")
    assert src_idx == len(source)
    return "".join(output)


def sample_string(rng: random.Random, min_len: int, max_len: int, alphabet: list[str]) -> str:
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(length))


def sample_repetitive_string(
    rng: random.Random,
    min_len: int,
    max_len: int,
    alphabet: list[str],
) -> str:
    length = rng.randint(min_len, max_len)
    dominant = rng.choice(alphabet)
    chars = []
    for _ in range(length):
        if rng.random() < 0.75:
            chars.append(dominant)
        else:
            chars.append(rng.choice(alphabet))
    return "".join(chars)


def sample_blocky_string(
    rng: random.Random,
    min_len: int,
    max_len: int,
    alphabet: list[str],
) -> str:
    target_len = rng.randint(min_len, max_len)
    chars: list[str] = []
    while len(chars) < target_len:
        char = rng.choice(alphabet)
        remaining = target_len - len(chars)
        if remaining == 1:
            block_len = 1
        else:
            block_len = rng.randint(2, min(5, remaining))
        chars.extend(char for _ in range(block_len))
    return "".join(chars[:target_len])


def mutate_string_sparse(
    rng: random.Random,
    source: str,
    alphabet: list[str],
    min_len: int,
    max_len: int,
) -> str:
    target = list(source)
    n_edits = max(1, len(source) // 5)
    for _ in range(n_edits):
        allowed_actions = ["sub"]
        if len(target) < max_len:
            allowed_actions.append("ins")
        if len(target) > min_len:
            allowed_actions.append("del")
        action = rng.choice(allowed_actions)
        if action == "sub" and target:
            idx = rng.randrange(len(target))
            choices = [char for char in alphabet if char != target[idx]]
            target[idx] = rng.choice(choices or alphabet)
        elif action == "ins":
            idx = rng.randrange(len(target) + 1)
            target.insert(idx, rng.choice(alphabet))
        elif action == "del" and target:
            idx = rng.randrange(len(target))
            target.pop(idx)
    return "".join(target) or rng.choice(alphabet)


def mutate_string_dense(
    rng: random.Random,
    source: str,
    alphabet: list[str],
    min_len: int,
    max_len: int,
) -> str:
    target_len = rng.randint(min_len, max_len)
    target = []
    for idx in range(target_len):
        if idx < len(source) and rng.random() < 0.15:
            target.append(source[idx])
        else:
            target.append(rng.choice(alphabet))
    return "".join(target)


def build_example(source: str, target: str, split_type: str) -> dict:
    script, cursor_states = compute_edit_script(source, target)
    reconstructed = apply_script(source, script)
    if reconstructed != target:
        raise ValueError(f"Invalid script: {source!r} -> {target!r} reconstructed {reconstructed!r}")
    op_counts = Counter(token.split("_", 1)[0] for token in script)
    return {
        "source": source,
        "target": target,
        "script": script,
        "cursor_states": cursor_states,
        "source_length": len(source),
        "target_length": len(target),
        "edit_distance": len(script),
        "split_type": split_type,
        "op_counts": {key: op_counts.get(key, 0) for key in ("COPY", "SUB", "DEL", "INS")},
    }


def generate_dataset(
    rng: random.Random,
    n_examples: int,
    min_len: int,
    max_len: int,
    alphabet: list[str],
    split_type: str,
) -> list[dict]:
    return [
        build_example(
            sample_string(rng, min_len, max_len, alphabet),
            sample_string(rng, min_len, max_len, alphabet),
            split_type=split_type,
        )
        for _ in range(n_examples)
    ]


def generate_repetitive_dataset(
    rng: random.Random,
    n_examples: int,
    min_len: int,
    max_len: int,
    alphabet: list[str],
) -> list[dict]:
    return [
        build_example(
            sample_repetitive_string(rng, min_len, max_len, alphabet),
            sample_repetitive_string(rng, min_len, max_len, alphabet),
            split_type="repetitive",
        )
        for _ in range(n_examples)
    ]


def generate_blocky_dataset(
    rng: random.Random,
    n_examples: int,
    min_len: int,
    max_len: int,
    alphabet: list[str],
) -> list[dict]:
    return [
        build_example(
            sample_blocky_string(rng, min_len, max_len, alphabet),
            sample_blocky_string(rng, min_len, max_len, alphabet),
            split_type="blocky",
        )
        for _ in range(n_examples)
    ]


def generate_sparse_edits_dataset(
    rng: random.Random,
    n_examples: int,
    min_len: int,
    max_len: int,
    alphabet: list[str],
) -> list[dict]:
    items = []
    for _ in range(n_examples):
        source = sample_string(rng, min_len, max_len, alphabet)
        target = mutate_string_sparse(rng, source, alphabet, min_len, max_len)
        items.append(build_example(source, target, split_type="sparse_edits"))
    return items


def generate_dense_edits_dataset(
    rng: random.Random,
    n_examples: int,
    min_len: int,
    max_len: int,
    alphabet: list[str],
) -> list[dict]:
    items = []
    for _ in range(n_examples):
        source = sample_string(rng, min_len, max_len, alphabet)
        target = mutate_string_dense(rng, source, alphabet, min_len, max_len)
        items.append(build_example(source, target, split_type="dense_edits"))
    return items


def summarize(name: str, items: list[dict]):
    avg_src = sum(item["source_length"] for item in items) / max(len(items), 1)
    avg_tgt = sum(item["target_length"] for item in items) / max(len(items), 1)
    avg_edit = sum(item["edit_distance"] for item in items) / max(len(items), 1)
    op_counts = Counter()
    for item in items:
        op_counts.update(item["op_counts"])
    print(
        f"{name}: n={len(items)} | "
        f"src=[{min(item['source_length'] for item in items)}, {max(item['source_length'] for item in items)}] | "
        f"tgt=[{min(item['target_length'] for item in items)}, {max(item['target_length'] for item in items)}] | "
        f"avg_src={avg_src:.2f} avg_tgt={avg_tgt:.2f} avg_edit={avg_edit:.2f} | "
        f"ops={dict(op_counts)}"
    )


def write_json(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate string edit sequence data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=80_000)
    parser.add_argument("--n_val", type=int, default=1_000)
    parser.add_argument("--n_test", type=int, default=1_000)
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=15)
    parser.add_argument("--alphabet", type=str, default="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    alphabet = list(args.alphabet)

    splits = {
        "train": generate_dataset(rng, args.n_train, args.min_len, args.max_len, alphabet, "random"),
        "val": generate_dataset(rng, args.n_val, args.min_len, args.max_len, alphabet, "random"),
        "test_id": generate_dataset(rng, args.n_test, args.min_len, args.max_len, alphabet, "random"),
        "test_ood_repetitive": generate_repetitive_dataset(
            rng, args.n_test, args.min_len, args.max_len, alphabet
        ),
        "test_ood_blocky": generate_blocky_dataset(
            rng, args.n_test, args.min_len, args.max_len, alphabet
        ),
        "test_ood_sparse_edits": generate_sparse_edits_dataset(
            rng, args.n_test, args.min_len, args.max_len, alphabet
        ),
        "test_ood_dense_edits": generate_dense_edits_dataset(
            rng, args.n_test, args.min_len, args.max_len, alphabet
        ),
    }

    output_dir = Path(args.output_dir)
    for name, items in splits.items():
        write_json(output_dir / f"{name}.json", items)
        summarize(name, items)


if __name__ == "__main__":
    main()
