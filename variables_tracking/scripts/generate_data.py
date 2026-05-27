"""Generate datasets for the variable tracking experiment."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


VARIABLES = ["x", "y", "z", "w"]


def wrap(value: int, max_value: int) -> int:
    return value % max_value


def build_set(var: str, value: int) -> tuple[str, str, list[str], int | None]:
    return f"SET({var},{value})", var, [], value


def build_copy(dst: str, src: str) -> tuple[str, str, list[str], None]:
    return f"COPY({dst},{src})", dst, [src], None


def build_inc(var: str) -> tuple[str, str, list[str], None]:
    return f"INC({var})", var, [var], None


def build_dec(var: str) -> tuple[str, str, list[str], None]:
    return f"DEC({var})", var, [var], None


def build_add(dst: str, a: str, b: str) -> tuple[str, str, list[str], None]:
    return f"ADD({dst},{a},{b})", dst, [a, b], None


def build_addc(dst: str, src: str, value: int) -> tuple[str, str, list[str], int]:
    return f"ADDC({dst},{src},{value})", dst, [src], value


def execute_instruction(state: dict[str, int], token: str, max_value: int):
    if token.startswith("SET("):
        inside = token[4:-1]
        var, value = inside.split(",")
        state[var] = int(value) % max_value
        return
    if token.startswith("COPY("):
        inside = token[5:-1]
        dst, src = inside.split(",")
        state[dst] = state[src]
        return
    if token.startswith("INC("):
        var = token[4:-1]
        state[var] = (state[var] + 1) % max_value
        return
    if token.startswith("DEC("):
        var = token[4:-1]
        state[var] = (state[var] - 1) % max_value
        return
    if token.startswith("ADD("):
        inside = token[4:-1]
        dst, a, b = inside.split(",")
        state[dst] = (state[a] + state[b]) % max_value
        return
    if token.startswith("ADDC("):
        inside = token[5:-1]
        dst, src, value = inside.split(",")
        state[dst] = (state[src] + int(value)) % max_value
        return
    raise ValueError(f"Unknown token: {token}")


def sample_random_instruction(
    rng: random.Random,
    variables: list[str],
    max_value: int,
) -> tuple[str, str, list[str], int | None]:
    r = rng.random()
    if r < 0.20:
        return build_set(rng.choice(variables), rng.randrange(max_value))
    if r < 0.40:
        return build_copy(rng.choice(variables), rng.choice(variables))
    if r < 0.50:
        return build_inc(rng.choice(variables))
    if r < 0.60:
        return build_dec(rng.choice(variables))
    if r < 0.80:
        return build_add(rng.choice(variables), rng.choice(variables), rng.choice(variables))
    return build_addc(rng.choice(variables), rng.choice(variables), rng.randrange(max_value))


def generate_random_program(
    rng: random.Random,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
) -> dict:
    length = rng.randint(min_len, max_len)
    state = {var: 0 for var in variables}
    dep_depth = {var: 0 for var in variables}
    program = []
    states = []

    for _ in range(length):
        token, dst, parents, _ = sample_random_instruction(rng, variables, max_value)
        execute_instruction(state, token, max_value)
        if token.startswith("SET("):
            dep_depth[dst] = 1
        elif parents:
            dep_depth[dst] = 1 + max(dep_depth[parent] for parent in parents)
        else:
            dep_depth[dst] = 1
        program.append(token)
        states.append([state[var] for var in variables])

    query_var = rng.choice(variables)
    answer = state[query_var]
    dependency_depth = dep_depth[query_var]

    return {
        "program": program,
        "states": states,
        "query_var": query_var,
        "query_var_idx": variables.index(query_var),
        "answer": answer,
        "length": length,
        "dependency_depth": dependency_depth,
        "program_type": "random",
    }


def generate_chain_program(
    rng: random.Random,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
) -> dict:
    length = rng.randint(min_len, max_len)
    state = {var: 0 for var in variables}
    dep_depth = {var: 0 for var in variables}
    program = []
    states = []

    order = []
    current = rng.choice(variables)
    token, _, _, _ = build_set(current, rng.randrange(max_value))
    execute_instruction(state, token, max_value)
    dep_depth[current] = 1
    program.append(token)
    states.append([state[var] for var in variables])
    order.append(current)

    while len(program) < length:
        dst = rng.choice(variables)
        src = order[-1]
        if rng.random() < 0.5:
            token, _, _, _ = build_copy(dst, src)
            dep_depth[dst] = dep_depth[src] + 1
        else:
            token, _, _, _ = build_addc(dst, src, rng.randrange(max_value))
            dep_depth[dst] = dep_depth[src] + 1
        execute_instruction(state, token, max_value)
        program.append(token)
        states.append([state[var] for var in variables])
        order.append(dst)

    query_var = order[-1]
    answer = state[query_var]
    dependency_depth = dep_depth[query_var]
    return {
        "program": program,
        "states": states,
        "query_var": query_var,
        "query_var_idx": variables.index(query_var),
        "answer": answer,
        "length": length,
        "dependency_depth": dependency_depth,
        "program_type": "chain",
    }


def generate_distractor_program(
    rng: random.Random,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
) -> dict:
    length = rng.randint(min_len, max_len)
    anchor_var = rng.choice(variables)
    state = {var: 0 for var in variables}
    dep_depth = {var: 0 for var in variables}
    program = []
    states = []

    token, _, _, _ = build_set(anchor_var, rng.randrange(max_value))
    execute_instruction(state, token, max_value)
    dep_depth[anchor_var] = 1
    program.append(token)
    states.append([state[var] for var in variables])

    for step in range(1, length):
        if step < max(2, length // 3):
            if rng.random() < 0.5:
                token, _, _, _ = build_addc(anchor_var, anchor_var, rng.randrange(max_value))
            else:
                token, _, _, _ = build_inc(anchor_var)
            dep_depth[anchor_var] = dep_depth[anchor_var] + 1
        else:
            other_vars = [var for var in variables if var != anchor_var]
            token, dst, parents, _ = sample_random_instruction(rng, other_vars, max_value)
            if token.startswith("SET("):
                dep_depth[dst] = 1
            elif parents:
                dep_depth[dst] = 1 + max(dep_depth[parent] for parent in parents)
            else:
                dep_depth[dst] = 1
        execute_instruction(state, token, max_value)
        program.append(token)
        states.append([state[var] for var in variables])

    answer = state[anchor_var]
    return {
        "program": program,
        "states": states,
        "query_var": anchor_var,
        "query_var_idx": variables.index(anchor_var),
        "answer": answer,
        "length": length,
        "dependency_depth": dep_depth[anchor_var],
        "program_type": "distractor",
    }


def generate_copy_heavy_program(
    rng: random.Random,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
) -> dict:
    length = rng.randint(min_len, max_len)
    state = {var: 0 for var in variables}
    dep_depth = {var: 0 for var in variables}
    program = []
    states = []

    seed_var = rng.choice(variables)
    token, _, _, _ = build_set(seed_var, rng.randrange(max_value))
    execute_instruction(state, token, max_value)
    dep_depth[seed_var] = 1
    program.append(token)
    states.append([state[var] for var in variables])

    current_source = seed_var
    target_cycle = [var for var in variables if var != seed_var] or [seed_var]

    while len(program) < length:
        step_idx = len(program)
        dst = target_cycle[(step_idx - 1) % len(target_cycle)]
        r = rng.random()
        if r < 0.65:
            token, _, _, _ = build_copy(dst, current_source)
            dep_depth[dst] = dep_depth[current_source] + 1
            current_source = dst
        elif r < 0.85:
            token, _, _, _ = build_addc(dst, current_source, rng.randrange(max_value))
            dep_depth[dst] = dep_depth[current_source] + 1
            current_source = dst
        else:
            token, _, _, _ = build_set(dst, rng.randrange(max_value))
            dep_depth[dst] = 1
            current_source = dst
        execute_instruction(state, token, max_value)
        program.append(token)
        states.append([state[var] for var in variables])

    query_var = current_source
    answer = state[query_var]
    return {
        "program": program,
        "states": states,
        "query_var": query_var,
        "query_var_idx": variables.index(query_var),
        "answer": answer,
        "length": length,
        "dependency_depth": dep_depth[query_var],
        "program_type": "copy_heavy",
    }


def generate_arithmetic_heavy_program(
    rng: random.Random,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
) -> dict:
    length = rng.randint(min_len, max_len)
    state = {var: 0 for var in variables}
    dep_depth = {var: 0 for var in variables}
    program = []
    states = []

    for var in variables:
        if len(program) >= length:
            break
        token, _, _, _ = build_set(var, rng.randrange(max_value))
        execute_instruction(state, token, max_value)
        dep_depth[var] = 1
        program.append(token)
        states.append([state[name] for name in variables])

    while len(program) < length:
        dst = rng.choice(variables)
        if rng.random() < 0.55:
            a = rng.choice(variables)
            b = rng.choice(variables)
            token, _, parents, _ = build_add(dst, a, b)
        else:
            src = rng.choice(variables)
            token, _, parents, _ = build_addc(dst, src, rng.randrange(max_value))
        execute_instruction(state, token, max_value)
        dep_depth[dst] = 1 + max(dep_depth[parent] for parent in parents)
        program.append(token)
        states.append([state[name] for name in variables])

    query_var = max(variables, key=lambda var: (dep_depth[var], state[var]))
    answer = state[query_var]
    return {
        "program": program,
        "states": states,
        "query_var": query_var,
        "query_var_idx": variables.index(query_var),
        "answer": answer,
        "length": length,
        "dependency_depth": dep_depth[query_var],
        "program_type": "arithmetic_heavy",
    }


def generate_overwrite_heavy_program(
    rng: random.Random,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
) -> dict:
    length = rng.randint(min_len, max_len)
    state = {var: 0 for var in variables}
    dep_depth = {var: 0 for var in variables}
    program = []
    states = []

    target = rng.choice(variables)
    sources = [var for var in variables if var != target] or [target]

    for var in variables:
        if len(program) >= length:
            break
        token, _, _, _ = build_set(var, rng.randrange(max_value))
        execute_instruction(state, token, max_value)
        dep_depth[var] = 1
        program.append(token)
        states.append([state[name] for name in variables])

    while len(program) < length:
        if rng.random() < 0.35:
            src_var = rng.choice(sources)
            if rng.random() < 0.5:
                token, _, _, _ = build_inc(src_var)
                dep_depth[src_var] = dep_depth[src_var] + 1
            else:
                token, _, _, _ = build_addc(src_var, src_var, rng.randrange(max_value))
                dep_depth[src_var] = dep_depth[src_var] + 1
        else:
            r = rng.random()
            if r < 0.4:
                src = rng.choice(sources)
                token, _, parents, _ = build_copy(target, src)
            elif r < 0.75:
                src = rng.choice(sources)
                token, _, parents, _ = build_addc(target, src, rng.randrange(max_value))
            else:
                a = rng.choice(sources)
                b = rng.choice(variables)
                token, _, parents, _ = build_add(target, a, b)
            dep_depth[target] = 1 + max(dep_depth[parent] for parent in parents)
        execute_instruction(state, token, max_value)
        program.append(token)
        states.append([state[name] for name in variables])

    answer = state[target]
    return {
        "program": program,
        "states": states,
        "query_var": target,
        "query_var_idx": variables.index(target),
        "answer": answer,
        "length": length,
        "dependency_depth": dep_depth[target],
        "program_type": "overwrite_heavy",
    }


def generate_dataset(
    n_examples: int,
    min_len: int,
    max_len: int,
    variables: list[str],
    max_value: int,
    seed: int,
    mode: str = "random",
) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n_examples):
        if mode == "random":
            item = generate_random_program(rng, min_len, max_len, variables, max_value)
        elif mode == "chain":
            item = generate_chain_program(rng, min_len, max_len, variables, max_value)
        elif mode == "distractor":
            item = generate_distractor_program(rng, min_len, max_len, variables, max_value)
        elif mode == "copy_heavy":
            item = generate_copy_heavy_program(rng, min_len, max_len, variables, max_value)
        elif mode == "arithmetic_heavy":
            item = generate_arithmetic_heavy_program(rng, min_len, max_len, variables, max_value)
        elif mode == "overwrite_heavy":
            item = generate_overwrite_heavy_program(rng, min_len, max_len, variables, max_value)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        items.append(item)
    return items


def save_json(path: Path, data: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_summary(name: str, items: list[dict]):
    by_type = defaultdict(int)
    lengths = []
    depths = []
    for item in items:
        by_type[item["program_type"]] += 1
        lengths.append(item["length"])
        depths.append(item["dependency_depth"])
    print(
        f"{name}: n={len(items)} | "
        f"len=[{min(lengths)}, {max(lengths)}] | "
        f"avg_len={sum(lengths)/len(lengths):.2f} | "
        f"avg_depth={sum(depths)/len(depths):.2f} | "
        f"types={dict(by_type)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate variable tracking datasets")
    parser.add_argument("--output_dir", type=str, default="variables_tracking/data")
    parser.add_argument("--n_train", type=int, default=40000)
    parser.add_argument("--n_val", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--ood_min_len", type=int, default=24)
    parser.add_argument("--ood_max_len", type=int, default=40)
    parser.add_argument("--max_value", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    variables = list(VARIABLES)

    train = generate_dataset(
        args.n_train,
        args.min_len,
        args.max_len,
        variables,
        args.max_value,
        seed=args.seed,
        mode="random",
    )
    val = generate_dataset(
        args.n_val,
        args.min_len,
        args.max_len,
        variables,
        args.max_value,
        seed=args.seed + 1,
        mode="random",
    )
    test_id = generate_dataset(
        args.n_test,
        args.min_len,
        args.max_len,
        variables,
        args.max_value,
        seed=args.seed + 2,
        mode="random",
    )
    test_ood_long = generate_dataset(
        args.n_test,
        args.ood_min_len,
        args.ood_max_len,
        variables,
        args.max_value,
        seed=args.seed + 3,
        mode="overwrite_heavy",
    )
    test_ood_chain = generate_dataset(
        args.n_test,
        args.ood_min_len,
        args.ood_max_len,
        variables,
        args.max_value,
        seed=args.seed + 4,
        mode="chain",
    )
    test_ood_distractor = generate_dataset(
        args.n_test,
        args.ood_min_len,
        args.ood_max_len,
        variables,
        args.max_value,
        seed=args.seed + 5,
        mode="distractor",
    )
    test_ood_copy_heavy = generate_dataset(
        args.n_test,
        args.ood_min_len,
        args.ood_max_len,
        variables,
        args.max_value,
        seed=args.seed + 6,
        mode="copy_heavy",
    )
    test_ood_arithmetic = generate_dataset(
        args.n_test,
        args.ood_min_len,
        args.ood_max_len,
        variables,
        args.max_value,
        seed=args.seed + 7,
        mode="arithmetic_heavy",
    )

    save_json(output_dir / "train.json", train)
    save_json(output_dir / "val.json", val)
    save_json(output_dir / "test_id.json", test_id)
    save_json(output_dir / "test_ood_long.json", test_ood_long)
    save_json(output_dir / "test_ood_chain.json", test_ood_chain)
    save_json(output_dir / "test_ood_distractor.json", test_ood_distractor)
    save_json(output_dir / "test_ood_copy_heavy.json", test_ood_copy_heavy)
    save_json(output_dir / "test_ood_arithmetic.json", test_ood_arithmetic)

    print_summary("train", train)
    print_summary("val", val)
    print_summary("test_id", test_id)
    print_summary("test_ood_long", test_ood_long)
    print_summary("test_ood_chain", test_ood_chain)
    print_summary("test_ood_distractor", test_ood_distractor)
    print_summary("test_ood_copy_heavy", test_ood_copy_heavy)
    print_summary("test_ood_arithmetic", test_ood_arithmetic)


if __name__ == "__main__":
    main()
