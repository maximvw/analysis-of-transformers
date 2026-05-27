"""Evaluation utilities for topological graph sorting models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import TopologicalSortDataset, collate_fn
from .model import InDegreeProbe, TopologicalSortGPT, compute_losses
from .tokenizer import END_ID, TopologicalSortTokenizer


@torch.no_grad()
def evaluate_teacher_forced(
    model: TopologicalSortGPT,
    probe: InDegreeProbe | None,
    loader: DataLoader,
    lambda_state: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if probe is not None:
        probe.eval()

    total_loss = 0.0
    total_main = 0.0
    total_state = 0.0
    token_correct = 0
    token_total = 0
    n_batches = 0

    for batch in loader:
        result = compute_losses(model, probe, batch, lambda_state, device)
        total_loss += result["loss"].item()
        total_main += result["loss_main"].item()
        total_state += result["loss_state"].item()
        token_correct += result["token_correct"].item()
        token_total += result["token_total"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_main": total_main / max(n_batches, 1),
        "loss_state": total_state / max(n_batches, 1),
        "token_accuracy": token_correct / max(token_total, 1),
    }


@torch.no_grad()
def greedy_decode(
    model: TopologicalSortGPT,
    tokenizer: TopologicalSortTokenizer,
    prompt_ids: list[int],
    n: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    generated = list(prompt_ids)
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    max_seq_len = getattr(model_module, "max_seq_len", model_module.pos_emb.num_embeddings)
    max_new_tokens = n + 1
    if len(prompt_ids) + max_new_tokens > max_seq_len:
        raise ValueError(
            f"Prompt plus fixed answer length exceeds model max_seq_len={max_seq_len}: "
            f"prompt={len(prompt_ids)}, answer={max_new_tokens}"
        )

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
        _, _, logits = model(input_ids, attn_mask)
        next_token = logits[0, -1].argmax(dim=-1).item()
        generated.append(next_token)
        if next_token == END_ID:
            break

    return tokenizer.decode_vertices(generated[len(prompt_ids) :])


@torch.no_grad()
def evaluate_generation(
    model: TopologicalSortGPT,
    tokenizer: TopologicalSortTokenizer,
    dataset: TopologicalSortDataset,
    device: torch.device,
) -> dict[str, float]:
    exact_matches = 0
    valid_permutations = 0
    valid_topological = 0
    total = len(dataset)
    total_pred_len = 0
    total_gold_len = 0

    for item in dataset.data:
        n = item["n"]
        prompt_ids = tokenizer.encode_prompt(n, item["edges"])
        predicted = greedy_decode(model, tokenizer, prompt_ids, n, device)
        gold = item["order"]
        if predicted == gold:
            exact_matches += 1
        if len(predicted) == n and sorted(predicted) == list(range(n)):
            valid_permutations += 1
            position = {vertex: idx for idx, vertex in enumerate(predicted)}
            if all(position[u] < position[v] for u, v in item["edges"]):
                valid_topological += 1
        total_pred_len += len(predicted)
        total_gold_len += len(gold)

    return {
        "exact_match": exact_matches / max(total, 1),
        "valid_permutation": valid_permutations / max(total, 1),
        "valid_topological": valid_topological / max(total, 1),
        "avg_pred_len": total_pred_len / max(total, 1),
        "avg_gold_len": total_gold_len / max(total, 1),
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    tokenizer: TopologicalSortTokenizer,
    device: torch.device,
    max_n: int = 18,
    max_indegree: int = 18,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    max_seq_len: int = 256,
) -> tuple[TopologicalSortGPT, InDegreeProbe | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = TopologicalSortGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    probe = None
    if "probe_state_dict" in ckpt:
        probe = InDegreeProbe(
            n_head=n_head,
            d_head=d_model // n_head,
            max_n=max_n,
            max_indegree=max_indegree,
        ).to(device)
        probe.load_state_dict(ckpt["probe_state_dict"])

    return model, probe


def evaluate_all(
    checkpoint_path: str,
    data_dir: str,
    batch_size: int = 128,
    lambda_state: float = 0.0,
    max_n: int = 18,
    max_indegree: int = 18,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    max_seq_len: int = 256,
) -> dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TopologicalSortTokenizer(max_n=max_n)
    model, probe = load_checkpoint(
        checkpoint_path,
        tokenizer,
        device,
        max_n=max_n,
        max_indegree=max_indegree,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
    )

    data_dir = Path(data_dir)
    results = {}
    for test_file in sorted(data_dir.glob("test*.json")):
        dataset = TopologicalSortDataset(test_file, tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        metrics = evaluate_teacher_forced(model, probe, loader, lambda_state, device)
        gen_metrics = evaluate_generation(model, tokenizer, dataset, device)
        results[test_file.stem] = {**metrics, **gen_metrics, "n_samples": len(dataset)}
        print(
            f"{test_file.stem}: exact_match={gen_metrics['exact_match']:.4f} "
            f"valid_perm={gen_metrics['valid_permutation']:.4f} "
            f"valid_topo={gen_metrics['valid_topological']:.4f} "
            f"token_acc={metrics['token_accuracy']:.4f} ntp_loss={metrics['loss_main']:.4f} "
            f"(n={len(dataset)})"
        )
    return results
