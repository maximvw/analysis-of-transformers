"""Evaluation utilities for string edit sequence models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import StringEditDataset, collate_fn
from .model import CursorStateProbe, StringEditGPT, compute_losses
from .tokenizer import END_ID, StringEditTokenizer


@torch.no_grad()
def evaluate_teacher_forced(
    model: StringEditGPT,
    probe: CursorStateProbe | None,
    loader: DataLoader,
    lambda_state: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    if probe is not None:
        probe.eval()

    total_loss = 0.0
    total_lm = 0.0
    total_state = 0.0
    token_correct = 0
    token_total = 0
    n_batches = 0

    for batch in loader:
        result = compute_losses(model, probe, batch, lambda_state, device)
        total_loss += result["loss"].item()
        total_lm += result["loss_lm"].item()
        total_state += result["loss_state"].item()
        token_correct += result["token_correct"].item()
        token_total += result["token_total"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_lm": total_lm / max(n_batches, 1),
        "loss_state": total_state / max(n_batches, 1),
        "token_accuracy": token_correct / max(token_total, 1),
    }


@torch.no_grad()
def greedy_decode(
    model: StringEditGPT,
    tokenizer: StringEditTokenizer,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    model.eval()
    generated = list(prompt_ids)
    model_module = model.module if isinstance(model, torch.nn.DataParallel) else model
    max_seq_len = getattr(model_module, "max_seq_len", model_module.pos_emb.num_embeddings)
    if len(prompt_ids) > max_seq_len:
        raise ValueError(
            f"Prompt length {len(prompt_ids)} exceeds model max_seq_len={max_seq_len}. "
            "Increase --max_seq_len or reduce input lengths."
        )
    available_new_tokens = max(0, max_seq_len - len(prompt_ids))
    decode_steps = min(max_new_tokens, available_new_tokens)

    for _ in range(decode_steps):
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
        _, _, logits = model(input_ids, attn_mask)
        next_token = logits[0, -1].argmax(dim=-1).item()
        generated.append(next_token)
        if next_token == END_ID:
            break

    script_ids = generated[len(prompt_ids) :]
    return tokenizer.decode_script(script_ids)


@torch.no_grad()
def evaluate_generation(
    model: StringEditGPT,
    tokenizer: StringEditTokenizer,
    dataset: StringEditDataset,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, float]:
    exact_matches = 0
    total = len(dataset)
    total_pred_tokens = 0
    total_gold_tokens = 0

    for item in dataset.data:
        prompt_ids = tokenizer.encode_prompt(item["source"], item["target"])
        predicted = greedy_decode(model, tokenizer, prompt_ids, device, max_new_tokens)
        gold = item["script"]
        if predicted == gold:
            exact_matches += 1
        total_pred_tokens += len(predicted)
        total_gold_tokens += len(gold)

    return {
        "exact_match": exact_matches / max(total, 1),
        "avg_pred_len": total_pred_tokens / max(total, 1),
        "avg_gold_len": total_gold_tokens / max(total, 1),
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    tokenizer: StringEditTokenizer,
    device: torch.device,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    max_seq_len: int = 128,
    max_cursor_pos: int = 33,
) -> tuple[StringEditGPT, CursorStateProbe | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = StringEditGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    probe = None
    if "probe_state_dict" in ckpt:
        probe = CursorStateProbe(
            n_head=n_head,
            d_head=d_model // n_head,
            max_cursor_pos=max_cursor_pos,
        ).to(device)
        probe.load_state_dict(ckpt["probe_state_dict"])

    return model, probe


def evaluate_all(
    checkpoint_path: str,
    data_dir: str,
    alphabet: list[str],
    batch_size: int = 128,
    lambda_state: float = 0.0,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    max_seq_len: int = 128,
    max_cursor_pos: int = 33,
    max_new_tokens: int = 80,
) -> dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = StringEditTokenizer(alphabet=alphabet)
    model, probe = load_checkpoint(
        checkpoint_path,
        tokenizer,
        device,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
        max_cursor_pos=max_cursor_pos,
    )

    data_dir = Path(data_dir)
    results = {}
    for test_file in sorted(data_dir.glob("test*.json")):
        dataset = StringEditDataset(test_file, tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        metrics = evaluate_teacher_forced(model, probe, loader, lambda_state, device)
        gen_metrics = evaluate_generation(model, tokenizer, dataset, device, max_new_tokens)
        results[test_file.stem] = {**metrics, **gen_metrics, "n_samples": len(dataset)}
        print(
            f"{test_file.stem}: exact_match={gen_metrics['exact_match']:.4f} "
            f"token_acc={metrics['token_accuracy']:.4f} loss={metrics['loss']:.4f} "
            f"(n={len(dataset)})"
        )
    return results
