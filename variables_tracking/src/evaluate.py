"""Evaluation utilities for variable tracking models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import VariableTrackingDataset, collate_fn
from .model import StateProbe, VariableTrackingGPT, compute_losses
from .tokenizer import VariableTrackingTokenizer


@torch.no_grad()
def evaluate_dataset(
    model: VariableTrackingGPT,
    probe: StateProbe | None,
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
    correct = 0
    total = 0
    n_batches = 0

    for batch in loader:
        result = compute_losses(model, probe, batch, lambda_state, device)
        total_loss += result["loss"].item()
        total_main += result["loss_main"].item()
        total_state += result["loss_state"].item()
        preds = result["answer_logits"].argmax(dim=-1)
        correct += (preds == batch["targets"].to(device)).sum().item()
        total += batch["targets"].size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_main": total_main / max(n_batches, 1),
        "loss_state": total_state / max(n_batches, 1),
        "accuracy": correct / max(total, 1),
        "n_samples": total,
    }


def load_checkpoint(
    checkpoint_path: str | Path,
    tokenizer: VariableTrackingTokenizer,
    device: torch.device,
    max_value: int = 32,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    max_seq_len: int = 64,
) -> tuple[VariableTrackingGPT, StateProbe | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VariableTrackingGPT(
        vocab_size=tokenizer.vocab_size,
        num_classes=max_value,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    probe = None
    if "probe_state_dict" in ckpt:
        probe = StateProbe(
            n_head=n_head,
            d_head=d_model // n_head,
            num_vars=len(tokenizer.variables),
            max_value=max_value,
        ).to(device)
        probe.load_state_dict(ckpt["probe_state_dict"])

    return model, probe


def evaluate_all(
    checkpoint_path: str,
    data_dir: str,
    variables: list[str],
    max_value: int = 32,
    batch_size: int = 256,
    lambda_state: float = 0.0,
    d_model: int = 128,
    n_layer: int = 2,
    n_head: int = 4,
    max_seq_len: int = 64,
) -> dict[str, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = VariableTrackingTokenizer(variables=variables, max_value=max_value)
    model, probe = load_checkpoint(
        checkpoint_path,
        tokenizer,
        device,
        max_value=max_value,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        max_seq_len=max_seq_len,
    )

    data_dir = Path(data_dir)
    results = {}
    for test_file in sorted(data_dir.glob("test*.json")):
        ds = VariableTrackingDataset(test_file, tokenizer, num_vars=len(variables))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
        metrics = evaluate_dataset(model, probe, loader, lambda_state, device)
        results[test_file.stem] = metrics
        print(
            f"{test_file.stem}: accuracy={metrics['accuracy']:.4f} "
            f"loss={metrics['loss']:.4f} (n={metrics['n_samples']})"
        )
    return results

