"""Training loop for variable tracking."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import VariableTrackingDataset, collate_fn
from .model import StateProbe, VariableTrackingGPT, compute_losses
from .tokenizer import VariableTrackingTokenizer


def train_epoch(
    model: VariableTrackingGPT | torch.nn.DataParallel,
    probe: StateProbe | torch.nn.DataParallel | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_state: float,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    model.train()
    if probe is not None:
        probe.train()

    total_loss = 0.0
    total_main = 0.0
    total_state = 0.0
    correct = 0
    total = 0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False, dynamic_ncols=True):
        optimizer.zero_grad()
        result = compute_losses(model, probe, batch, lambda_state, device)
        result["loss"].backward()

        params = list(model.parameters())
        if probe is not None:
            params += list(probe.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

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
    }


@torch.no_grad()
def eval_epoch(
    model: VariableTrackingGPT | torch.nn.DataParallel,
    probe: StateProbe | torch.nn.DataParallel | None,
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

    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
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
    }


def save_checkpoint(
    model: VariableTrackingGPT | torch.nn.DataParallel,
    probe: StateProbe | torch.nn.DataParallel | None,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    state = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if probe is not None:
        probe_state = probe.module.state_dict() if isinstance(probe, torch.nn.DataParallel) else probe.state_dict()
        state["probe_state_dict"] = probe_state

    checkpoint_path = save_dir / "checkpoint_best.pt"
    torch.save(state, checkpoint_path)
    with open(save_dir / "best_checkpoint_info.json", "w") as f:
        json.dump(
            {
                "best_epoch": epoch,
                "metrics": metrics,
                "checkpoint_path": str(checkpoint_path),
            },
            f,
            indent=2,
        )


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    variables = args.variables.split(",")
    tokenizer = VariableTrackingTokenizer(variables=variables, max_value=args.max_value)
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_ds = VariableTrackingDataset(args.train_path, tokenizer, num_vars=len(variables))
    val_ds = VariableTrackingDataset(args.val_path, tokenizer, num_vars=len(variables))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = VariableTrackingGPT(
        vocab_size=tokenizer.vocab_size,
        num_classes=args.max_value,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    probe = None
    if args.lambda_state > 0:
        probe = StateProbe(
            n_head=args.n_head,
            d_head=args.d_model // args.n_head,
            num_vars=len(variables),
            max_value=args.max_value,
        ).to(device)
        print(f"Probe parameters: {sum(p.numel() for p in probe.parameters()):,}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        if probe is not None:
            probe = torch.nn.DataParallel(probe)

    params = list(model.parameters())
    if probe is not None:
        params += list(probe.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    best_val_loss = float("inf")
    patience_counter = 0
    log = []

    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = train_epoch(
            model,
            probe,
            train_loader,
            optimizer,
            args.lambda_state,
            device,
            args.max_grad_norm,
        )
        val_metrics = eval_epoch(
            model,
            probe,
            val_loader,
            args.lambda_state,
            device,
        )

        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": time.time() - start_time,
        }
        log.append(entry)

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(model, probe, optimizer, epoch, val_metrics, save_dir)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return log


def main():
    parser = argparse.ArgumentParser(description="Train variable tracking model")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--variables", type=str, default="x,y,z,w")
    parser.add_argument("--max_value", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_state", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/variables_tracking")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
