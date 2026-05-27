"""Training loop for fixed-size shortest-path models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import ShortestPathDataset, collate_fn
from .model import DijkstraStateProbe, ShortestPathGPT, compute_losses
from .tokenizer import ShortestPathTokenizer


def train_epoch(
    model: ShortestPathGPT | torch.nn.DataParallel,
    probe: DijkstraStateProbe | torch.nn.DataParallel | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lambda_state: float,
    device: torch.device,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    if probe is not None:
        probe.train()

    total_loss = 0.0
    total_main = 0.0
    total_state = 0.0
    token_correct = 0
    token_total = 0
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
def eval_epoch(
    model: ShortestPathGPT | torch.nn.DataParallel,
    probe: DijkstraStateProbe | torch.nn.DataParallel | None,
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

    for batch in tqdm(loader, desc="Val", leave=False, dynamic_ncols=True):
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


def save_checkpoint(
    model: ShortestPathGPT | torch.nn.DataParallel,
    probe: DijkstraStateProbe | torch.nn.DataParallel | None,
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

    # Overwrite the same file every time. Kaggle keeps only one best model per run.
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


def load_training_state(
    checkpoint_path: str | Path,
    model: ShortestPathGPT | torch.nn.DataParallel,
    probe: DijkstraStateProbe | torch.nn.DataParallel | None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = ckpt["model_state_dict"]
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    if probe is not None and "probe_state_dict" in ckpt:
        probe_state = ckpt["probe_state_dict"]
        if isinstance(probe, torch.nn.DataParallel):
            probe.module.load_state_dict(probe_state)
        else:
            probe.load_state_dict(probe_state)
    elif probe is not None:
        raise ValueError(
            "Checkpoint does not contain probe_state_dict, but current run uses --lambda_state > 0"
        )

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt["epoch"], ckpt.get("metrics", {})


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = ShortestPathTokenizer(
        n=args.graph_n,
        max_weight=args.max_weight,
        max_distance=args.max_distance,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")

    train_ds = ShortestPathDataset(args.train_path, tokenizer)
    val_ds = ShortestPathDataset(args.val_path, tokenizer)
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

    model = ShortestPathGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    probe = None
    if args.lambda_state > 0:
        probe = DijkstraStateProbe(
            n_head=args.n_head,
            d_head=args.d_model // args.n_head,
            n=args.graph_n,
            max_distance=args.max_distance,
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
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    log = []
    start_epoch = 0

    if args.resume_from:
        resumed_epoch, resumed_metrics = load_training_state(
            args.resume_from, model, probe, optimizer, device
        )
        start_epoch = resumed_epoch + 1
        best_val_loss = resumed_metrics.get("loss", float("inf"))
        log_path = save_dir / "train_log.json"
        if log_path.exists():
            with open(log_path) as f:
                log = json.load(f)
            log = [entry for entry in log if entry.get("epoch", -1) <= resumed_epoch]
        print(
            f"Resumed from {args.resume_from} | "
            f"best_epoch={resumed_epoch} best_val_loss={best_val_loss:.4f} "
            f"next_epoch={start_epoch}"
        )

    for epoch in range(start_epoch, args.epochs):
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
        val_metrics = eval_epoch(model, probe, val_loader, args.lambda_state, device)

        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "time": time.time() - start_time,
        }
        log.append(entry)

        print(
            f"Epoch {epoch:3d} | "
            f"train loss={train_metrics['loss']:.4f} tok_acc={train_metrics['token_accuracy']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} tok_acc={val_metrics['token_accuracy']:.4f}"
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

    with open(save_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return log


def main():
    parser = argparse.ArgumentParser(description="Train fixed-size shortest-path model")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--graph_n", type=int, default=10)
    parser.add_argument("--max_weight", type=int, default=9)
    parser.add_argument("--max_distance", type=int, default=99)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_state", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints/shortest_path_dijkstra")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)


if __name__ == "__main__":
    main()
