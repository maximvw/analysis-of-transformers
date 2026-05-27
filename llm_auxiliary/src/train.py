"""Training loop for LLM fine-tuning with auxiliary state-tracking losses."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

from .data import PythonCodeDataset, collate_fn
from .model import CausalLMWithAuxProbes, compute_auxiliary_losses


@dataclass
class TrainConfig:
    # Model
    model_name: str = "EleutherAI/pythia-160m"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_8bit: bool = False

    # Data
    seq_length: int = 512
    max_train_samples: int | None = 50_000
    max_eval_samples: int | None = 2_000
    min_tokens: int = 64

    # Aux loss config
    lambda_bracket: float = 1.0
    lambda_indent: float = 1.0
    lambda_mode: float = 1.0
    max_bracket_depth: int = 20
    max_indent_level: int = 15

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    max_steps: int = 2000
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    fp16: bool = True

    # Logging
    log_every: int = 10
    eval_every: int = 200
    save_every: int = 500
    output_dir: str = "checkpoints/llm_aux"
    seed: int = 42


def train(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = CausalLMWithAuxProbes(
        model_name=config.model_name,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        load_in_8bit=config.load_in_8bit,
        max_bracket_depth=config.max_bracket_depth,
        max_indent_level=config.max_indent_level,
    ).to(device)
    model.print_trainable_parameters()

    # Datasets
    train_ds = PythonCodeDataset(
        tokenizer=tokenizer,
        seq_length=config.seq_length,
        max_bracket_depth=config.max_bracket_depth,
        max_indent_level=config.max_indent_level,
        split="train",
        max_samples=config.max_train_samples,
        seed=config.seed,
        min_tokens=config.min_tokens,
    )

    pad_id = tokenizer.pad_token_id or 0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_id),
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer: LoRA params + probe params
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.fp16)

    # Training loop
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    log = []
    model.train()
    optimizer.zero_grad()

    step = 0
    accum_loss_lm = 0.0
    accum_loss_bracket = 0.0
    accum_loss_indent = 0.0
    accum_loss_mode = 0.0
    accum_loss_total = 0.0
    accum_count = 0

    pbar = tqdm(total=config.max_steps, desc="Training")
    train_iter = iter(train_loader)

    while step < config.max_steps:
        # Get next batch (restart iterator if exhausted)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=config.fp16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss_lm = outputs["loss_lm"]

            # Auxiliary losses
            aux_losses = compute_auxiliary_losses(
                outputs,
                {k: batch[k].to(device) for k in ["bracket_depth", "indent_level", "lexical_mode"]},
                attention_mask,
            )

            loss_total = loss_lm
            if config.lambda_bracket > 0:
                loss_total = loss_total + config.lambda_bracket * aux_losses["loss_bracket"]
            if config.lambda_indent > 0:
                loss_total = loss_total + config.lambda_indent * aux_losses["loss_indent"]
            if config.lambda_mode > 0:
                loss_total = loss_total + config.lambda_mode * aux_losses["loss_mode"]

            loss_scaled = loss_total / config.gradient_accumulation_steps

        scaler.scale(loss_scaled).backward()

        # Accumulate metrics
        accum_loss_lm += loss_lm.item()
        accum_loss_bracket += aux_losses["loss_bracket"].item()
        accum_loss_indent += aux_losses["loss_indent"].item()
        accum_loss_mode += aux_losses["loss_mode"].item()
        accum_loss_total += loss_total.item()
        accum_count += 1

        # Gradient accumulation step
        if accum_count % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            pbar.update(1)

            # Logging
            if step % config.log_every == 0:
                n = config.gradient_accumulation_steps
                entry = {
                    "step": step,
                    "loss_total": accum_loss_total / n,
                    "loss_lm": accum_loss_lm / n,
                    "loss_bracket": accum_loss_bracket / n,
                    "loss_indent": accum_loss_indent / n,
                    "loss_mode": accum_loss_mode / n,
                    "lr": scheduler.get_last_lr()[0],
                }
                log.append(entry)
                pbar.set_postfix(
                    lm=f"{entry['loss_lm']:.3f}",
                    br=f"{entry['loss_bracket']:.3f}",
                    ind=f"{entry['loss_indent']:.3f}",
                    mode=f"{entry['loss_mode']:.3f}",
                )

            # Reset accumulators
            if accum_count % config.gradient_accumulation_steps == 0:
                accum_loss_lm = 0.0
                accum_loss_bracket = 0.0
                accum_loss_indent = 0.0
                accum_loss_mode = 0.0
                accum_loss_total = 0.0
                accum_count = 0

            # Save checkpoint
            if step % config.save_every == 0:
                ckpt_dir = output_dir / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.model.save_pretrained(ckpt_dir / "lora")
                torch.save(model.probes.state_dict(), ckpt_dir / "probes.pt")
                with open(output_dir / "train_log.json", "w") as f:
                    json.dump(log, f, indent=2)

    pbar.close()

    # Final save
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(final_dir / "lora")
    torch.save(model.probes.state_dict(), final_dir / "probes.pt")
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training complete. {step} steps. Saved to {output_dir}")
    return log
