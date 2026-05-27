"""GPT-style causal LM with in-degree auxiliary probe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_ID


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attn_mask is not None:
            pad_mask = ~attn_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(pad_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_out = torch.matmul(attn_probs, v)
        head_outputs = attn_out
        out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, channels)
        out = self.out_proj(out)
        return out, head_outputs


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, head_outputs = self.attn(self.ln1(x), attn_mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, head_outputs


class TopologicalSortGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_head = d_model // n_head
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds model max_seq_len={self.max_seq_len}."
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        last_head_outputs = None
        for block in self.blocks:
            x, head_outputs = block(x, attn_mask)
            last_head_outputs = head_outputs

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        return hidden, last_head_outputs, logits


class InDegreeProbe(nn.Module):
    def __init__(self, n_head: int, d_head: int, max_n: int, max_indegree: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.max_n = max_n
        self.max_indegree = max_indegree
        self.probes = nn.ModuleList(
            [nn.Linear(d_head, max_n * max_indegree) for _ in range(n_head)]
        )

    def forward(self, head_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_steps, _ = head_outputs.shape
        outputs = []
        for head_idx in range(num_heads):
            logits = self.probes[head_idx](head_outputs[:, head_idx])
            logits = logits.view(batch_size, num_steps, self.max_n, self.max_indegree)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)


def compute_losses(
    model: TopologicalSortGPT | nn.DataParallel,
    probe: InDegreeProbe | nn.DataParallel | None,
    batch: dict,
    lambda_state: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attn_mask = input_ids != PAD_ID

    _, last_head_outputs, lm_logits = model(input_ids, attn_mask)
    vocab_size = lm_logits.size(-1)
    loss_main = F.cross_entropy(
        lm_logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

    supervised = labels != -100
    preds = lm_logits.argmax(dim=-1)
    token_correct = ((preds == labels) & supervised).sum()
    token_total = supervised.sum()

    result = {
        "loss_main": loss_main,
        "loss_state": torch.tensor(0.0, device=device),
        "loss": loss_main,
        "token_correct": token_correct.detach(),
        "token_total": token_total.detach(),
    }

    if probe is not None and lambda_state > 0:
        states = batch["states"].to(device)
        state_mask = batch["state_mask"].to(device)
        step_mask = batch["step_mask"].to(device)
        probe_positions = batch["probe_positions"].to(device)
        num_steps = states.size(1)

        if num_steps > 0:
            valid_probe_positions = probe_positions[step_mask]
            if valid_probe_positions.numel() > 0:
                min_probe_pos = valid_probe_positions.min().item()
                max_probe_pos = valid_probe_positions.max().item()
                seq_len = last_head_outputs.size(2)
                if min_probe_pos < 0 or max_probe_pos >= seq_len:
                    raise ValueError(
                        "Probe position is out of range for model outputs: "
                        f"seq_len={seq_len}, min_pos={min_probe_pos}, max_pos={max_probe_pos}"
                    )

            gather_idx = probe_positions.unsqueeze(1).unsqueeze(-1).expand(
                -1,
                last_head_outputs.size(1),
                -1,
                last_head_outputs.size(-1),
            )
            step_head_outputs = last_head_outputs.gather(2, gather_idx)
            probe_logits = probe(step_head_outputs)

            probe_module = probe.module if isinstance(probe, nn.DataParallel) else probe
            max_n = probe_module.max_n
            max_indegree = probe_module.max_indegree
            if states.size(2) > max_n:
                raise ValueError(
                    f"Batch contains n={states.size(2)}, but probe max_n={max_n}. "
                    "Increase --max_n or regenerate data."
                )
            if states[state_mask].numel() > 0 and states[state_mask].max().item() >= max_indegree:
                raise ValueError(
                    "In-degree target is out of range for probe: "
                    f"max_indegree={max_indegree}, target_max={states[state_mask].max().item()}"
                )

            target_pad = torch.zeros(
                states.size(0),
                states.size(1),
                max_n,
                dtype=states.dtype,
                device=device,
            )
            mask_pad = torch.zeros(
                states.size(0),
                states.size(1),
                max_n,
                dtype=torch.bool,
                device=device,
            )
            target_pad[:, :, : states.size(2)] = states.clamp_min(0)
            mask_pad[:, :, : state_mask.size(2)] = state_mask
            targets_flat = target_pad.reshape(-1)
            mask_flat = mask_pad.reshape(-1)

            loss_state = torch.tensor(0.0, device=device)
            count = 0
            for head_idx in range(probe_logits.size(1)):
                logits = probe_logits[:, head_idx].reshape(-1, max_indegree)
                valid_logits = logits[mask_flat]
                valid_targets = targets_flat[mask_flat]
                if valid_logits.numel() > 0:
                    loss_state = loss_state + F.cross_entropy(valid_logits, valid_targets)
                    count += 1

            if count > 0:
                loss_state = loss_state / count
            result["loss_state"] = loss_state
            result["loss"] = loss_main + lambda_state * loss_state

    return result
