"""GPT-style model for variable tracking with auxiliary interpreter-state probe."""

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


class VariableTrackingGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_head = d_model // n_head

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        last_head_outputs = None
        for block in self.blocks:
            x, head_outputs = block(x, attn_mask)
            last_head_outputs = head_outputs

        hidden = self.ln_f(x)
        return hidden, last_head_outputs


class StateProbe(nn.Module):
    def __init__(self, n_head: int, d_head: int, num_vars: int, max_value: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.num_vars = num_vars
        self.max_value = max_value
        self.probes = nn.ModuleList(
            [nn.Linear(d_head, num_vars * max_value) for _ in range(n_head)]
        )

    def forward(self, head_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_steps, _ = head_outputs.shape
        outputs = []
        for head_idx in range(num_heads):
            logits = self.probes[head_idx](head_outputs[:, head_idx])
            logits = logits.view(batch_size, num_steps, self.num_vars, self.max_value)
            outputs.append(logits)
        return torch.stack(outputs, dim=1)


def compute_losses(
    model: VariableTrackingGPT | nn.DataParallel,
    probe: StateProbe | nn.DataParallel | None,
    batch: dict,
    lambda_state: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    targets = batch["targets"].to(device)
    ans_pos = batch["ans_pos"].to(device)
    attn_mask = input_ids != PAD_ID
    batch_size = input_ids.size(0)

    hidden, last_head_outputs = model(input_ids, attn_mask)
    answer_hidden = hidden[torch.arange(batch_size, device=device), ans_pos]
    classifier = model.module.classifier if isinstance(model, nn.DataParallel) else model.classifier
    answer_logits = classifier(answer_hidden)
    loss_main = F.cross_entropy(answer_logits, targets)

    result = {
        "loss_main": loss_main,
        "loss_state": torch.tensor(0.0, device=device),
        "answer_logits": answer_logits.detach(),
    }

    if probe is not None and lambda_state > 0:
        states = batch["states"].to(device)
        step_mask = batch["step_mask"].to(device)
        num_steps = states.size(1)

        step_positions = torch.arange(1, num_steps + 1, device=device)
        step_head_outputs = last_head_outputs[:, :, step_positions, :]
        probe_logits = probe(step_head_outputs)

        probe_module = probe.module if isinstance(probe, nn.DataParallel) else probe
        num_vars = probe_module.num_vars
        max_value = probe_module.max_value

        loss_state = torch.tensor(0.0, device=device)
        count = 0
        targets_flat = states.reshape(-1)
        step_mask_flat = step_mask.unsqueeze(-1).expand(-1, -1, num_vars).reshape(-1)

        for head_idx in range(probe_logits.size(1)):
            logits = probe_logits[:, head_idx].reshape(-1, max_value)
            valid_logits = logits[step_mask_flat]
            valid_targets = targets_flat[step_mask_flat]
            if valid_logits.numel() > 0:
                loss_state = loss_state + F.cross_entropy(valid_logits, valid_targets)
                count += 1

        if count > 0:
            loss_state = loss_state / count
        result["loss_state"] = loss_state

    result["loss"] = result["loss_main"] + lambda_state * result["loss_state"]
    return result

