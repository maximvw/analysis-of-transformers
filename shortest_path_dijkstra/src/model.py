"""GPT-style causal LM with Dijkstra-state auxiliary probes."""

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


class ShortestPathGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        max_seq_len: int = 160,
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


class DijkstraStateProbe(nn.Module):
    def __init__(self, n_head: int, d_head: int, n: int, max_distance: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.n = n
        self.max_distance = max_distance
        self.distance_classes = max_distance + 1
        self.dist_probes = nn.ModuleList(
            [nn.Linear(d_head, n * self.distance_classes) for _ in range(n_head)]
        )
        self.visited_probes = nn.ModuleList([nn.Linear(d_head, n * 2) for _ in range(n_head)])

    def forward(self, head_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_heads, num_steps, _ = head_outputs.shape
        dist_outputs = []
        visited_outputs = []
        for head_idx in range(num_heads):
            dist_logits = self.dist_probes[head_idx](head_outputs[:, head_idx])
            dist_logits = dist_logits.view(
                batch_size,
                num_steps,
                self.n,
                self.distance_classes,
            )
            visited_logits = self.visited_probes[head_idx](head_outputs[:, head_idx])
            visited_logits = visited_logits.view(batch_size, num_steps, self.n, 2)
            dist_outputs.append(dist_logits)
            visited_outputs.append(visited_logits)
        return torch.stack(dist_outputs, dim=1), torch.stack(visited_outputs, dim=1)


def compute_losses(
    model: ShortestPathGPT | nn.DataParallel,
    probe: DijkstraStateProbe | nn.DataParallel | None,
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
        "loss_dist_state": torch.tensor(0.0, device=device),
        "loss_visited_state": torch.tensor(0.0, device=device),
        "loss": loss_main,
        "token_correct": token_correct.detach(),
        "token_total": token_total.detach(),
    }

    if probe is not None and lambda_state > 0:
        dist_states = batch["dist_states"].to(device)
        visited_states = batch["visited_states"].to(device)
        probe_positions = batch["probe_positions"].to(device)

        gather_idx = probe_positions.unsqueeze(1).unsqueeze(-1).expand(
            -1,
            last_head_outputs.size(1),
            -1,
            last_head_outputs.size(-1),
        )
        step_head_outputs = last_head_outputs.gather(2, gather_idx)
        dist_logits, visited_logits = probe(step_head_outputs)

        probe_module = probe.module if isinstance(probe, nn.DataParallel) else probe
        if dist_states.max().item() > probe_module.max_distance:
            raise ValueError(
                "Distance target is out of range for probe: "
                f"max_distance={probe_module.max_distance}, "
                f"target_max={dist_states.max().item()}"
            )

        loss_dist = torch.tensor(0.0, device=device)
        loss_visited = torch.tensor(0.0, device=device)
        count = 0
        for head_idx in range(dist_logits.size(1)):
            loss_dist = loss_dist + F.cross_entropy(
                dist_logits[:, head_idx].reshape(-1, probe_module.distance_classes),
                dist_states.reshape(-1),
            )
            loss_visited = loss_visited + F.cross_entropy(
                visited_logits[:, head_idx].reshape(-1, 2),
                visited_states.reshape(-1),
            )
            count += 1
        loss_dist = loss_dist / max(count, 1)
        loss_visited = loss_visited / max(count, 1)
        loss_state = loss_dist + loss_visited

        result["loss_dist_state"] = loss_dist
        result["loss_visited_state"] = loss_visited
        result["loss_state"] = loss_state
        result["loss"] = loss_main + lambda_state * loss_state

    return result
