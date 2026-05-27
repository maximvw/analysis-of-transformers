"""HuggingFace causal LM + LoRA + linear auxiliary probe heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class AuxiliaryProbes(nn.Module):
    """Linear probes on hidden states for auxiliary state prediction.

    Each probe is a simple Linear(d_model, n_classes) applied to every token.
    """

    def __init__(
        self,
        d_model: int,
        max_bracket_depth: int = 20,
        max_indent_level: int = 15,
        n_lexical_modes: int = 3,
    ):
        super().__init__()
        self.bracket_probe = nn.Linear(d_model, max_bracket_depth + 1)
        self.indent_probe = nn.Linear(d_model, max_indent_level + 1)
        self.mode_probe = nn.Linear(d_model, n_lexical_modes)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [B, T, d_model]

        Returns:
            dict with logits for each probe:
                bracket_logits: [B, T, max_bracket_depth + 1]
                indent_logits:  [B, T, max_indent_level + 1]
                mode_logits:    [B, T, n_lexical_modes]
        """
        return {
            "bracket_logits": self.bracket_probe(hidden_states),
            "indent_logits": self.indent_probe(hidden_states),
            "mode_logits": self.mode_probe(hidden_states),
        }


class CausalLMWithAuxProbes(nn.Module):
    """Wraps a HuggingFace CausalLM with auxiliary probe heads.

    The probes attach to the **last hidden states** (before the LM head).
    Gradients from auxiliary losses flow through the base model.
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-160m",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        load_in_8bit: bool = False,
        max_bracket_depth: int = 20,
        max_indent_level: int = 15,
    ):
        super().__init__()

        # Load base model
        if load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": torch.cuda.current_device()},
            )
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model.enable_input_require_grads()
        self.model = get_peft_model(base_model, lora_config)

        # Determine hidden size
        config = self.model.config
        d_model = getattr(config, "hidden_size", None) or getattr(config, "d_model", 768)

        # Auxiliary probes
        self.probes = AuxiliaryProbes(
            d_model=d_model,
            max_bracket_depth=max_bracket_depth,
            max_indent_level=max_indent_level,
        )

        self.max_bracket_depth = max_bracket_depth
        self.max_indent_level = max_indent_level

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                loss_lm:         scalar, NTP cross-entropy (if labels given)
                bracket_logits:  [B, T, max_bracket+1]
                indent_logits:   [B, T, max_indent+1]
                mode_logits:     [B, T, 3]
                logits:          [B, T, vocab_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # Last hidden state (before LM head)
        hidden_states = outputs.hidden_states[-1]  # [B, T, d_model]

        # Auxiliary probe predictions
        probe_outputs = self.probes(hidden_states)

        result = {
            "logits": outputs.logits,
            "bracket_logits": probe_outputs["bracket_logits"],
            "indent_logits": probe_outputs["indent_logits"],
            "mode_logits": probe_outputs["mode_logits"],
        }

        if labels is not None:
            result["loss_lm"] = outputs.loss

        return result

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def print_trainable_parameters(self):
        """Print trainable params summary (LoRA + probes)."""
        lora_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        probe_trainable = sum(p.numel() for p in self.probes.parameters())
        total = sum(p.numel() for p in self.parameters())
        print(f"LoRA trainable: {lora_trainable:,}")
        print(f"Probe trainable: {probe_trainable:,}")
        print(f"Total params: {total:,}")
        print(f"Trainable %: {100 * (lora_trainable + probe_trainable) / total:.2f}%")


def compute_auxiliary_losses(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute masked CE loss for each auxiliary probe.

    Args:
        model_output: dict from CausalLMWithAuxProbes.forward()
        batch: dict with bracket_depth, indent_level, lexical_mode targets
        attention_mask: [B, T] mask for valid (non-pad) tokens

    Returns:
        dict with loss_bracket, loss_indent, loss_mode scalars
    """
    mask = attention_mask.bool()  # [B, T]

    losses = {}
    for key, logits_key, target_key in [
        ("loss_bracket", "bracket_logits", "bracket_depth"),
        ("loss_indent", "indent_logits", "indent_level"),
        ("loss_mode", "mode_logits", "lexical_mode"),
    ]:
        logits = model_output[logits_key]  # [B, T, C]
        targets = batch[target_key]        # [B, T]

        # Flatten and mask
        logits_flat = logits[mask]         # [N_valid, C]
        targets_flat = targets[mask]       # [N_valid]

        if logits_flat.size(0) > 0:
            losses[key] = F.cross_entropy(logits_flat, targets_flat)
        else:
            losses[key] = torch.tensor(0.0, device=logits.device)

    return losses
