"""Speculators MTP training entry point with a memory-efficient loss for long Qwen3.5 sequences.

Runs `python -m speculators.train` after replacing MTPDraftModel.forward. The upstream forward applies lm_head to every
position of the packed row and keeps [total_seq_len, vocab] logits per step for the backward pass (Qwen3.5 has a
248,320-token vocabulary, so a 32K row needs > 100 GB over 3 steps). This forward is otherwise identical (same
recursion, positions, causal mask and step weights) but scores only the supervised positions, in chunks whose logits
are recomputed in the backward pass. It also logs per-step top-1 accuracy (acc_step_k) and the share of positions
whose drafts 0..k are all correct (cond_acc_step_k), the offline proxy for vLLM's per-position acceptance.

Usage: torchrun --standalone --nproc_per_node N train_mtp.py <speculators.train arguments>
Env: MTP_LOSS_CHUNK (default 4096) rows per lm_head chunk.
"""

import inspect
import os
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import speculators.models.mtp.core as mtp_core

# Lines of the upstream forward this replacement mirrors (speculators commit pinned in requirements_speculators.txt).
UPSTREAM_MARKERS = (
    "logits = self.lm_head(mtp_output)",
    "step_targets = input_ids[:, step + 2 : step + 2 + valid_len]",
    "input_ids[:, step + 1 : step + 1 + valid_len]",
    "current_hidden = mtp_output",
)
LOSS_CHUNK = int(os.environ.get("MTP_LOSS_CHUNK", "4096"))


def check_upstream() -> None:
    source = open(inspect.getsourcefile(mtp_core), encoding="utf-8").read()
    missing = [marker for marker in UPSTREAM_MARKERS if marker not in source]
    if missing:
        raise RuntimeError(f"speculators MTPDraftModel.forward changed ({missing} not found); update train_mtp.py "
                           "or install the commit in requirements_speculators.txt")


def chunked_lm_head_ce(lm_head: nn.Module, hidden: torch.Tensor, targets: torch.Tensor):
    """Summed cross entropy and argmax of lm_head(hidden) without keeping [N, vocab] logits for the backward pass."""
    def score(chunk_hidden, chunk_targets):
        logits = lm_head(chunk_hidden).float()
        return nn.functional.cross_entropy(logits, chunk_targets, reduction="sum"), logits.argmax(-1)

    loss = hidden.new_zeros((), dtype=torch.float32)
    predictions = []
    for start in range(0, hidden.shape[0], LOSS_CHUNK):
        chunk_loss, chunk_predictions = checkpoint(
            score, hidden[start:start + LOSS_CHUNK], targets[start:start + LOSS_CHUNK], use_reentrant=False)
        loss = loss + chunk_loss
        predictions.append(chunk_predictions)
    return loss, torch.cat(predictions)


def forward(
    self,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    step_weights: list[float] | None = None,
    return_dict: bool = True,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> tuple:
    input_ids = input_ids.long()
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    num_steps = self.config.num_speculative_steps
    if step_weights is not None and len(step_weights) != num_steps:
        raise ValueError(f"step_weights has {len(step_weights)} entries but num_speculative_steps={num_steps}")
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    total_loss = torch.tensor(0.0, device=device)
    metrics: dict[str, torch.Tensor] = {}
    effective_steps = min(num_steps, max(0, seq_len - 2))
    valid_len = seq_len - effective_steps - 1
    if valid_len <= 0 or effective_steps == 0:
        metrics["loss_sum"] = total_loss.detach().clone()
        metrics["loss_total"] = torch.tensor(1.0, device=device)
        return [], total_loss, metrics

    step_pos_ids, rotary_step_pos_ids = mtp_core._prepare_mtp_position_ids(  # noqa: SLF001
        position_ids, valid_len, self.config.transformer_layer_config.model_type)
    causal_mask = mtp_core.create_causal_mask(
        config=self.config.transformer_layer_config,
        inputs_embeds=hidden_states[:, :valid_len],
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=step_pos_ids,
    )

    all_correct = torch.ones(batch_size, valid_len, dtype=torch.bool, device=device)
    current_hidden = hidden_states
    for step in range(effective_steps):
        step_hidden = current_hidden[:, :valid_len]
        step_embeds = self.embed_tokens(input_ids[:, step + 1:step + 1 + valid_len])
        step_pos_emb = self.rotary_emb(step_hidden, rotary_step_pos_ids)
        mtp_output = self.mtp_layers[0](
            hidden_states=step_hidden,
            token_embeddings=step_embeds,
            attention_mask=causal_mask,
            position_ids=step_pos_ids,
            position_embeddings=step_pos_emb,
        )

        step_targets = input_ids[:, step + 2:step + 2 + valid_len]
        supervised = (torch.ones_like(step_targets, dtype=torch.bool) if loss_mask is None
                      else loss_mask[:, step + 2:step + 2 + valid_len] != 0)
        targets = step_targets[supervised]
        count = targets.numel()
        weight = step_weights[step] if step_weights is not None else 1.0
        if count:
            loss_sum, predictions = chunked_lm_head_ce(self.lm_head, mtp_output[supervised], targets)
            correct = predictions == targets
        else:
            loss_sum = mtp_output.sum() * 0.0  # keeps the graph for an all-prompt row
            correct = targets.new_zeros(0, dtype=torch.bool)
        step_loss = weight * loss_sum / max(count, 1)
        total_loss = total_loss + step_loss

        step_correct = torch.ones_like(all_correct)
        step_correct[supervised] = correct
        all_correct = all_correct & step_correct
        metrics[f"loss_step_{step}"] = step_loss.detach().clone()
        metrics[f"acc_step_{step}_sum"] = correct.sum().float()
        metrics[f"acc_step_{step}_total"] = torch.tensor(float(count), device=device)
        metrics[f"cond_acc_step_{step}_sum"] = (all_correct & supervised).sum().float()
        metrics[f"cond_acc_step_{step}_total"] = torch.tensor(float(count), device=device)

        current_hidden = mtp_output

    metrics["loss_sum"] = total_loss.detach().clone()
    metrics["loss_total"] = torch.tensor(1.0, device=device)
    return [], total_loss, metrics


def main() -> None:
    check_upstream()
    mtp_core.MTPDraftModel.forward = forward

    from speculators.train.cli import main as train_main
    from speculators.train.config import TrainConfig

    train_main(TrainConfig.resolve())


if __name__ == "__main__":
    main()
