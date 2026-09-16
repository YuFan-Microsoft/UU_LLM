import math
import os
from abc import ABC

import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

from lm_model import masked_mean
from utils import DistributedSampler


class NLLLoss(nn.Module):
    """Negative log-likelihood averaged under the (response) loss mask."""

    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (
            masked_mean(-per_token_logps, loss_mask, dim=None)
            if self.token_level_loss
            else masked_mean(-per_token_logps, loss_mask, dim=-1).mean()
        )
        return loss


class PretrainTrainer(ABC):
    """
    Trainer for causal-LM pretraining / continued-pretraining.

    Args:
        model (torch.nn.Module): The model to be trained.
        strategy (Strategy): The training strategy to be applied.
        optim (Optimizer): The optimizer for model training.
        train_dataloader (DataLoader): The dataloader for the training dataset.
        eval_dataloader (DataLoader): The dataloader for the evaluation dataset.
        scheduler (Scheduler): The learning rate scheduler to adjust training rates.
        max_norm (float, defaults to 1): Maximum gradient norm for clipping to prevent exploding gradients.
        pretrain_mode (bool, defaults to False): Flag to indicate if the trainer is in pre-training mode.
        batch_size (int, defaults to 1): Batch size for training.
        max_epochs (int, defaults to 2): The maximum number of training epochs.
        tokenizer (Tokenizer, optional): The tokenizer for processing input data.
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args

        self.loss_fn = NLLLoss()

        # packing samples
        self.packing_samples = strategy.args.ds.packing_samples

        # wandb setting
        self._wandb = None
        if self.strategy.args.logger.wandb.key and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.logger.wandb.key)
            wandb.init(
                entity=strategy.args.logger.wandb.org,
                project=strategy.args.logger.wandb.project,
                group=strategy.args.logger.wandb.group,
                name=strategy.args.logger.wandb.run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args, num_update_steps_per_epoch=None):
        # Infer num_update_steps_per_epoch from dataloader if not provided
        if num_update_steps_per_epoch is None:
            num_update_steps_per_epoch = len(self.train_dataloader)
        if num_update_steps_per_epoch <= 0:
            raise ValueError(
                f"num_update_steps_per_epoch must be positive, got {num_update_steps_per_epoch}. "
                "Check that your dataset is not smaller than train_batch_size."
            )

        # get eval steps
        if args.eval.steps == -1:
            args.eval.steps = num_update_steps_per_epoch  # Evaluate once per epoch

        # how many checkpoint saves per epoch (default 1 = end of epoch only)
        saves_per_epoch = max(1, int(getattr(args.ckpt, "saves_per_epoch", 1)))
        # boundaries (1-indexed step counts within the epoch) at which to save
        save_step_marks = [
            max(1, (num_update_steps_per_epoch * (i + 1)) // saves_per_epoch)
            for i in range(saves_per_epoch)
        ]
        # ensure last mark == end of epoch and marks are unique+sorted
        save_step_marks = sorted(set(save_step_marks))
        save_step_marks[-1] = num_update_steps_per_epoch

        global_step = 0
        epoch_bar = tqdm(
            range(self.epochs),
            desc="Train epoch",
            disable=not self.strategy.is_rank_0(),
        )

        # Step-0 baseline: eval + checkpoint BEFORE any training updates so we
        # have a "pre-train" reference (sanity check for loss curves and an
        # easy fallback if training diverges).
        baseline_eval_loss = None
        if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
            baseline_eval_loss = self.evaluate(self.eval_dataloader, global_step)
        if baseline_eval_loss is not None and math.isfinite(baseline_eval_loss):
            ppl = math.exp(min(baseline_eval_loss, 20.0))
            baseline_tag = f"epoch_0_step_0_ppl_{ppl:.4f}"
        else:
            baseline_tag = "epoch_0_step_0"
        baseline_path = os.path.join(args.ckpt.output_dir, baseline_tag)
        self.strategy.save_model(self.model, self.tokenizer, baseline_path)
        if self.strategy.is_rank_0():
            print(f"Saved baseline checkpoint to {baseline_path}")

        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            # train
            self.model.train()
            device = next(self.model.parameters()).device
            last_eval_loss = None
            epoch_step = 0
            for inputs, attention_masks, loss_masks in self.train_dataloader:
                # NOTE: collate_fn already returns [B, seq_len] (zero_pad_sequences uses
                # torch.cat on dim 0). Do NOT .squeeze(1) here: when seq_len happens to
                # equal 1 for an entire micro-batch, squeeze(1) would collapse the tensor
                # to 1D and break forward (batch, seqlen = sequences.size()).
                inputs = inputs.to(device)
                attention_mask = attention_masks.to(device)
                loss_mask = loss_masks.to(device)
                per_token_log_probs, output = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_output=True,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                gpt_loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])
                loss = gpt_loss
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                global_step += 1
                epoch_step += 1
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": self.strategy.get_grad_norm(self.model),
                }
                logs_dict = self.strategy.all_reduce(logs_dict)
                # ppl from the DP-averaged loss; cap input to avoid overflow on early/divergent steps
                logs_dict["ppl"] = math.exp(min(logs_dict["gpt_loss"], 20.0))
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logging + eval
                if global_step % args.logger.logging_steps == 0 and self._wandb is not None and self.strategy.is_rank_0():
                    logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                    self._wandb.log(logs)
                if global_step % args.eval.steps == 0:
                    if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                        last_eval_loss = self.evaluate(self.eval_dataloader, global_step)

                # intra-epoch checkpoint saves
                if epoch_step in save_step_marks:
                    is_epoch_end = epoch_step == num_update_steps_per_epoch
                    eval_loss = last_eval_loss
                    if self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
                        # always run a fresh eval right before saving so ppl reflects this checkpoint
                        eval_loss = self.evaluate(self.eval_dataloader, global_step)
                        last_eval_loss = eval_loss

                    if eval_loss is not None and math.isfinite(eval_loss):
                        ppl = math.exp(min(eval_loss, 20.0))
                        tag = f"epoch_{epoch + 1}_step_{epoch_step}_ppl_{ppl:.4f}"
                    else:
                        tag = f"epoch_{epoch + 1}_step_{epoch_step}"
                    if is_epoch_end:
                        # keep the cleaner epoch-only suffix at the boundary
                        tag = (f"epoch_{epoch + 1}_ppl_{ppl:.4f}"
                               if eval_loss is not None and math.isfinite(eval_loss)
                               else f"epoch_{epoch + 1}")

                    save_path = os.path.join(args.ckpt.output_dir, tag)
                    self.strategy.save_model(self.model, self.tokenizer, save_path)
                    if self.strategy.is_rank_0():
                        print(f"Saved checkpoint to {save_path}")

            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            device = next(self.model.parameters()).device
            for inputs, attention_masks, loss_masks in eval_dataloader:
                inputs = inputs.to(device)
                attention_mask = attention_masks.to(device)
                loss_mask = loss_masks.to(device)
                per_token_log_probs = self.model(
                    inputs,
                    attention_mask=attention_mask,
                    return_logprobs=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                )

                loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            avg_loss = loss_sum / max(times, 1)
            # Reduce across all ranks so callers (ckpt naming, wandb) see the
            # global eval loss, not just this rank's data shard.
            avg_loss = self.strategy.all_reduce(avg_loss, op="mean")
            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
        self.model.train()  # reset model state
        return avg_loss
