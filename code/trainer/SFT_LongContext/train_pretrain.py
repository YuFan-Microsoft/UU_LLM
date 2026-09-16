import argparse
import math
import os
import sys
from datetime import datetime
from types import SimpleNamespace

# Make sibling local modules importable regardless of how this script is launched
# (deepspeed/torchrun launchers do NOT auto-add the script dir to sys.path).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pretrain_dataset import PretrainDataset, blending_datasets
from lm_model import LMModel
from pretrain_trainer import PretrainTrainer
from utils import get_strategy, get_tokenizer


def hierarchize(args):
    """Convert a flat argparse Namespace whose dest names contain dots
    (e.g. ``"muon.lr"``) into a nested SimpleNamespace so callers can write
    ``args.muon.lr`` instead of ``getattr(args, "muon.lr")``.  Keys without
    dots stay at the top level.
    """
    root = {}
    for k, v in vars(args).items():
        parts = k.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v

    def build(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: build(v) for k, v in x.items()})
        return x

    return build(root)


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    model = LMModel(
        args.model.model_name_or_path,
        attn_implementation=args.ds.attn_implementation,
        param_dtype=args.ds.param_dtype,  # default: bf16
        ds_config=strategy.get_ds_train_config(),
        packing_samples=args.ds.packing_samples,
        use_liger_kernel=args.ds.use_liger_kernel,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(
        args.model.model_name_or_path, model.model, "right", strategy, use_fast=not args.data.disable_fast_tokenizer
    )
    strategy.print(model)

    # gradient_checkpointing
    if args.model.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.model.gradient_checkpointing_reentrant}
        )

    # prepare for data and dataset
    train_data = blending_datasets(
        args.data.dataset,
        args.data.dataset_probs,
        strategy,
        args.train.seed,
        max_count=args.data.max_samples,
        dataset_split=args.data.dataset_split,
    )
    train_data = train_data.select(range(min(args.data.max_samples, len(train_data))))
    train_dataset = PretrainDataset(
        train_data,
        tokenizer,
        args.data.max_len,
        strategy,
        pretrain_mode=args.model.pretrain_mode_enable,
    )
    # prepare dataloader
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.train.micro_batch_size,
        True,
        True,
        train_dataset.collate_fn,
        num_workers=args.data.dataloader_num_workers,
    )

    eval_dataloader = None
    if getattr(args.eval, "dataset", None):
        eval_data = blending_datasets(
            args.eval.dataset,
            None,
            strategy,
            dataset_split=args.eval.split,
        )
        eval_dataset = PretrainDataset(
            eval_data,
            tokenizer,
            args.data.max_len,
            strategy,
            pretrain_mode=args.model.pretrain_mode_enable,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.train.micro_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
            num_workers=args.data.dataloader_num_workers,
        )

    # scheduler
    num_update_steps_per_epoch = len(train_dataset) // args.train.batch_size
    max_steps = math.ceil(args.train.max_epochs * num_update_steps_per_epoch)

    cfg = dict(
        optim=args.optim,
        muon=vars(args.muon),
        adam=vars(args.adam),
        lr_scheduler=args.lr_scheduler,
        lr_warmup_ratio=args.lr_warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        max_norm=args.max_norm,
        scheduler_steps=max_steps,
    )
    model, optim, scheduler = strategy.prepare((model, cfg))

    os.makedirs(args.ckpt.output_dir, exist_ok=True)

    # configure Trainer
    trainer = PretrainTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.model.pretrain_mode_enable,
        batch_size=args.train.batch_size,
        max_epochs=args.train.max_epochs,
        tokenizer=tokenizer,
    )

    trainer.fit(args, num_update_steps_per_epoch)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, tokenizer, args.ckpt.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--ckpt.output_dir", type=str, default="/yufan/projects/llm_training/checkpoints")
    parser.add_argument("--ckpt.saves_per_epoch", type=int, default=1,
                        help="How many checkpoints to save per epoch (>=1). "
                             "Last save of an epoch always coincides with the epoch boundary.")
    parser.add_argument("--logger.logging_steps", type=int, default=1)
    parser.add_argument("--eval.steps", type=int, default=-1)
    parser.add_argument("--ds.use_universal_ckpt", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--train.micro_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train.batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--model.gradient_checkpointing_enable", action="store_true", default=False)
    parser.add_argument("--ds.deepcompile", action="store_true", default=False)
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument(
        "--train.full_determinism_enable",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--ds.zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument(
        "--ds.param_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Model data type",
    )
    parser.add_argument("--ds.zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--ds.adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument(
        "--ds.attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation (e.g., eager, flash_attention_2, flash_attention_3, kernels-community/vllm-flash-attn3)",
    )
    parser.add_argument("--ds.use_liger_kernel", action="store_true", default=False, help="Enable Liger Kernel")
    parser.add_argument("--ds.grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--ds.overlap_comm", action="store_true", default=False)
    parser.add_argument("--model.gradient_checkpointing_reentrant", action="store_true", default=False)
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--data.dataloader_num_workers", type=int, default=0, help="Number of dataloader workers for IO"
    )
    parser.add_argument("--ds.tensor_parallel_size", type=int, default=1, help="DeepSpeed Tensor parallel size")

    # Training
    parser.add_argument("--train.max_epochs", type=int, default=2)
    parser.add_argument("--model.model_name_or_path", type=str, default=None)
    parser.add_argument("--model.pretrain_mode_enable", action="store_true", default=False, help="Use pretrain loss")

    # Optimizer + scheduler + grad clip.  Two sections:
    #   --muon.*  Muon-specific hypers (only used when --optim=muon)
    #   --adam.*  AdamW hypers — drives pure AdamW when --optim=adam,
    #             and Muon's aux-Adam subgroup when --optim=muon.
    # Note: DS v0.18.2 Muon ignores ns_steps / nesterov (hard-coded 5 / True) so
    # they are intentionally not exposed here.
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "muon"])
    # Muon-specific
    parser.add_argument("--muon.lr", type=float, default=0.02, help="LR for Muon 2D-weight group")
    parser.add_argument("--muon.momentum", type=float, default=0.95)
    # Placeholder slots: DS v0.18.x hard-codes ns_steps=5, nesterov=True inside
    # muon_update() and ignores these via config. Retained for forward-compat;
    # runtime warns when user sets a non-default value.
    parser.add_argument("--muon.ns_steps", type=int, default=5)
    parser.add_argument("--muon.nesterov", action="store_true", default=True)
    parser.add_argument("--muon.no_nesterov", dest="muon.nesterov", action="store_false")
    # AdamW (shared: pure-AdamW when --optim=adam, Muon's aux-Adam subgroup when --optim=muon)
    parser.add_argument("--adam.lr", type=float, default=5e-6)
    parser.add_argument("--adam.betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--adam.eps", type=float, default=1e-8)
    parser.add_argument("--adam.weight_decay", type=float, default=0.0)
    # Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    # Gradient clip
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")

    # ring-attention
    parser.add_argument("--ds.ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ds.ring_attn_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # packing samples without CrossAttention
    parser.add_argument("--ds.packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--data.dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument(
        "--data.dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets"
    )
    parser.add_argument("--eval.dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--data.dataset_split", type=str, default="train")
    parser.add_argument("--eval.split", type=str, default="train")
    parser.add_argument("--data.max_samples", type=int, default=1000000, help="Maximum number of samples to use")

    parser.add_argument("--data.input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--data.output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--data.max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--logger.wandb.key", type=str, default=None)
    parser.add_argument("--logger.wandb.org", type=str, default=None)
    parser.add_argument("--logger.wandb.group", type=str, default=None)
    parser.add_argument("--logger.wandb.project", type=str, default="openrlhf_train_pretrain")
    parser.add_argument(
        "--logger.wandb.run_name",
        type=str,
        default="pretrain_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    args = parser.parse_args()
    args = hierarchize(args)

    if args.ds.ring_attn_size > 1:
        assert args.ds.packing_samples, "packing_samples must be enabled when using ring attention"

    if args.ds.packing_samples and "flash_attention" not in args.ds.attn_implementation:
        print(
            "[Warning] Please use --attn_implementation with flash_attention to accelerate when --packing_samples is enabled."
        )
        args.ds.attn_implementation = "flash_attention_2"

    train(args)
