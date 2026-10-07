import argparse
import os
import datetime
import json
import math
import re
import socket
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import deepspeed
import numpy as np
from safetensors import safe_open

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
)
from deepspeed.ops.adam import FusedAdam
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

QWEN3_5_MULTIMODAL_ARCH = "Qwen3_5ForConditionalGeneration"
MISNESTED_VISUAL_PREFIX = "model.language_model.visual."
HF_VISUAL_PREFIX = "model.visual."

def encode_sft_example(messages, tokenizer, max_seq_length):
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if isinstance(input_ids, dict):
        input_ids = input_ids["input_ids"]
    input_ids = np.asarray(input_ids, dtype=np.int64)
    labels = np.full_like(input_ids, -100)

    generation_prompt_ids = tokenizer.apply_chat_template(
        conversation=messages[:-1],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(generation_prompt_ids, dict):
        generation_prompt_ids = generation_prompt_ids["input_ids"]
    generation_prompt_ids = np.asarray(generation_prompt_ids, dtype=np.int64)

    response_start = len(generation_prompt_ids)
    if not np.array_equal(input_ids[:response_start], generation_prompt_ids):
        raise RuntimeError(
            "The training sequence prefix does not match the non-thinking "
            "generation prompt used for inference."
        )
    response_end = int(np.flatnonzero(input_ids == tokenizer.eos_token_id)[-1])
    labels[response_start:response_end + 1] = input_ids[response_start:response_end + 1]

    if max_seq_length and len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    attention_mask = np.ones_like(input_ids)
    return input_ids, labels, attention_mask

class LLMDataset(Dataset):
    """With config_names, every item also carries "config_index" (its dataset config), used by the
    per-task eval perplexity; pair it with ConfigIndexCollator."""

    def __init__(self, dataset, tokenizer, max_seq_len, data_slice, config_names=None) -> None:
        super().__init__()
        self.dataset = dataset[data_slice]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.config_names = config_names
        self.print = 0

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        row = self.dataset[idx]
        input_ids, labels, _ = encode_sft_example(row["messages"], self.tokenizer, self.max_seq_len)
        item = {"input_ids": input_ids.tolist(), "labels": labels.tolist()}
        if self.config_names:
            item["config_index"] = row["config_index"]
        return item


class ConfigIndexCollator:
    """Pads with the wrapped collator and passes "config_index" through as a [batch] tensor."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        config_index = [feature.pop("config_index") for feature in features] if "config_index" in features[0] else None
        batch = self.collator(features)
        if config_index is not None:
            batch["config_index"] = torch.tensor(config_index, dtype=torch.long)
        return batch


def mixture_quotas(sizes, alpha):
    """Rows per epoch for each config: total rows of one natural epoch split in proportion to n ** alpha
    (alpha=1 keeps the natural mix, alpha=0 gives every config the same share)."""
    total = sum(sizes)
    weights = [size ** alpha for size in sizes]
    return [max(1, int(round(total * weight / sum(weights)))) for weight in weights]


class MixedEpochLLMDataset(Dataset):
    """Train set that mixes several configs with per-epoch quotas (temperature sampling).

    Each config keeps one fixed shuffled order. Epoch e takes positions [e * quota, (e + 1) * quota) of that order,
    wrapping around, so a large config sees fresh rows every epoch until it is exhausted and a small config is
    repeated about quota / size times per epoch. The epoch's rows are then shuffled together. The epoch length is
    constant, so the LR schedule is unchanged. Call set_epoch(epoch) on every rank before each epoch.
    """

    def __init__(self, config_datasets, config_names, tokenizer, max_seq_len, alpha, seed) -> None:
        super().__init__()
        self.dataset = concatenate_datasets(config_datasets)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.sizes = [len(dataset) for dataset in config_datasets]
        self.quotas = mixture_quotas(self.sizes, alpha)
        self.offsets = np.cumsum([0] + self.sizes[:-1])
        rng = np.random.default_rng(seed)
        self.orders = [rng.permutation(size) for size in self.sizes]
        self.config_names = config_names
        self.set_epoch(0)

    def describe(self):
        return [
            {"config": name, "rows": size, "rows_per_epoch": quota, "passes_per_epoch": round(quota / size, 2)}
            for name, size, quota in zip(self.config_names, self.sizes, self.quotas)
        ]

    def set_epoch(self, epoch):
        parts = [
            offset + order[(epoch * quota + np.arange(quota)) % size]
            for offset, order, size, quota in zip(self.offsets, self.orders, self.sizes, self.quotas)
        ]
        indices = np.concatenate(parts)
        np.random.default_rng(self.seed + epoch + 1).shuffle(indices)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        messages = self.dataset[int(self.indices[idx])]["messages"]
        input_ids, labels, _ = encode_sft_example(messages, self.tokenizer, self.max_seq_len)
        return {"input_ids": input_ids.tolist(),
            "labels": labels.tolist()}


def create_dataset(dataset_name,
                   tokenizer,
                   max_seq_len,
                   dataset_configs=None,
                   hf_token=None,
                   shuffle_seed=42,
                   mixing_alpha=None):
    load_kwargs = {"token": hf_token} if hf_token else {}
    if dataset_configs:
        config_datasets = [
            load_dataset(dataset_name, config_name, **load_kwargs)
            for config_name in dataset_configs
        ]
        dataset = {
            split_name: concatenate_datasets([
                config_dataset[split_name]
                for config_dataset in config_datasets
            ]).shuffle(seed=shuffle_seed)
            for split_name in ("train",)
        }
        dataset["test"] = concatenate_datasets([
            config_dataset["test"].add_column("config_index", [index] * len(config_dataset["test"]))
            for index, config_dataset in enumerate(config_datasets)
        ]).shuffle(seed=shuffle_seed)
    else:
        dataset = load_dataset(dataset_name, **load_kwargs)
    if dataset_configs and mixing_alpha is not None:
        train_llm_dataset = MixedEpochLLMDataset(
            [config_dataset["train"] for config_dataset in config_datasets], list(dataset_configs),
            tokenizer, max_seq_len, mixing_alpha, shuffle_seed)
        if torch.distributed.get_rank() == 0:
            print(f"Train mixture (alpha={mixing_alpha}, {len(train_llm_dataset)} rows per epoch): "
                  f"{json.dumps(train_llm_dataset.describe(), indent=2)}", flush=True)
    else:
        train_llm_dataset = LLMDataset(dataset, tokenizer, max_seq_len, "train")
    test_llm_dataset = LLMDataset(dataset, tokenizer, max_seq_len, "test",
                                  list(dataset_configs) if dataset_configs else None)
    return train_llm_dataset, test_llm_dataset


def log_sft_template_sample(messages, tokenizer):
    messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
    ]
    full_text = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    generation_prompt_text = tokenizer.apply_chat_template(
        conversation=messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not full_text.startswith(generation_prompt_text):
        raise RuntimeError(
            "The logged training sequence does not start with the inference "
            "generation prompt."
        )

    input_ids, labels, _ = encode_sft_example(messages, tokenizer, None)
    supervised_indices = np.flatnonzero(labels != -100)
    response_start = int(supervised_indices[0])
    supervised_text = full_text[len(generation_prompt_text):]
    raw_messages = json.dumps(messages, ensure_ascii=False, indent=2)

    print("***** SFT template sample *****")
    print(f"Raw messages:\n{raw_messages}")
    print(f"Full training text (repr):\n{full_text!r}")
    print(f"Inference generation prompt (repr):\n{generation_prompt_text!r}")
    print(f"Supervised suffix (repr):\n{supervised_text!r}")
    print(f"Response start token: {response_start}")
    print(f"Total tokens: {len(input_ids)}")
    print("***** End SFT template sample *****", flush=True)

def get_train_ds_config(stage=3):
    # Communication tuning benchmarked in README "Training speed optimization log" (ZeRO-3 communication tuning).
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 1e9,
        "stage3_prefetch_bucket_size": 5e8,
        "memory_efficient_linear": False,
        "reduce_bucket_size": 5e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
    }
    return {
        "train_batch_size": -1,
        "train_micro_batch_size_per_gpu": -1,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0
    }

def _z3_params_to_fetch(param_list):
    return [
        param for param in param_list
        if hasattr(param, 'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def validate_full_multimodal_model(model):
    named_parameters = list(model.named_parameters())
    parameter_names = {name for name, _ in named_parameters}
    has_visual = any(
        name.startswith((HF_VISUAL_PREFIX, MISNESTED_VISUAL_PREFIX))
        for name in parameter_names
    )
    has_language_model = any(
        name.startswith("model.language_model.")
        and not name.startswith(MISNESTED_VISUAL_PREFIX)
        for name in parameter_names
    )
    if not has_visual or not has_language_model:
        raise RuntimeError(
            "Refusing to save an incomplete Qwen3.5 checkpoint: "
            f"has_visual={has_visual}, has_language_model={has_language_model}"
        )
    if not hasattr(model, "lm_head"):
        raise RuntimeError("Refusing to save a Qwen3.5 checkpoint without an LM head")
    frozen_parameters = [name for name, param in named_parameters if not param.requires_grad]
    if frozen_parameters:
        raise RuntimeError(
            "Full-model SFT requires every parameter to be trainable; frozen "
            f"parameters include: {frozen_parameters[:10]}"
        )


def normalize_checkpoint_state_dict(state_dict):
    normalized_state_dict = {}
    for name, value in state_dict.items():
        if name.startswith(MISNESTED_VISUAL_PREFIX):
            name = HF_VISUAL_PREFIX + name.removeprefix(MISNESTED_VISUAL_PREFIX)
        if name in normalized_state_dict:
            raise RuntimeError(f"Duplicate checkpoint tensor after key normalization: {name}")
        normalized_state_dict[name] = value
    return normalized_state_dict


def get_saved_tensor_names(save_dir):
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as file:
            index = json.load(file)
        return set(index["weight_map"])

    tensor_names = set()
    checkpoint_files = sorted(
        filename
        for filename in os.listdir(save_dir)
        if filename.endswith(".safetensors")
    )
    if not checkpoint_files:
        raise RuntimeError(f"No safetensors weights were written to {save_dir}")
    for filename in checkpoint_files:
        with safe_open(
            os.path.join(save_dir, filename), framework="pt", device="cpu"
        ) as reader:
            tensor_names.update(reader.keys())
    return tensor_names


def validate_saved_checkpoint(model, save_dir, expected_state_dict):
    saved_names = get_saved_tensor_names(save_dir)
    tied_weights = getattr(model, "_tied_weights_keys", {})
    allowed_missing = set(tied_weights) if isinstance(tied_weights, dict) else set()
    missing_names = set(expected_state_dict) - saved_names - allowed_missing
    malformed_names = {
        name for name in saved_names if name.startswith(MISNESTED_VISUAL_PREFIX)
    }
    has_visual = any(name.startswith(HF_VISUAL_PREFIX) for name in saved_names)
    if missing_names or malformed_names or not has_visual:
        raise RuntimeError(
            "Saved Qwen3.5 checkpoint failed verification: "
            f"missing={sorted(missing_names)[:20]}, "
            f"malformed={sorted(malformed_names)[:20]}, "
            f"has_visual={has_visual}"
        )
    print(
        f"Verified checkpoint: {len(saved_names)} tensors, all expected "
        "parameters saved with standard Qwen3.5 names."
    )


def save_zero_three_model(model, processor, save_dir):
    # Release parameters left INFLIGHT by ZeRO-3 prefetching; _z3_params_to_fetch skips them otherwise.
    if hasattr(model, "empty_partition_cache"):
        model.empty_partition_cache()
    model_to_save = model.module if hasattr(model, 'module') else model
    validate_full_multimodal_model(model_to_save)
    output_state_dict = {}
    for name, param in model_to_save.named_parameters():
        if hasattr(param, 'ds_id'):
            with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([param]), enabled=True):
                param_cpu = param.data.cpu()
        else:
            param_cpu = param.cpu()
        if torch.distributed.get_rank() == 0:
            output_state_dict[name] = param_cpu

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        output_state_dict = normalize_checkpoint_state_dict(output_state_dict)
        os.makedirs(save_dir, exist_ok=True)
        model_to_save.save_pretrained(
            save_dir,
            state_dict=output_state_dict,
            safe_serialization=True,
            save_original_format=False,
        )
        validate_saved_checkpoint(model_to_save, save_dir, output_state_dict)
        processor.save_pretrained(save_dir)
    del output_state_dict
    torch.distributed.barrier()

def save_hf_checkpoint(model, processor, save_dir, zero_stage):
    if zero_stage == 3:
        save_zero_three_model(model, processor, save_dir)
        return

    model_to_save = model.module if hasattr(model, 'module') else model
    validate_full_multimodal_model(model_to_save)
    if torch.distributed.get_rank() == 0:
        output_state_dict = normalize_checkpoint_state_dict(model_to_save.state_dict())
        os.makedirs(save_dir, exist_ok=True)
        model_to_save.save_pretrained(
            save_dir,
            state_dict=output_state_dict,
            safe_serialization=True,
            save_original_format=False,
        )
        validate_saved_checkpoint(model_to_save, save_dir, output_state_dict)
        processor.save_pretrained(save_dir)
    torch.distributed.barrier()

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def parse_args(argument_defaults=None):
    parser = argparse.ArgumentParser(description="Train and save a full Qwen3.5 multimodal model")

    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_project', type=str, default="user_profile_slm_qwen35_4B")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--do_eval', type=int, default=1)

    parser.add_argument('--dataset_name', type=str, default='yufan/UltraData-SFT-2605-Chinese')
    parser.add_argument('--dataset_configs', nargs='+', default=None)
    parser.add_argument('--hf_token', type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument('--dataset_shuffle_seed', type=int, default=42)
    parser.add_argument('--dataset_mixing_alpha', type=float, default=None,
                        help="Mix --dataset_configs with per-epoch quotas proportional to rows ** alpha "
                             "(e.g. 0.5); large configs rotate through fresh rows, small ones repeat. "
                             "Default: concatenate all configs")
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')

    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--num_train_steps', type=int, default=-1)
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    parser.add_argument('--checkpoint_steps', type=int, default=5000)
    parser.add_argument('--logging_steps', type=int, default=1)

    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=8192)
    parser.add_argument('--pad_to_max_seq_len', action=argparse.BooleanOptionalAction, default=True,
                        help="Right-pad every batch to exactly --max_seq_len (fixed shapes, worst-case memory)")
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2',
                        help="Attention for the full-attention layers, e.g. flash_attention_2 or sdpa")
    parser.add_argument('--require_linear_attention_kernels', action=argparse.BooleanOptionalAction, default=True,
                        help="Exit at startup unless flash-linear-attention and causal-conv1d are importable")
    parser.add_argument('--label_logits_only', action=argparse.BooleanOptionalAction, default=True,
                        help="Compute lm_head logits only at supervised positions (logits_to_keep) and the loss in "
                             "the trainer; --no-label_logits_only uses the model's full-sequence loss")
    parser.add_argument('--max_eval_steps', type=int, default=-1)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--zero_stage', type=int, default=3)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default='cosine', choices=['linear', 'cosine'])

    # Generation-based evaluation with a colocated vLLM engine (runs at step 0 and at every checkpoint).
    parser.add_argument('--rollout_eval', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--rollout_eval_configs', nargs='+', default=None,
                        help="Dataset configs to roll out (default: --dataset_configs)")
    parser.add_argument('--rollout_eval_split', type=str, default='test')
    parser.add_argument('--rollout_eval_samples', type=int, default=256,
                        help="Fixed number of examples per config (<=0 for the whole split)")
    parser.add_argument('--rollout_max_model_len', type=int, default=15360)
    parser.add_argument('--rollout_max_tokens', type=int, default=8192)
    parser.add_argument('--rollout_gpu_memory_utilization', type=float, default=0.3)
    parser.add_argument('--rollout_tensor_parallel_size', type=int, default=1)
    parser.add_argument('--rollout_max_num_seqs', type=int, default=64)
    parser.add_argument('--rollout_max_num_batched_tokens', type=int, default=8192)
    # Eager mode keeps no CUDA-graph memory pool, so a sleeping vLLM holds (almost) no GPU memory during training.
    parser.add_argument('--rollout_enforce_eager', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--rollout_temperature', type=float, default=0.0)
    parser.add_argument('--rollout_top_p', type=float, default=1.0)
    parser.add_argument('--rollout_top_k', type=int, default=-1)
    parser.add_argument('--rollout_repetition_penalty', type=float, default=1.0)

    parser.add_argument("--global_rank", type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    if argument_defaults:
        parser.set_defaults(**argument_defaults)
    args = parser.parse_args()
    return args

def save_checkpoint(args, model, processor, epoch, step, ppl):
    ppl = round(ppl, 4)
    tag = f"epoch_{epoch}_step_{step}_ppl_{ppl}"
    cur_save_path = os.path.join(args.output_dir, tag)
    if torch.distributed.get_rank() == 0 and not os.path.exists(cur_save_path):
        os.makedirs(cur_save_path, exist_ok=True)

    if torch.distributed.get_rank() == 0:
        print("Saving model checkpoint ...")
    save_hf_checkpoint(model, processor, cur_save_path, args.zero_stage)


def get_optimizer_grouped_parameters(model, weight_decay):
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight", "ln_f.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return [g for g in grouped if g["params"]]


def distributed_config(args):
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
        if (torch.cuda.device_count() > 1 and not torch.distributed.is_initialized()):
            deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=1800000), init_method=None)
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed(timeout=datetime.timedelta(seconds=1800000))

    args.global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    ds_config = get_train_ds_config(stage=args.zero_stage)
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    ds_config["train_batch_size"] = args.per_device_train_batch_size * world_size

    if torch.distributed.is_initialized():
        deepspeed.comm.barrier()
    return device, ds_config


def prepare_model(args):
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.architectures = [QWEN3_5_MULTIMODAL_ARCH]
    model.config.use_cache = False
    validate_full_multimodal_model(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
    return model, processor, optimizer


def supervised_logits(model, batch, label_logits_only=True):
    """(logits [N, vocab], targets [N], rows [N]) for the N supervised positions of the batch, in row-major
    order; rows[i] is the batch row of position i. See compute_loss for label_logits_only."""
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key not in ("labels", "config_index")}
    # Logit t predicts token t + 1, so keep position t when labels[t + 1] is supervised.
    keep = torch.zeros_like(labels, dtype=torch.bool)
    keep[:, :-1] = labels[:, 1:] != -100
    targets = labels[:, 1:][keep[:, :-1]]
    rows = keep.nonzero(as_tuple=True)[0]
    if not label_logits_only:
        return model(**inputs, use_cache=False).logits[keep].float(), targets, rows

    module = model.module if hasattr(model, "module") else model
    lm_head = module.lm_head

    def select_supervised_hidden_states(_, args):
        return (args[0][keep],) + tuple(args[1:])

    # lm_head sits outside the checkpointed decoder layers, so the hook is not needed for backward recompute.
    handle = lm_head.register_forward_pre_hook(select_supervised_hidden_states)
    try:
        # logits_to_keep=0 hands all positions to lm_head; the hook narrows them to [N, hidden].
        logits = model(**inputs, use_cache=False, logits_to_keep=0).logits.float()
    finally:
        handle.remove()
    return logits, targets, rows


def compute_loss(model, batch, label_logits_only=True):
    """Causal-LM loss, the mean over supervised tokens (same value as the model's built-in loss).

    With label_logits_only, lm_head runs only at positions whose next-token label is supervised
    (assistant answer) instead of all max_seq_len positions. Prompt and padding positions have label -100
    and never contribute to the loss, so skipping them saves the [seq_len, vocab=248k] logits memory and
    their lm_head compute.

    Positions are selected per row with a boolean mask in a forward pre-hook on lm_head, so the logits are
    [total supervised tokens in the batch, vocab]. (`logits_to_keep` takes one index shared by all rows;
    the union of answer spans made logits grow with batch_size x sum of answer lengths.)
    """
    if not label_logits_only:
        return model(**batch, use_cache=False).loss

    logits, targets, _ = supervised_logits(model, batch, label_logits_only)
    if targets.numel() == 0:
        # Every rank must still run the forward (ZeRO-3 gathers collectively); contribute a zero loss.
        return logits.sum() * 0.0
    return F.cross_entropy(logits, targets)


def _ppl(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def evaluation(model, eval_dataloader, device, max_eval_steps=-1, label_logits_only=True):
    """Evaluate eval loss / perplexity and return a dict.

    batch_*: mean of per-batch losses, then mean over ranks (each batch and rank weighted equally).
    token_*: total NLL over all supervised tokens on all ranks / total supervised tokens (standard perplexity,
             independent of batch size, rank count, and batch composition).
    configs: {dataset config: {token_loss, token_ppl, tokens}}, the token-level numbers per task, when the eval
             set carries config_index.
    """
    model.eval()
    config_names = getattr(eval_dataloader.dataset, "config_names", None) or []
    batch_losses = torch.zeros((), device=device, dtype=torch.float64)
    config_nll = torch.zeros(len(config_names) + 1, device=device, dtype=torch.float64)
    config_tokens = torch.zeros(len(config_names) + 1, device=device, dtype=torch.float64)
    evaluated_steps = 0
    progress_bar = tqdm(
        eval_dataloader,
        desc="Evaluating",
        disable=torch.distributed.get_rank() != 0,
        dynamic_ncols=True,
    )
    for step, batch in enumerate(progress_bar):
        if max_eval_steps > 0 and step >= max_eval_steps:
            break
        batch = to_device(batch, device)
        with torch.no_grad():
            logits, targets, rows = supervised_logits(model, batch, label_logits_only)
            token_nll = F.cross_entropy(logits, targets, reduction="none").double()
        # Rows without a config (no config_index) go to the last slot, which only feeds the overall numbers.
        row_config = batch.get("config_index")
        if row_config is None:
            row_config = torch.full((batch["labels"].shape[0],), len(config_names), device=device, dtype=torch.long)
        config_nll.index_add_(0, row_config[rows], token_nll)
        config_tokens.index_add_(0, row_config[rows], torch.ones_like(token_nll))
        batch_losses += token_nll.mean() if token_nll.numel() else 0.0
        evaluated_steps += 1
    if evaluated_steps == 0:
        raise RuntimeError("Evaluation dataloader produced no batches")
    batch_loss = batch_losses / evaluated_steps
    try:
        batch_loss = get_all_reduce_mean(batch_loss)
        torch.distributed.all_reduce(config_nll, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(config_tokens, op=torch.distributed.ReduceOp.SUM)
    except:
        pass
    token_count = config_tokens.sum().item()
    token_loss = config_nll.sum().item() / max(token_count, 1)
    batch_loss = batch_loss.item()
    configs = {}
    for index, name in enumerate(config_names):
        tokens = config_tokens[index].item()
        if tokens:
            loss = config_nll[index].item() / tokens
            configs[name] = {"token_loss": loss, "token_ppl": _ppl(loss), "tokens": int(tokens)}
    if configs and torch.distributed.get_rank() == 0:
        for name, values in configs.items():
            print(f"  {task_name(name):<24} token ppl {values['token_ppl']:.4f} | token loss "
                  f"{values['token_loss']:.4f} | tokens {values['tokens']}", flush=True)
    model.train()
    return {
        "batch_loss": batch_loss,
        "batch_ppl": _ppl(batch_loss),
        "token_loss": token_loss,
        "token_ppl": _ppl(token_loss),
        "configs": configs,
    }


def eval_log(result: dict, wandb_module) -> dict:
    """wandb keys for an evaluation() result: overall numbers under Eval/, each task's token_ppl under
    <Task>_Evaluation/ (the same section as that task's rollout metrics)."""
    log = {f"Eval/{key}": value for key, value in result.items() if key != "configs"}
    for config, values in result.get("configs", {}).items():
        section = task_wandb_section(config)
        wandb_module.define_metric(f"{section}/*", step_metric="eval_step")
        log[f"{section}/token_ppl"] = values["token_ppl"]
    return log


def create_rollout_evaluator(args, tokenizer, rollout_scorer):
    """Build the colocated vLLM engine and the fixed rollout eval set. Must run before deepspeed.initialize."""
    if rollout_scorer is None:
        raise ValueError("--rollout_eval needs a rollout scorer (use deepspeed_user_profile_trainer.py)")
    from vllm_colocate_rollout import ColocatedVLLMRollout, load_rollout_eval_examples

    configs = args.rollout_eval_configs or args.dataset_configs or [None]
    examples = load_rollout_eval_examples(
        args.dataset_name,
        configs,
        args.rollout_eval_split,
        args.rollout_eval_samples,
        args.dataset_shuffle_seed,
        args.hf_token,
    )
    rollout = ColocatedVLLMRollout(
        args.model_name_or_path,
        tensor_parallel_size=args.rollout_tensor_parallel_size,
        gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        max_model_len=args.rollout_max_model_len,
        max_num_seqs=args.rollout_max_num_seqs,
        max_num_batched_tokens=args.rollout_max_num_batched_tokens,
        enforce_eager=args.rollout_enforce_eager,
        architectures=[QWEN3_5_MULTIMODAL_ARCH],
        trust_remote_code=True,
    )
    if torch.distributed.get_rank() == 0:
        print(f"Colocated vLLM rollout ready: {len(examples)} eval examples from {configs}", flush=True)
    return rollout, examples


def rollout_evaluation(args, rollout, examples, rollout_scorer, model, tokenizer, step, wandb_module=None):
    from vllm_colocate_rollout import run_rollout_evaluation

    if torch.distributed.get_rank() == 0:
        print(f"***** Rollout evaluation with vLLM ({len(examples)} examples) *****", flush=True)
    sampling_kwargs = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "repetition_penalty": args.rollout_repetition_penalty,
    }
    if args.rollout_top_k > 0:
        sampling_kwargs["top_k"] = args.rollout_top_k
    metrics, _ = run_rollout_evaluation(
        rollout,
        model,
        tokenizer,
        examples,
        rollout_scorer.score_example,
        rollout_scorer.summarize_records,
        sampling_kwargs,
        args.rollout_max_tokens,
        output_path=Path(args.output_dir) / "rollout_eval" / f"step_{step}.jsonl",
    )
    if torch.distributed.get_rank() != 0:
        return
    print(f"Rollout metrics (step {step}): {json.dumps(metrics, indent=2)}", flush=True)
    if wandb_module is not None:
        log = {}
        for name, value in metrics.items():
            config, _, metric = name.rpartition("/")
            section = task_wandb_section(config)
            wandb_module.define_metric(f"{section}/*", step_metric="eval_step")
            log[f"{section}/{metric}"] = value
        log["eval_step"] = step
        wandb_module.log(log)


def task_name(config):
    """Short task name of a dataset config: "User_Profile_L1_gpt54_MaxLen15360" or "V1_User_Profile_L1_gpt54"
    -> "L1", "V1_User_Profile_L3_Persona_gpt54" -> "L3_Persona"."""
    match = re.search(r"(?:^|_)(L\d+(?:_[A-Za-z]+)?)_gpt54(?:_|$)", config or "") \
        or re.search(r"(?:^|_)(L\d+)(?:_|$)", config or "")
    return match.group(1) if match else config or "all"


def task_wandb_section(config):
    """W&B panel section of a task, shared by its eval perplexity and rollout metrics, e.g. "L3_Persona_Evaluation"."""
    name = task_name(config)
    return f"{'All' if name == 'all' else name}_Evaluation"


def require_linear_attention_kernels():
    """Fail fast if this node would fall back to the slow PyTorch Gated Delta Rule / causal Conv1D path.

    Every rank waits for the slowest one, so a single node without the kernels slows the whole job ~10x.
    """
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    missing = [
        package for package, available in (
            ("flash-linear-attention", is_flash_linear_attention_available()),
            ("causal-conv1d", is_causal_conv1d_available()),
        ) if not available
    ]
    if missing:
        raise RuntimeError(
            f"[{socket.gethostname()} rank {os.environ.get('RANK', '?')}] Qwen3.5 linear-attention fast path "
            f"unavailable: {', '.join(missing)} not importable. Install them on every node (see README "
            "'Qwen3.5 linear-attention fast-path installation') or pass --no-require_linear_attention_kernels."
        )


def main(argument_defaults=None, rollout_scorer=None):
    args = parse_args(argument_defaults)
    if args.require_linear_attention_kernels:
        require_linear_attention_kernels()
    device, ds_config = distributed_config(args)
    model, processor, optimizer = prepare_model(args)
    tokenizer = processor.tokenizer
    train_dataset, eval_dataset = create_dataset(
        args.dataset_name,
        tokenizer,
        args.max_seq_len,
        dataset_configs=args.dataset_configs,
        hf_token=args.hf_token,
        shuffle_seed=args.dataset_shuffle_seed,
        mixing_alpha=args.dataset_mixing_alpha,
    )
    rollout, rollout_examples = None, None
    if args.rollout_eval:
        rollout, rollout_examples = create_rollout_evaluator(args, tokenizer, rollout_scorer)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    if args.pad_to_max_seq_len:
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="max_length", max_length=args.max_seq_len)
    else:
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collator,
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=ConfigIndexCollator(collator),
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True
    )

    args.num_train_steps = int(len(train_dataloader) * args.num_train_epochs)
    args.num_warmup_steps = min(1000, int(args.num_train_steps * 0.1)) if args.num_warmup_steps == -1 else args.num_warmup_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_steps,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )

    cur_rank = torch.distributed.get_rank()
    use_wandb = cur_rank == 0 and args.use_wandb

    if cur_rank == 0:
        print("***** Running Qwen3.5 full-model SFT *****")
        os.makedirs(args.output_dir, exist_ok=True)

    if use_wandb:
        import wandb

        os.environ["WANDB_PROJECT"] = args.wandb_project
        wandb.login(key="3f14084582ffbf0986b305f813aea34ca59c77c5")
        wandb_config = vars(args).copy()
        wandb_config.pop("hf_token", None)
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": wandb_config,
        }
        if args.wandb_run_id:
            init_kwargs.update({"id": args.wandb_run_id, "resume": "allow"})
        wandb.init(**init_kwargs)
        wandb.define_metric("train_step")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("Eval/*", step_metric="eval_step")

    if cur_rank == 0:
        log_sft_template_sample(train_dataset.dataset[0]["messages"], tokenizer)

    if args.do_eval:
        if cur_rank == 0:
            print(f"***** Evaluating perplexity before training *****")
        evaluation_result = evaluation(
            model, eval_dataloader, device, args.max_eval_steps, args.label_logits_only
        )
        initial_ppl = evaluation_result["token_ppl"]
        if cur_rank == 0:
            print(f"Init token ppl: {evaluation_result['token_ppl']}, token loss: {evaluation_result['token_loss']} | "
                  f"batch ppl: {evaluation_result['batch_ppl']}, batch loss: {evaluation_result['batch_loss']}")
        if use_wandb:
            wandb.log({**eval_log(evaluation_result, wandb), "eval_step": 0})
    if rollout is not None:
        rollout_evaluation(args, rollout, rollout_examples, rollout_scorer, model, tokenizer, 0,
                           wandb if use_wandb else None)
    if args.do_eval:
        save_checkpoint(args, model, processor, epoch=0, step=0, ppl=initial_ppl)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        if cur_rank == 0:
            print(f"===== Epoch {epoch + 1}/{args.num_train_epochs} =====")
        model.train()
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch = to_device(batch, device)

            loss = compute_loss(model, batch, args.label_logits_only)

            model.backward(loss)
            model.step()
            global_step += 1

            max_seq_len = None
            if global_step % args.logging_steps == 0:
                # Longest non-padding sequence in the global batch (max over all ranks).
                max_seq_len = batch["attention_mask"].sum(dim=1).max().to(torch.int64)
                torch.distributed.all_reduce(max_seq_len, op=torch.distributed.ReduceOp.MAX)
                max_seq_len = max_seq_len.item()

            if cur_rank == 0 and global_step % args.logging_steps == 0:
                current_loss = loss.item()
                ppl = math.exp(min(20.0, current_loss))
                try:
                    current_lr = model.get_lr()[0]
                except Exception:
                    current_lr = optimizer.param_groups[0]["lr"]
                elapsed_ms = (time.time() - start_time) * 1000
                print(
                    f"epoch {epoch + 1}/{args.num_train_epochs} | "
                    f"step {step + 1}/{len(train_dataloader)} | "
                    f"global-step {global_step} | loss {current_loss:.4f} | "
                    f"ppl {ppl:.3f} | lr {current_lr:.2e} | max_len {max_seq_len} | {elapsed_ms:.0f} ms",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "Train/loss": current_loss,
                        "Train/ppl": ppl,
                        "Train/lr": current_lr,
                        "Train/max_seq_len": max_seq_len,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_step": global_step,
                    })

            is_training_end = (
                epoch + 1 == args.num_train_epochs
                and step + 1 == len(train_dataloader)
            )
            is_checkpoint_step = global_step % args.checkpoint_steps == 0
            if is_checkpoint_step or is_training_end:
                ppl_eval = -1
                if args.do_eval:
                    if cur_rank == 0:
                        print("***** Evaluating perplexity *****")
                    evaluation_result = evaluation(
                        model, eval_dataloader, device, args.max_eval_steps, args.label_logits_only
                    )
                    ppl_eval = evaluation_result["token_ppl"]
                    if cur_rank == 0:
                        print(f"Eval token ppl: {evaluation_result['token_ppl']}, "
                              f"token loss: {evaluation_result['token_loss']} | "
                              f"batch ppl: {evaluation_result['batch_ppl']}, "
                              f"batch loss: {evaluation_result['batch_loss']}")
                    if use_wandb:
                        wandb.log({
                            **eval_log(evaluation_result, wandb),
                            "epoch": epoch,
                            "global_step": global_step,
                            "eval_step": global_step,
                        })
                if rollout is not None:
                    rollout_evaluation(args, rollout, rollout_examples, rollout_scorer, model, tokenizer,
                                       global_step, wandb if use_wandb else None)

                save_checkpoint(args, model, processor, epoch, global_step, ppl_eval)
                torch.cuda.empty_cache()

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
