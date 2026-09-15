import argparse
import os
import datetime
import math
import time
import torch
import deepspeed
import numpy as np

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
from datasets import load_dataset
from tqdm import tqdm

QWEN3_5_MULTIMODAL_ARCH = "Qwen3_5ForConditionalGeneration"

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

    prompt_ids = tokenizer.apply_chat_template(
        conversation=messages[:-1],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = np.asarray(prompt_ids, dtype=np.int64)

    assistant_prefix = tokenizer.encode(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    response_start = len(prompt_ids) + len(assistant_prefix)
    response_end = int(np.flatnonzero(input_ids == tokenizer.eos_token_id)[-1])
    labels[response_start:response_end + 1] = input_ids[response_start:response_end + 1]

    if max_seq_length and len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    attention_mask = np.ones_like(input_ids)
    return input_ids, labels, attention_mask

class LLMDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, data_slice) -> None:
        super().__init__()
        self.dataset = dataset[data_slice]
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.print = 0

    def __len__(self):
        length = len(self.dataset)
        return length

    def __getitem__(self, idx):
        messages = self.dataset[idx]["messages"]
        input_ids, labels, _ = encode_sft_example(messages, self.tokenizer, self.max_seq_len)
        return {"input_ids": input_ids.tolist(),
            "labels": labels.tolist()}

def create_dataset(dataset_name,
                   tokenizer,
                   max_seq_len):
    dataset = load_dataset(dataset_name)
    train_llm_dataset = LLMDataset(dataset, tokenizer, max_seq_len, "train")
    test_llm_dataset = LLMDataset(dataset, tokenizer, max_seq_len, "test")
    return train_llm_dataset, test_llm_dataset

def get_train_ds_config(stage=3):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
        "reduce_bucket_size": 1e6
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
    required_prefixes = ("model.visual.", "model.language_model.")
    missing_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in parameter_names)
    ]
    if missing_prefixes:
        raise RuntimeError(
            "Refusing to save an incomplete Qwen3.5 checkpoint; missing "
            f"parameter prefixes: {missing_prefixes}"
        )
    if not hasattr(model, "lm_head"):
        raise RuntimeError("Refusing to save a Qwen3.5 checkpoint without an LM head")
    frozen_parameters = [name for name, param in named_parameters if not param.requires_grad]
    if frozen_parameters:
        raise RuntimeError(
            "Full-model SFT requires every parameter to be trainable; frozen "
            f"parameters include: {frozen_parameters[:10]}"
        )


def save_zero_three_model(model, processor, save_dir):
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
        os.makedirs(save_dir, exist_ok=True)
        model_to_save.save_pretrained(save_dir, state_dict=output_state_dict)
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
        os.makedirs(save_dir, exist_ok=True)
        model_to_save.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
    torch.distributed.barrier()

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def parse_args():
    parser = argparse.ArgumentParser(description="Train and save a full Qwen3.5 multimodal model")

    parser.add_argument('--use_wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_project', type=str, default="qwen3_5_text_training")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--do_eval', type=int, default=1)

    parser.add_argument('--dataset_name', type=str, default='yufan/UltraData-SFT-2605-Chinese')
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')

    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--num_train_steps', type=int, default=-1)
    parser.add_argument('--num_warmup_steps', type=int, default=-1)
    parser.add_argument('--checkpoints_every_epoch', type=int, default=2)
    parser.add_argument('--logging_steps', type=int, default=1)

    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--max_seq_len', type=int, default=8192)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--zero_stage', type=int, default=3)
    parser.add_argument('--gradient_checkpointing', default=True)
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default='cosine', choices=['linear', 'cosine'])

    parser.add_argument("--global_rank", type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
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
    )
    model.config.architectures = [QWEN3_5_MULTIMODAL_ARCH]
    model.config.use_cache = False
    validate_full_multimodal_model(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
    return model, processor, optimizer


def evaluation(model, eval_dataloader, device):
    model.eval()
    losses = 0
    step = 0
    progress_bar = tqdm(
        eval_dataloader,
        desc="Evaluating",
        disable=torch.distributed.get_rank() != 0,
        dynamic_ncols=True,
    )
    for step, batch in enumerate(progress_bar):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        losses += loss.float()
    losses = losses / (step + 1)
    try:
        losses = get_all_reduce_mean(losses)
    except:
        pass
    try:
        ppl = torch.exp(losses).item()
    except OverflowError:
        ppl = float("inf")
    model.train()
    return ppl, losses.item()

def main():
    args = parse_args()
    device, ds_config = distributed_config(args)
    model, processor, optimizer = prepare_model(args)
    tokenizer = processor.tokenizer
    train_dataset, eval_dataset = create_dataset(
        args.dataset_name,
        tokenizer,
        args.max_seq_len,
    )

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        sampler=train_sampler,
        batch_size=args.per_device_train_batch_size,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True
    )

    args.num_train_steps = int(len(train_dataloader) * args.num_train_epochs)
    args.checkpoint_steps = max(1, math.ceil(len(train_dataloader) / args.checkpoints_every_epoch))
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
        init_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
        }
        if args.wandb_run_id:
            init_kwargs.update({"id": args.wandb_run_id, "resume": "allow"})
        wandb.init(**init_kwargs)
        wandb.define_metric("train_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    if args.do_eval:
        if cur_rank == 0:
            print(f"***** Evaluating perplexity before training *****")
        ppl, loss = evaluation(model, eval_dataloader, device)
        if cur_rank == 0:
            print(f"Init ppl: {ppl}, loss: {loss}")
        if use_wandb:
            wandb.log({"eval/loss": loss, "eval/ppl": ppl, "eval_step": 0})

    global_step = 0
    for epoch in range(args.num_train_epochs):
        if cur_rank == 0:
            print(f"===== Epoch {epoch + 1}/{args.num_train_epochs} =====")
        model.train()

        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch = to_device(batch, device)

            outputs = model(**batch, use_cache=False)
            loss = outputs.loss

            model.backward(loss)
            model.step()
            global_step += 1

            if cur_rank == 0 and step % args.logging_steps == 0:
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
                    f"ppl {ppl:.3f} | lr {current_lr:.2e} | {elapsed_ms:.0f} ms",
                    flush=True,
                )
                if use_wandb:
                    wandb.log({
                        "train/loss": current_loss,
                        "train/ppl": ppl,
                        "train/lr": current_lr,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_step": global_step,
                    })

            is_epoch_end = step + 1 == len(train_dataloader)
            if (step + 1) % args.checkpoint_steps == 0 or is_epoch_end:
                ppl_eval = -1
                if args.do_eval:
                    if cur_rank == 0:
                        print("***** Evaluating perplexity *****")
                    ppl_eval, eval_loss = evaluation(model, eval_dataloader, device)
                    if cur_rank == 0:
                        print(f"Eval ppl: {ppl_eval}, loss: {eval_loss}")
                    if use_wandb:
                        wandb.log({
                            "eval/loss": eval_loss,
                            "eval/ppl": ppl_eval,
                            "epoch": epoch,
                            "global_step": global_step,
                            "eval_step": global_step,
                        })

                save_checkpoint(args, model, processor, epoch, global_step, ppl_eval)
                torch.cuda.empty_cache()

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
