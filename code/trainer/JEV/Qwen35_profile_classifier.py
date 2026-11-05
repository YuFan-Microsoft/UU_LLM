"""Qwen3.5 text-only 3-class classifier on EVAL_Dataset of yufan/user_profile_dataset.

Input: a short instruction + mai_profile + future_behavior (both JSON strings without null keys, as stored on HF).
Output: the next-token logits of "No" / "Partial" / "Perfect" after the non-thinking assistant prompt, i.e. the
labels "no match" / "partial match" / "perfect match" (the same scoring idea as Qwen3VL.py with Bad / Fair / Good).
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, get_cosine_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

LABELS = ["no match", "partial match", "perfect match"]
ANSWERS = ["No", "Partial", "Perfect"]
PROMPT = ("Does the user profile predict the future behavior? Answer No, Partial or Perfect.\n\n"
          "Profile: {profile}\n\nBehavior: {behavior}")
UU_LLM = Path(__file__).resolve().parents[3]
# Git-ignored key file, in UU_LLM or in the directory above it (the job directory UU_LLM was copied into).
WANDB_KEY_FILES = tuple(directory / "wandb_key.txt" for directory in (UU_LLM, UU_LLM.parent))


def wandb_login():
    """WANDB_API_KEY, else the first line of the first existing WANDB_KEY_FILES."""
    import wandb
    key = os.environ.get("WANDB_API_KEY")
    for path in WANDB_KEY_FILES:
        if not key and path.exists() and path.read_text(encoding="utf-8").split():
            key = path.read_text(encoding="utf-8").split()[0]
    if not key:
        raise RuntimeError(f"No W&B key: set WANDB_API_KEY or write it to {WANDB_KEY_FILES[0]} "
                           "(or pass --wandb_project '' to disable W&B)")
    wandb.login(key=key)


# =========================================================
# Data
# =========================================================
def load_split(data_dir, split):
    """Rows of <data_dir>/<split>-*.parquet. Each user's profile repeats on many rows, so profiles are stored once:
    (unique profile strings, profile index per row, behavior string per row, label id per row)."""
    files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No {split}-*.parquet under {data_dir}")
    label_ids = {label: i for i, label in enumerate(LABELS)}
    profile_ids, profile_idx, behaviors, labels = {}, [], [], []
    for path in files:
        table = pq.read_table(path, columns=["mai_profile", "future_behavior", "label"])
        encoded = pc.dictionary_encode(table.column("mai_profile")).combine_chunks()
        local = np.array([profile_ids.setdefault(text, len(profile_ids)) for text in encoded.dictionary.to_pylist()],
                         dtype=np.int64)
        profile_idx.append(local[encoded.indices.to_numpy()])
        behaviors.extend(table.column("future_behavior").to_pylist())
        labels.extend(label_ids[label] for label in table.column("label").to_pylist())
    return list(profile_ids), np.concatenate(profile_idx), behaviors, np.array(labels, dtype=np.int64)


class ProfileBehaviorDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, indices=None):
        self.profiles, self.profile_idx, self.behaviors, self.labels = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.indices = np.arange(len(self.labels)) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def encode(self, profile, behavior):
        messages = [{"role": "user", "content": PROMPT.format(profile=profile, behavior=behavior)}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __getitem__(self, i):
        row = self.indices[i]
        profile, behavior = self.profiles[self.profile_idx[row]], self.behaviors[row]
        input_ids = self.encode(profile, behavior)
        if len(input_ids) > self.max_length:
            # Keep the instruction, the behavior and the answer position; cut the end of the profile.
            profile_ids = self.tokenizer.encode(profile, add_special_tokens=False)
            while len(input_ids) > self.max_length:
                profile_ids = profile_ids[:max(0, len(profile_ids) - (len(input_ids) - self.max_length) - 8)]
                input_ids = self.encode(self.tokenizer.decode(profile_ids), behavior)
        return {"input_ids": input_ids, "class_label": int(self.labels[row])}


class LeftPadCollator:
    """Every batch is left-padded to the same fixed length (--max_length): constant memory, so an OOM shows up at the
    first step. Left padding keeps the answer position at index -1 for every row."""

    def __init__(self, pad_token_id, length):
        self.pad_token_id = pad_token_id
        self.length = length

    def __call__(self, examples):
        length = self.length
        input_ids = torch.full((len(examples), length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
        for row, example in enumerate(examples):
            ids = example["input_ids"]
            input_ids[row, length - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, length - len(ids):] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "class_label": torch.tensor([example["class_label"] for example in examples], dtype=torch.long)}


def subset(n, size, seed):
    if not size or size >= n:
        return None
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


# =========================================================
# Scoring
# =========================================================
def answer_token_ids(tokenizer):
    ids = [tokenizer.encode(answer, add_special_tokens=False) for answer in ANSWERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError(f"Each answer must be one token: {dict(zip(ANSWERS, ids))}")
    return torch.tensor([i[0] for i in ids])


def class_logits(model, batch, answer_ids):
    """[B, 3] logits of No / Partial / Perfect at the last position (lm_head runs on that position only)."""
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
                    logits_to_keep=1)
    return outputs.logits[:, -1, :].index_select(-1, answer_ids).float()


@torch.no_grad()
def evaluate(model, loader, answer_ids, accelerator, desc="Eval"):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc=desc, disable=not accelerator.is_main_process, dynamic_ncols=True):
        logits, labels = accelerator.gather_for_metrics((class_logits(model, batch, answer_ids), batch["class_label"]))
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    model.train()
    if not accelerator.is_main_process:
        return None
    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    preds = logits.argmax(-1)
    y_true, y_pred = labels.numpy(), preds.numpy()
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    metrics = {
        "ce": F.cross_entropy(logits, labels).item(),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1.mean()),
        **{f"{name}/{k}": float(v[i]) for i, name in enumerate(ANSWERS)
           for k, v in (("precision", p), ("recall", r), ("f1", f1))},
    }
    print(f"\n===== {desc}: n={len(y_true)} CE={metrics['ce']:.4f} Acc={metrics['accuracy']:.4f} "
          f"MacroF1={metrics['macro_f1']:.4f} =====")
    for i, name in enumerate(ANSWERS):
        print(f"{name:>8}: P={p[i]:.4f} R={r[i]:.4f} F1={f1[i]:.4f} support={support[i]}")
    print("Confusion matrix (rows = true No/Partial/Perfect, cols = predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))
    return metrics


# =========================================================
# Model
# =========================================================
def require_linear_attention_kernels():
    """Without these kernels the Qwen3.5 linear-attention layers fall back to PyTorch code that is ~10x slower."""
    from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
    missing = [name for name, ok in (("flash-linear-attention", is_flash_linear_attention_available()),
                                     ("causal-conv1d", is_causal_conv1d_available())) if not ok]
    if missing:
        raise RuntimeError(f"Install {' and '.join(missing)} for the Qwen3.5 linear-attention kernels")


def load_model(args):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=not args.eval_only)
    elif args.use_lora and not args.eval_only:
        # Text-only forward never runs the vision tower or the MTP head, so they get no LoRA (unused DDP params).
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules="all-linear", exclude_modules=r".*(visual|mtp).*"))
    return model


def save_model(model, tokenizer, accelerator, args, step, metrics):
    save_dir = os.path.join(args.output_dir, f"step_{step}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(
        save_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model))
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(metrics or {}, f, indent=2)
        print(f"Saved {save_dir}")
    accelerator.wait_for_everyone()


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("Qwen3.5 profile-behavior 3-class classifier")
    parser.add_argument("--model_name_or_path", default="/yufan/open_source_models/Qwen3.5_VLM/Qwen3.5-9B/")
    parser.add_argument("--data_dir", default="../data/EVAL_Dataset",
                        help="Local EVAL_Dataset folder with train-*.parquet and test-*.parquet")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Every batch is padded to this length. The longest EVAL_Dataset row is ~15.1k Qwen3.5 "
                             "tokens; longer profiles are cut at the end")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Random training subset (0 = all rows)")
    parser.add_argument("--eval_samples", type=int, default=5000,
                        help="Evaluate during training on the first N test rows (0 = all), the same rows every time; "
                             "the final eval uses the whole test split. Evaluation is sharded across all GPUs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_path", default="", help="Resume from, or evaluate, a saved LoRA adapter")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate --lora_path (or the base model)")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4, help="Per GPU (no gradient accumulation)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="Attention of the full-attention layers (flash_attention_2 or sdpa)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU (tokenization)")

    parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluate and save every N optimizer steps")
    parser.add_argument("--output_dir", default="../output/qwen35_9B_profile_classifier")
    parser.add_argument("--wandb_project", default="qwen35_9B_profile_classifier",
                        help="W&B project ('' disables W&B); key from WANDB_API_KEY or UU_LLM/wandb_key.txt")
    parser.add_argument("--wandb_name", default="qwen35_9B_profile_classifier")
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16", dataloader_config=DataLoaderConfiguration(non_blocking=True))
    if args.eval_only:
        args.wandb_project = ""
    if args.wandb_project and accelerator.is_main_process:
        wandb_login()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    answer_ids = answer_token_ids(tokenizer).to(accelerator.device)
    collator = LeftPadCollator(pad_token_id, args.max_length)

    test = load_split(args.data_dir, "test")
    full_eval_ds = ProfileBehaviorDataset(test, tokenizer, args.max_length)
    eval_ds = ProfileBehaviorDataset(test, tokenizer, args.max_length,
                                     np.arange(min(args.eval_samples, len(test[3]))) if args.eval_samples else None)
    if accelerator.is_main_process:
        print(f"test: {len(test[3])} rows, {len(test[0])} profiles; eval subset {len(eval_ds)} rows")
        print("===== First prompt =====\n" + tokenizer.decode(full_eval_ds[0]["input_ids"]) + "\n========================")

    require_linear_attention_kernels()
    model = load_model(args)
    if accelerator.is_main_process and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    def make_eval_loader(dataset):
        return DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    if args.eval_only:
        model, full_eval_loader = accelerator.prepare(model, make_eval_loader(full_eval_ds))
        evaluate(model, full_eval_loader, answer_ids, accelerator, desc="Eval (full test)")
        return

    train = load_split(args.data_dir, "train")
    train_ds = ProfileBehaviorDataset(train, tokenizer, args.max_length,
                                      subset(len(train[3]), args.max_train_samples, args.seed))
    if accelerator.is_main_process:
        counts = np.bincount(train[3][train_ds.indices], minlength=3)
        print(f"train: {len(train_ds)} rows, {len(train[0])} profiles; labels "
              + ", ".join(f"{name}={c}" for name, c in zip(ANSWERS, counts)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
                                  weight_decay=args.weight_decay, fused=True)
    model, optimizer, train_loader, eval_loader, full_eval_loader = accelerator.prepare(
        model, optimizer, train_loader, make_eval_loader(eval_ds), make_eval_loader(full_eval_ds))
    total_steps = args.epochs * len(train_loader)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    if accelerator.is_main_process:
        print(f"{len(train_loader)} steps per epoch, global batch {args.batch_size * accelerator.num_processes}")

    if args.wandb_project and accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, name=f"{args.wandb_name}_lr{args.learning_rate}", config=vars(args))

    def log(metrics, prefix):
        if args.wandb_project and accelerator.is_main_process and metrics:
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=global_step)

    global_step = 0
    log(evaluate(model, eval_loader, answer_ids, accelerator, desc="Eval @ step 0"), "eval")
    model.train()
    for epoch in range(args.epochs):
        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", disable=not accelerator.is_main_process,
                        dynamic_ncols=True)
        for batch in train_loader:
            loss = F.cross_entropy(class_logits(model, batch, answer_ids), batch["class_label"])
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
            log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, "train")
            if global_step % args.eval_steps == 0:
                metrics = evaluate(model, eval_loader, answer_ids, accelerator, desc=f"Eval @ step {global_step}")
                log(metrics, "eval")
                save_model(model, tokenizer, accelerator, args, global_step, metrics)
        progress.close()

    metrics = evaluate(model, full_eval_loader, answer_ids, accelerator, desc="Final eval (full test)")
    save_model(model, tokenizer, accelerator, args, global_step, metrics)
    if args.wandb_project and accelerator.is_main_process:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
