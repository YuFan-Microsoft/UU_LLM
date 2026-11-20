"""Inference of a Qwen35_profile_classifier.py checkpoint on the EVAL_Dataset test split of yufan/user_profile_dataset.

Downloads only EVAL_Dataset/test-*.parquet (gated dataset: needs HF_TOKEN or `hf auth login`), scores every row with
the trainer's prompt, truncation and class head, then writes to --output_dir:
  metrics.json       accuracy, CE, macro P/R/F1, per-class precision / recall / F1 / support, confusion matrix
  predictions.jsonl  one test row per line with all its original columns (the ground truth stays in "label") plus
                     pred_label:  the predicted class, "no match" / "partial match" / "perfect match"
                     probability: {class: softmax probability} of the three classes
The progress bar (rows done, rows/s, ETA, running accuracy) also goes to <output_dir>/inference.log via the launcher;
every --eval_rows rows (default 1000) the metrics of all rows scored so far are printed.

  accelerate launch --multi_gpu --num_processes 8 Qwen35_profile_classifier_inference.py --checkpoint <step_N dir>
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from Qwen35_profile_classifier import (CLASS_HEAD_FILE, LABELS, FixedLengthCollator, ProfileBehaviorDataset,
                                       answer_token_ids, class_logits, load_classifier, load_split,
                                       require_linear_attention_kernels)

DATASET_REPO = "yufan/user_profile_dataset"
SUBSET = "EVAL_Dataset"


# =========================================================
# Data
# =========================================================
def test_files(data_root):
    return sorted(glob.glob(os.path.join(data_root, SUBSET, "test-*.parquet")))


def download_test_split(data_root):
    """Only <SUBSET>/test-*.parquet; skipped when already under data_root (e.g. downloaded by the trainer)."""
    if not test_files(data_root):
        snapshot_download(DATASET_REPO, repo_type="dataset", allow_patterns=[f"{SUBSET}/test-*.parquet"],
                          local_dir=data_root)
    if not test_files(data_root):
        raise FileNotFoundError(f"No {SUBSET}/test-*.parquet under {data_root}")


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {**self.dataset[i], "index": i}


class BatchLengthCollator:
    """Right-pads each batch to its longest row instead of the trainer's fixed --max_length: with right padding the
    real tokens never see a pad token, so the scores do not change and short batches run faster."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch = FixedLengthCollator(self.pad_token_id, max(len(e["input_ids"]) for e in examples))(examples)
        batch["index"] = torch.tensor([e["index"] for e in examples], dtype=torch.long)
        return batch


# =========================================================
# Inference and metrics
# =========================================================
@torch.no_grad()
def predict(model, loader, accelerator, n, eval_rows=0):
    """[n, 3] class logits in test-row order on the main process (None on the others). Every eval_rows scored rows
    (over all GPUs; 0 = never) prints the metrics of all rows scored so far."""
    all_logits, all_index, all_labels = [], [], []
    done = correct = 0
    progress = tqdm(total=n, desc="Inference", unit="row", disable=not accelerator.is_main_process,
                    dynamic_ncols=True)
    for batch in loader:
        # gather_for_metrics drops the duplicate rows that pad the last batch across GPUs
        logits, index, labels = accelerator.gather_for_metrics(
            (class_logits(model, batch), batch["index"], batch["class_label"]))
        all_logits.append(logits.cpu())
        all_index.append(index.cpu())
        all_labels.append(labels.cpu())
        previous, done = done, done + len(index)
        correct += int((logits.argmax(-1) == labels).sum())
        progress.update(len(index))
        progress.set_postfix(acc=f"{correct / done:.4f}")
        if eval_rows and done // eval_rows > previous // eval_rows and accelerator.is_main_process:
            metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
            progress.write(format_metrics(metrics, f"Eval @ {done}/{n} rows"))
    progress.close()
    if not accelerator.is_main_process:
        return None
    logits, index = torch.cat(all_logits), torch.cat(all_index)
    order = torch.argsort(index)
    if not torch.equal(index[order], torch.arange(n)):
        raise RuntimeError(f"Gathered {len(index)} rows, expected each of the {n} test rows once")
    return logits[order]


def compute_metrics(logits, labels):
    y_true, y_pred = labels.numpy(), logits.argmax(-1).numpy()
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "n": int(len(y_true)),
        "ce": F.cross_entropy(logits, labels).item(),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
                              "support": int(support[i])} for i, label in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def format_metrics(metrics, title="Test"):
    lines = [f"\n===== {title}: n={metrics['n']} CE={metrics['ce']:.4f} Acc={metrics['accuracy']:.4f} "
             f"MacroP={metrics['macro_precision']:.4f} MacroR={metrics['macro_recall']:.4f} "
             f"MacroF1={metrics['macro_f1']:.4f} ====="]
    for label, m in metrics["per_class"].items():
        lines.append(f"{label:>13}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                     f"support={m['support']}")
    lines.append(f"Confusion matrix (rows = true, cols = predicted; order {LABELS}):")
    lines.append(str(np.array(metrics["confusion_matrix"])))
    return "\n".join(lines)


def write_predictions(files, probs, path):
    """Re-reads the parquet files in load_split's order, so row i gets probs[i]; stops after len(probs) rows."""
    row = 0
    with open(path, "w", encoding="utf-8") as out:
        for file in files:
            for record_batch in pq.ParquetFile(file).iter_batches(batch_size=1024):
                for record in record_batch.to_pylist():
                    if row == len(probs):
                        return
                    record["pred_label"] = LABELS[int(probs[row].argmax())]
                    record["probability"] = {label: round(float(p), 6) for label, p in zip(LABELS, probs[row])}
                    out.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    row += 1
    if row != len(probs):
        raise RuntimeError(f"Wrote {row} rows but have {len(probs)} predictions")


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser("Qwen3.5 profile-behavior classifier inference on the EVAL_Dataset test split")
    parser.add_argument("--checkpoint", required=True,
                        help="A step_<N> directory saved by Qwen35_profile_classifier.py (model, tokenizer, "
                             f"{CLASS_HEAD_FILE})")
    parser.add_argument("--data_root", default=str(Path(__file__).resolve().parents[1] / "data"),
                        help=f"Local folder holding {SUBSET}/test-*.parquet (downloaded from {DATASET_REPO} if "
                             "missing); the default is the trainer's ../data")
    parser.add_argument("--output_dir", default="",
                        help="Where metrics.json and predictions.jsonl go (default <checkpoint>/inference_test)")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Same as training: longer prompts have the end of the profile cut")
    parser.add_argument("--max_samples", type=int, default=0, help="Only the first N test rows (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per GPU")
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="Attention of the full-attention layers (flash_attention_2 or sdpa)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU (tokenization)")
    parser.add_argument("--eval_rows", type=int, default=1000,
                        help="Print the metrics of all rows scored so far every N rows (over all GPUs; 0 = only at "
                             "the end)")
    args = parser.parse_args()
    args.gradient_checkpointing = 0  # read by load_classifier
    args.output_dir = args.output_dir or os.path.join(args.checkpoint, "inference_test")
    return args


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    if not os.path.exists(os.path.join(args.checkpoint, CLASS_HEAD_FILE)):
        raise FileNotFoundError(f"{args.checkpoint} has no {CLASS_HEAD_FILE}: not a Qwen35_profile_classifier.py "
                                "checkpoint (its trained head cannot be rebuilt from the lm_head)")

    with accelerator.main_process_first():
        download_test_split(args.data_root)
    files = test_files(args.data_root)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = "right"
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    answer_ids = answer_token_ids(tokenizer)

    test = load_split(os.path.join(args.data_root, SUBSET), "test")
    n = min(args.max_samples, len(test[3])) if args.max_samples else len(test[3])
    dataset = IndexedDataset(ProfileBehaviorDataset(test, tokenizer, args.max_length, np.arange(n)))
    if accelerator.is_main_process:
        print(f"test: {n} of {len(test[3])} rows, {len(test[0])} profiles")
        print("===== First prompt =====\n" + tokenizer.decode(dataset[0]["input_ids"]) + "\n========================")

    require_linear_attention_kernels()
    # Forward only: each GPU holds the whole model, the test batches are sharded across GPUs.
    model = load_classifier(args.checkpoint, answer_ids, args).to(accelerator.device).eval()
    loader = accelerator.prepare(DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                            collate_fn=BatchLengthCollator(pad_token_id),
                                            num_workers=args.num_workers, pin_memory=True))
    logits = predict(model, loader, accelerator, n, args.eval_rows)
    if not accelerator.is_main_process:
        return

    metrics = compute_metrics(logits, torch.from_numpy(test[3][:n]))
    metrics["checkpoint"] = args.checkpoint
    print(format_metrics(metrics))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    predictions_file = os.path.join(args.output_dir, "predictions.jsonl")
    write_predictions(files, F.softmax(logits, dim=-1).numpy(), predictions_file)
    print(f"Saved {args.output_dir}/metrics.json and {predictions_file}")


if __name__ == "__main__":
    main()
