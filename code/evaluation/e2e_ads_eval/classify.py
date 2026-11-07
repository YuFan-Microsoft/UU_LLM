"""Step 3 of the e2e ads eval: each assessment row (user profile, future behavior) scored by the profile classifier.

Each user's first --top_k (10) commercial interests of commercial.jsonl (step 2, sorted by confidence) become the
mai_profile string of EVAL_Dataset, the classifier's training input, and every row of rows.jsonl (step 1) is scored by
a ../../trainer/JEV/Qwen35_profile_classifier.py checkpoint with its prompt, truncation and class head. Every
--report_rows rows (over all GPUs) the share of each predicted label so far is printed. Writes to --output_dir:
  predictions.jsonl  {"userid", "future_behavior", "pred_label", "probability": {label: p}} per row
  profiles.jsonl     {"userid", "n_commercial_interests", "mai_profile"} per user
  summary.json       predicted label counts / shares and the run settings

  accelerate launch --multi_gpu --num_processes 8 classify.py --checkpoint <classifier step_N dir> \
      --commercial_file commercial.jsonl --rows_file rows.jsonl --output_dir <dir>
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trainer" / "JEV"))

PROFILE_FIELDS = ("Brand", "Product", "Retailer")


# =========================================================
# Data
# =========================================================
def mai_profile(interests):
    """EVAL_Dataset's mai_profile string of these commercial interests: every Category; then per interest name its
    Brand, Product and Retailer items and the bare {"Interest": name}; then every Query and every Topic. Each group is
    sorted by the JSON text of its items, as EVAL_Dataset is ("Canva AI" before "Canva": a space sorts before the
    closing quote). Values are kept as written (EVAL_Dataset keeps e.g. " Crop Knit"); empty ones are skipped."""
    def text(value):
        return value if isinstance(value, str) and value.strip() else ""

    def values(key, group):
        return {text(v) for interest in group for v in interest.get(key) or []} - {""}

    def ordered(items):
        return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False))

    items = ordered({"Category": c} for c in {text(i.get("Category")) for i in interests} - {""})
    for bare in ordered({"Interest": name} for name in {text(i.get("Interest")) for i in interests} - {""}):
        group = [i for i in interests if text(i.get("Interest")) == bare["Interest"]]
        for key in PROFILE_FIELDS:
            items += ordered({**bare, key: value} for value in values(key, group))
        items.append(bare)
    items += ordered({"Query": q} for q in values("Query", interests))
    items += ordered({"Topic": t} for t in values("Topic", interests))
    return json.dumps(items, ensure_ascii=False)


def read_jsonl(path, desc):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in tqdm(f, desc=desc, unit="line", dynamic_ncols=True) if line.strip()]


def load_profiles(commercial_file, rows_file, top_k):
    """(profile strings, user ids, profile index per row, behaviors, rows, n interests per user, users missing from
    commercial_file). A missing user (no usable past signal) gets the empty profile "[]"."""
    commercial = {}
    for record in read_jsonl(commercial_file, "Reading commercial profiles"):
        interests = record["interests"]
        if interests and "Interest" not in interests[0]:
            raise ValueError(f"{commercial_file} has no interest names: write it with commercial_profiles.py, not "
                             "../../inference/run_commercial_inference.py")
        commercial[record["user_id"]] = interests[:top_k]
    rows = read_jsonl(rows_file, "Reading assessment rows")
    user_ids = list(dict.fromkeys(row["userid"] for row in rows))
    profiles = [mai_profile(commercial.get(user_id, []))
                for user_id in tqdm(user_ids, desc="Building mai_profile", unit="user", dynamic_ncols=True)]
    position = {user_id: i for i, user_id in enumerate(user_ids)}
    profile_idx = np.array([position[row["userid"]] for row in rows], dtype=np.int64)
    counts = [len(commercial.get(user_id, [])) for user_id in user_ids]
    missing = sum(user_id not in commercial for user_id in user_ids)
    return profiles, user_ids, profile_idx, [row["future_behavior"] for row in rows], rows, counts, missing


# =========================================================
# Inference
# =========================================================
def format_ratios(counts, title, labels):
    total = max(sum(counts), 1)
    return f"[{title}] " + "  ".join(f"{label}: {c} ({c / total:.2%})" for label, c in zip(labels, counts))


def predict(model, loader, accelerator, n, report_rows, labels):
    """[n, 3] softmax probabilities in row order on the main process (None on the others). Every report_rows rows
    (over all GPUs; 0 = never) prints the share of each predicted label so far."""
    import torch
    import torch.nn.functional as F
    from Qwen35_profile_classifier import class_logits

    all_probs, all_index = [], []
    counts = torch.zeros(len(labels), dtype=torch.long)
    done = 0
    progress = tqdm(total=n, desc="Classifier", unit="row", disable=not accelerator.is_main_process,
                    dynamic_ncols=True)
    with torch.no_grad():
        for batch in loader:
            # gather_for_metrics drops the duplicate rows that pad the last batch across GPUs
            logits, index = accelerator.gather_for_metrics((class_logits(model, batch), batch["index"]))
            probs = F.softmax(logits.float(), dim=-1).cpu()
            all_probs.append(probs)
            all_index.append(index.cpu())
            counts += torch.bincount(probs.argmax(-1), minlength=len(labels))
            previous, done = done, done + len(index)
            progress.update(len(index))
            progress.set_postfix({label.split()[0]: f"{c / done:.1%}" for label, c in zip(labels, counts.tolist())})
            if report_rows and done // report_rows > previous // report_rows and accelerator.is_main_process:
                progress.write(format_ratios(counts.tolist(), f"{done}/{n} rows", labels))
    progress.close()
    if not accelerator.is_main_process:
        return None
    probs, index = torch.cat(all_probs), torch.cat(all_index)
    order = torch.argsort(index)
    if not torch.equal(index[order], torch.arange(n)):
        raise RuntimeError(f"Gathered {len(index)} rows, expected each of the {n} rows once")
    return probs[order]


# =========================================================
# Main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True,
                        help="A step_<N> directory saved by ../../trainer/JEV/Qwen35_profile_classifier.py")
    parser.add_argument("--commercial_file", required=True, help="commercial_profiles.py output")
    parser.add_argument("--rows_file", required=True, help="prepare_data.py rows.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile_checkpoint", default="", help="Only recorded in summary.json")
    parser.add_argument("--top_k", type=int, default=10, help="Commercial interests per user, first by position")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Same as training: longer prompts have the end of the profile cut")
    parser.add_argument("--max_samples", type=int, default=0, help="Only the first N rows (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per GPU")
    parser.add_argument("--attn_implementation", default="flash_attention_2",
                        help="Attention of the full-attention layers (flash_attention_2 or sdpa)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU (tokenization)")
    parser.add_argument("--report_rows", type=int, default=1000,
                        help="Print the predicted label shares every N rows (over all GPUs; 0 = only at the end)")
    args = parser.parse_args()
    args.gradient_checkpointing = 0  # read by load_classifier
    return args


def main():
    args = parse_args()
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    from Qwen35_profile_classifier import (CLASS_HEAD_FILE, LABELS, ProfileBehaviorDataset, answer_token_ids,
                                           load_classifier, require_linear_attention_kernels)
    from Qwen35_profile_classifier_inference import BatchLengthCollator, IndexedDataset

    accelerator = Accelerator(mixed_precision="bf16")
    main_process = accelerator.is_main_process
    if not os.path.exists(os.path.join(args.checkpoint, CLASS_HEAD_FILE)):
        raise FileNotFoundError(f"{args.checkpoint} has no {CLASS_HEAD_FILE}: not a Qwen35_profile_classifier.py "
                                "checkpoint")
    profiles, user_ids, profile_idx, behaviors, rows, counts, missing = load_profiles(
        args.commercial_file, args.rows_file, args.top_k)
    n = min(args.max_samples, len(rows)) if args.max_samples else len(rows)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.padding_side = "right"
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    # No labels: class_label is a placeholder the collator carries and nothing reads.
    data = (profiles, profile_idx, behaviors, np.zeros(len(rows), dtype=np.int64))
    dataset = IndexedDataset(ProfileBehaviorDataset(data, tokenizer, args.max_length, np.arange(n)))
    if main_process:
        print(f"{n} of {len(rows)} rows, {len(user_ids)} users ({missing} without a profile, scored with \"[]\"); "
              f"commercial interests per user (first {args.top_k}): mean {np.mean(counts):.2f}, "
              f"empty {sum(c == 0 for c in counts)}")
        print("===== First prompt =====\n" + tokenizer.decode(dataset[0]["input_ids"]) + "\n========================")

    require_linear_attention_kernels()
    # Forward only: each GPU holds the whole model, the rows are sharded across GPUs.
    model = load_classifier(args.checkpoint, answer_token_ids(tokenizer), args).to(accelerator.device).eval()
    loader = accelerator.prepare(DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                            collate_fn=BatchLengthCollator(pad_token_id),
                                            num_workers=args.num_workers, pin_memory=True))
    probs = predict(model, loader, accelerator, n, args.report_rows, LABELS)
    if not main_process:
        return

    pred = probs.argmax(-1)
    label_counts = torch.bincount(pred, minlength=len(LABELS)).tolist()
    print(format_ratios(label_counts, f"Final, {n} rows", LABELS))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for i in tqdm(range(n), desc="Writing predictions", unit="row", dynamic_ncols=True):
            out.write(json.dumps({**rows[i], "pred_label": LABELS[int(pred[i])],
                                  "probability": {label: round(float(p), 6) for label, p in zip(LABELS, probs[i])}},
                                 ensure_ascii=False) + "\n")
    with (out_dir / "profiles.jsonl").open("w", encoding="utf-8") as out:
        for user_id, profile, count in tqdm(list(zip(user_ids, profiles, counts)), desc="Writing profiles",
                                            unit="user", dynamic_ncols=True):
            out.write(json.dumps({"userid": user_id, "n_commercial_interests": count, "mai_profile": profile},
                                 ensure_ascii=False) + "\n")
    summary = {"n_rows": n, "n_users": len(user_ids), "users_without_profile": missing,
               "users_with_empty_profile": sum(c == 0 for c in counts), "top_k": args.top_k,
               "label_counts": dict(zip(LABELS, label_counts)),
               "label_ratios": {label: c / max(n, 1) for label, c in zip(LABELS, label_counts)},
               "classifier_checkpoint": args.checkpoint, "profile_checkpoint": args.profile_checkpoint}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_dir}/predictions.jsonl, profiles.jsonl and summary.json")


if __name__ == "__main__":
    main()
